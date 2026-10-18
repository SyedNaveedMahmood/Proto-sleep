"""Truthful local A/B/C/D evidence report; never exports waveforms or fits data."""
import torch

from .audit import audit_group, jaccard_distance, selected_set


def vector(value):
    return value.detach().cpu().tolist()


@torch.no_grad()
def local_evidence_report(model, x, groups, *, donor_specs, transformations=None,
                          donor_spread_threshold=.25, numerical_tolerance=1e-5):
    """Groups and transformations must be declared before inspecting effects.

    A score, B individual waveform effects, C joint effects, D stability remain
    separate. Descriptive warnings do not establish physiology or causal truth.
    Source identities can be joined through the separate guarded source ledger.
    """
    if not groups or not donor_specs:
        raise ValueError("predeclared groups and donors required")
    original = model(x, return_evidence=True)
    report = {"physical_input_sha256": original.physical_input_sha256[0], "channel": "EEG Fpz-Cz", "units": "uV",
              "groups": {}, "D_attribution_stability": {},
              "interpretation": "Donor-conditional model dependence; internal paths are not physiological counterfactuals."}
    for name, intervals in groups.items():
        donors = []
        for spec in donor_specs:
            group = audit_group(model, x, intervals, **spec)
            individuals = [audit_group(model, x, [interval], **spec) for interval in intervals]
            flags = {"additive_magnitude_disagrees_with_full": bool((group.additive_contribution - group.full_effect).abs().max() > numerical_tolerance),
                     "reselection_material": bool(group.reselection_effect.abs().max() > numerical_tolerance),
                     "selection_membership_changed": group.diagnostics["selection_membership_changed"],
                     "replacement_unsupported": any(not d["elementary_guard_passed"] for d in group.diagnostics["replacements"]),
                     "individual_necessity_not_established": bool((group.full_effect.abs() - torch.stack([a.full_effect.abs() for a in individuals]).max(0).values).max() > donor_spread_threshold)}
            donors.append({"donor": spec["strategy"], "intervals": group.intervals,
                           "A_additive_contribution": vector(group.additive_contribution),
                           "B_individual_counterfactual_effects": [vector(a.full_effect) for a in individuals],
                           "C_group_counterfactual_effect": vector(group.full_effect),
                           "internal_fixed_witness_effect": vector(group.fixed_witness_effect),
                           "internal_event_path_effect": vector(group.fixed_event_path_effect),
                           "background_change": vector(group.background_effect),
                           "reselection_change": vector(group.reselection_effect),
                           "internal_score_deletion_effect": vector(group.score_deletion_effect),
                           "reconciliation_residual": vector(group.reconciliation_residual),
                           "flags": flags, "diagnostics": group.diagnostics})
        effects = torch.tensor([d["C_group_counterfactual_effect"] for d in donors])
        spread = effects.max(0).values - effects.min(0).values
        report["groups"][name] = {"donors": donors, "donor_spread_per_stage": vector(spread),
                                   "donor_sensitive": bool(spread.max() > donor_spread_threshold),
                                   "physiological_validity": "NOT_ESTABLISHED"}
    if transformations is None:
        report["D_attribution_stability"] = {"status": "NOT_TESTED"}
    else:
        for name, changed in transformations.items():
            after = model(changed, return_evidence=True)
            report["D_attribution_stability"][name] = {
                "witness_set_jaccard_distance": jaccard_distance(selected_set(original), selected_set(after)),
                "event_total_change": vector(original.contributions[0].sum(0) - after.contributions[0].sum(0)),
                "logit_change": vector(original.logits[0] - after.logits[0]),
                "selection_before": sorted(selected_set(original)), "selection_after": sorted(selected_set(after))}
    return report

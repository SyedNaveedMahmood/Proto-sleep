"""Separate additive bookkeeping, realizable waveform changes and internal paths."""
from dataclasses import dataclass
import torch

from mist_cove.model import spectral_features
from mist_cove.audit import validate_replacement
from .model import WitnessSet


def union_intervals(intervals):
    checked = []
    for lo, hi in intervals:
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 3000 or hi - lo < 40:
            raise ValueError("invalid physical half-open replacement interval")
        checked.append((lo, hi))
    if not checked:
        raise ValueError("a predeclared nonempty group is required")
    result = []
    for lo, hi in sorted(checked):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def replace_group(model, x, intervals, *, strategy="bridge", donor_interval=None, seed=0, blend_samples=20):
    if len(x) != 1:
        raise ValueError("audit one physical epoch at a time")
    intervals = union_intervals(intervals)
    original = model.physical_uV(x)
    changed = original.clone()
    if type(blend_samples) is not int or blend_samples < 1 or any(2 * blend_samples > hi - lo for lo, hi in intervals):
        raise ValueError("invalid boundary blend length")
    if strategy not in {"bridge", "background", "noise", "phase_scramble"}:
        raise ValueError("unregistered donor strategy")
    if strategy == "background":
        if donor_interval is None:
            raise ValueError("explicit background donor required")
        ds, de = donor_interval
        if (type(ds) is not int or type(de) is not int or not 0 <= ds < de <= 3000
                or any(ds < hi and de > lo for lo, hi in intervals)):
            raise ValueError("background donor must be disjoint from the whole group")
        donor_background = original[0, 0, ds:de]
        if donor_background.std(unbiased=False) > 3.:
            raise ValueError("background donor exceeds fixed synthetic compatibility bound")
    elif donor_interval is not None:
        raise ValueError("donor interval only applies to the background strategy")
    g = torch.Generator().manual_seed(seed)
    diagnostics = []
    for lo, hi in intervals:
        old = original[0, 0, lo:hi]
        bridge = old[0] + torch.linspace(0, 1, hi - lo, device=x.device) * (old[-1] - old[0])
        if strategy == "bridge":
            donor = bridge
        elif strategy == "background":
            # Repeat only the explicit background interval for unioned long groups.
            donor = donor_background.repeat((hi - lo + len(donor_background) - 1) // len(donor_background))[:hi - lo]
            donor = donor - donor.mean() + bridge
        elif strategy == "noise":
            # Known synthetic noise scale only; no fitting from any dataset.
            donor = torch.randn(hi - lo, generator=g).to(x.device) + bridge
        else:
            spectrum = torch.fft.rfft(old - old.mean())
            phase = torch.rand(len(spectrum), generator=g).to(x.device) * (2 * torch.pi)
            phase[0] = 0
            if (hi - lo) % 2 == 0:
                phase[-1] = 0
            donor = torch.fft.irfft(spectrum.abs() * torch.exp(1j * phase), n=hi - lo) + old.mean()
        weight = torch.ones_like(old)
        weight[:blend_samples] = torch.linspace(0, 1, blend_samples, device=x.device)
        weight[-blend_samples:] = torch.linspace(1, 0, blend_samples, device=x.device)
        replacement = old * (1 - weight) + donor * weight
        try:
            guards = validate_replacement(old, replacement)
            guards["elementary_guard_passed"] = True
        except ValueError as error:
            # Retain finite artifact failures for W4, visibly unsupported.
            if not torch.isfinite(replacement).all():
                raise
            guards = {"elementary_guard_passed": False, "reason": str(error)}
        pre_band, _, _ = spectral_features(old[None])
        post_band, _, _ = spectral_features(replacement[None])
        boundary_before, boundary_after = [], []
        if lo > 0:
            boundary_before.append(float((old[0] - original[0, 0, lo - 1]).abs()))
            boundary_after.append(float((replacement[0] - original[0, 0, lo - 1]).abs()))
        if hi < 3000:
            boundary_before.append(float((old[-1] - original[0, 0, hi]).abs()))
            boundary_after.append(float((replacement[-1] - original[0, 0, hi]).abs()))
        diagnostics.append({**guards, "interval": [lo, hi],
                            "boundary_jump_before_uV": max(boundary_before, default=0.),
                            "boundary_jump_after_uV": max(boundary_after, default=0.),
                            "energy_mismatch_relative": float((replacement.square().mean() - old.square().mean()).abs() / old.square().mean().clamp_min(1e-8)),
                            "spectral_fraction_L1": float((pre_band - post_band).abs().sum()),
                            "physiological_validity": "NOT_ESTABLISHED"})
        changed[0, 0, lo:hi] = replacement
    return (changed - model.mean_uV) / model.std_uV, intervals, diagnostics


@dataclass
class GroupAudit:
    intervals: list
    additive_contribution: torch.Tensor
    full_effect: torch.Tensor
    fixed_witness_effect: torch.Tensor
    fixed_event_path_effect: torch.Tensor
    background_effect: torch.Tensor
    reselection_effect: torch.Tensor
    score_deletion_effect: torch.Tensor
    reconciliation_residual: torch.Tensor
    selected_before: list
    selected_after: list
    diagnostics: dict


@torch.no_grad()
def audit_group(model, x, intervals, **replacement_kwargs):
    if model.training:
        raise ValueError("evidence audit requires eval mode")
    changed, intervals, diagnostics = replace_group(model, x, intervals, **replacement_kwargs)
    before = model(x, return_evidence=True)
    after = model(changed, return_evidence=True)
    fixed = model(changed, return_evidence=True, witnesses=WitnessSet.from_evidence(before))
    intersects = torch.zeros_like(before.valid[0])
    for lo, hi in intervals:
        intersects |= (before.event_intervals[0, :, 0] < hi) & (before.event_intervals[0, :, 1] > lo) & before.valid[0]
    additive = before.contributions[0, intersects].sum(0)
    full = before.logits[0] - after.logits[0]
    background = before.background_logits[0] - after.background_logits[0]
    event_path = before.contributions[0].sum(0) - fixed.contributions[0].sum(0)
    reselection = fixed.contributions[0].sum(0) - after.contributions[0].sum(0)
    fixed_total = before.logits[0] - fixed.logits[0]
    # Internal score deletion is separate from any waveform replacement.
    score_deleted = before.background_logits[0] + before.contributions[0, ~intersects].sum(0)
    score_deletion = before.logits[0] - score_deleted
    old_set = before.event_intervals[0, before.valid[0]].tolist()
    new_set = after.event_intervals[0, after.valid[0]].tolist()
    return GroupAudit(intervals, additive, full, fixed_total, event_path, background, reselection,
                      score_deletion, full - (background + event_path + reselection), old_set, new_set,
                      {"donor_strategy": replacement_kwargs.get("strategy", "bridge"),
                       "group_size_after_union": len(intervals), "selection_changed": old_set != new_set,
                       "selection_membership_changed": {tuple(v) for v in old_set} != {tuple(v) for v in new_set},
                       "selection_order_changed": old_set != new_set,
                       "probabilities_before": before.logits[0].softmax(-1).tolist(),
                       "probabilities_after": after.logits[0].softmax(-1).tolist(),
                       "replacements": diagnostics, "internal_paths_are_physiological_counterfactuals": False,
                       "group_membership": "caller_predeclared_not_effect_optimized"})


def selected_set(evidence, offset=0):
    return {tuple(int(v) - offset for v in interval) for interval in evidence.event_intervals[0, evidence.valid[0]].tolist()}


def jaccard_distance(first, second):
    return 1 - len(first & second) / len(first | second) if first or second else 0.

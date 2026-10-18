"""Confirmatory W1–W4 experiments with distinct internal/waveform estimands."""
from dataclasses import replace
import math
import resource
import time

import torch

from mist_cove.experiment import metrics
from mist_cove.model import CoveConfig
from .audit import audit_group, jaccard_distance, selected_set
from .model import METHODS, make_model
from .provenance import digest, environment
from .synthetic import make_fixtures, training_batch, transformed


def validate_config(c):
    if c.get("version") not in {"mist-cove-phase2-synthetic-v1", "mist-cove-phase2-synthetic-v2"} or c.get("scope") != "SYNTHETIC_ONLY":
        raise ValueError("Phase II synthetic configuration required")
    expected_generator = "known-components-phase2-v2" if c["version"].endswith("v2") else "known-components-phase2-v1"
    if c["generator"]["version"] != expected_generator:
        raise ValueError("generator and protocol versions disagree")
    if any(c.get(key) is not False for key in ("real_data_access", "reserved_access", "external_access")):
        raise ValueError("real data access is prohibited")
    if c["seeds"] != [123, 456, 789] or c["methods"] != list(METHODS) or c["tasks"] != ["substitutable", "complementary"]:
        raise ValueError("required seeds/methods/tasks changed")
    for key in ("train_size", "evaluation_size", "optimizer_steps", "audit_examples_per_case", "threads"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError("positive synthetic counts required")
    if any(c[k] < 5 or c[k] % 5 for k in ("train_size", "evaluation_size")):
        raise ValueError("balanced five-class synthetic sets required")
    if c["donors"] != ["bridge", "background", "noise"] or c["negative_control_donor"] != "phase_scramble":
        raise ValueError("predeclared donors required")
    if any(not math.isfinite(v) or v < 0 for v in c["criteria"].values()):
        raise ValueError("invalid falsification thresholds")
    CoveConfig(**c["model"])


def fit(c, seed, task, method):
    torch.manual_seed(seed)
    model = make_model(method, CoveConfig(**c["model"]), rank_bin_uV=c["rank_bin_uV"])
    x, y = training_batch(c["train_size"], seed + 1000, task, c["generator"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    initial_loss = None
    for step in range(c["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite synthetic loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite synthetic gradient")
        optimizer.step()
        initial_loss = float(loss.detach()) if initial_loss is None else initial_loss
    return model.eval(), {"optimizer_steps": step + 1, "example_exposures": (step + 1) * len(x),
                          "initial_loss": initial_loss, "final_loss": float(loss.detach()),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "active_parameters": sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)}


def mean(values):
    return sum(values) / len(values)


@torch.no_grad()
def audit_case(model, fixtures, c, seed):
    records = []
    donor_spreads = []
    phase_controls = []
    reordered_effect_changes = []
    for i, row in enumerate(fixtures[:c["audit_examples_per_case"]]):
        erasers = []
        for donor in c["donors"]:
            kwargs = {"strategy": donor, "seed": seed + 4000 + i, "blend_samples": c["blend_samples"]}
            if donor == "background":
                kwargs["donor_interval"] = row.donor_interval
            group = audit_group(model, row.x, row.diagnostic_intervals, **kwargs)
            individuals = [audit_group(model, row.x, [interval], **kwargs) for interval in row.diagnostic_intervals]
            s = row.label
            erasers.append(float(group.full_effect[s]))
            records.append({"donor": donor, "group": group, "individuals": individuals, "stage": s})
        donor_spreads.append(max(erasers) - min(erasers))
        phase = audit_group(model, row.x, row.diagnostic_intervals, strategy="phase_scramble", seed=seed + i)
        phase_controls.append(float(phase.full_effect[row.label]))
        if len(row.diagnostic_intervals) >= 2:
            (a, b), (d, e) = row.diagnostic_intervals[:2]
            if b <= d or e <= a:
                swap = row.x.clone()
                swap[..., a:b] = row.x[..., d:e]; swap[..., d:e] = row.x[..., a:b]
                old = audit_group(model, row.x, row.diagnostic_intervals)
                changed = audit_group(model, swap, row.diagnostic_intervals)
                reordered_effect_changes.append(float((old.full_effect - changed.full_effect).abs().mean()))
    keys = ("additive_contribution", "full_effect", "fixed_witness_effect", "fixed_event_path_effect",
            "background_effect", "reselection_effect", "score_deletion_effect")
    result = {"count": len(records), "signed_stage_vector_means": {
        key: torch.stack([getattr(r["group"], key) for r in records]).mean(0).tolist() for key in keys}}
    result.update({"mean_absolute_reselection_true_stage": mean([abs(float(r["group"].reselection_effect[r["stage"]])) for r in records]),
                   "max_reconciliation_error": max(float(r["group"].reconciliation_residual.abs().max()) for r in records),
                   "max_score_deletion_accounting_error": max(float((r["group"].additive_contribution - r["group"].score_deletion_effect).abs().max()) for r in records),
                   "group_effect_abs_mean": mean([abs(float(r["group"].full_effect[r["stage"]])) for r in records]),
                   "individual_effect_abs_mean": mean([mean([abs(float(a.full_effect[r["stage"]])) for a in r["individuals"]]) for r in records]),
                   "additive_vs_full_MAE": mean([float((r["group"].additive_contribution - r["group"].full_effect).abs().mean()) for r in records]),
                   "fixed_path_vs_full_MAE": mean([float((r["group"].fixed_event_path_effect - r["group"].full_effect).abs().mean()) for r in records]),
                   "selection_change_fraction": mean([float(r["group"].diagnostics["selection_changed"]) for r in records]),
                   "joint_replacement_confidence_mean": mean([max(r["group"].diagnostics["probabilities_after"]) for r in records]),
                   "joint_replacement_latent_class_accuracy": mean([float(max(range(5), key=lambda s: r["group"].diagnostics["probabilities_after"][s]) == r["stage"]) for r in records]),
                   "eraser_donor_spread_mean": mean(donor_spreads),
                   "phase_preserving_negative_control_delta_mean": mean(phase_controls),
                   "reordered_group_full_effect_MAE": mean(reordered_effect_changes) if reordered_effect_changes else None})
    diagnostics = [d for r in records for d in r["group"].diagnostics["replacements"]]
    result["replacement_diagnostics"] = {
        "elementary_guard_failure_count": sum(not d["elementary_guard_passed"] for d in diagnostics),
        "max_boundary_jump_change_uV": max(abs(d["boundary_jump_after_uV"] - d["boundary_jump_before_uV"]) for d in diagnostics),
        "energy_mismatch_relative_mean": mean([d["energy_mismatch_relative"] for d in diagnostics]),
        "spectral_fraction_L1_mean": mean([d["spectral_fraction_L1"] for d in diagnostics]),
        "physiological_validity": "NOT_ESTABLISHED"}
    return result


@torch.no_grad()
def stability(model, fixtures, c, seed):
    results = {name: [] for name in ("noise", "shift", "physical_gain", "calibrated_gain", "rank_jitter")}
    for i, row in enumerate(fixtures):
        x = row.x
        before = model(x, return_evidence=True)
        for name in results:
            if name == "rank_jitter":
                g = torch.Generator().manual_seed(seed + 6000 + i)
                count = (3000 - model.cfg.window) // model.cfg.stride + 1
                jitter = torch.randn(1, count, generator=g) * c["stability"]["rank_jitter_uV"]
                after = model(x, return_evidence=True, rank_delta=jitter)
            else:
                after = model(transformed(x, name, c["stability"], seed + 6000 + i), return_evidence=True)
            old, new = selected_set(before), selected_set(after)
            # Sample coverage IoU also permits sub-stride shifts; set identity and
            # spatial alignment are reported separately, never conflated.
            offset = c["stability"]["shift_samples"] if name == "shift" else 0
            covered_old = {j for a, b in old for j in range(a, b)}
            covered_new = {j - offset for a, b in new for j in range(a, b)}
            results[name].append({"set_distance": jaccard_distance(old, new),
                                  "aligned_sample_distance": jaccard_distance(covered_old, covered_new),
                                  "logit_max_abs_change": float((before.logits - after.logits).abs().max()),
                                  "event_total_MAE": float((before.contributions.sum(1) - after.contributions.sum(1)).abs().mean())})
    return {name: {key + "_mean": mean([row[key] for row in rows]) for key in rows[0]} for name, rows in results.items()}


@torch.no_grad()
def budget_probe(model, fixtures, c):
    results = {}
    for budget in c["budgets"]:
        cfg = replace(model.cfg, event_budget=budget, event_scale=c["budget_probe_per_event_scale"] * budget)
        probe = make_model("q1", cfg)
        probe.load_state_dict(model.state_dict()); probe.eval()
        counts, effects = [], []
        for row in fixtures[:5]:
            e = probe(row.x, return_evidence=True)
            counts.append(int(e.valid.sum()))
            effects.append(float(probe(row.x)[0, row.label]))
        results[str(budget)] = {"mean_selected_count": mean(counts), "mean_true_stage_logit": mean(effects),
                                "per_event_scale": c["budget_probe_per_event_scale"], "parameters": 75}
    return results


def run(c, progress=print):
    validate_config(c)
    torch.set_num_threads(c["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    runs = []
    for task_index, task in enumerate(c["tasks"]):
        for seed in c["seeds"]:
            # Task-specific seeds remain identical across all comparison methods.
            eval_seed = seed + 2000 + task_index * 10000
            fixtures = {name: make_fixtures(c["evaluation_size"], eval_seed + j * 100, task, name, c["generator"])
                        for j, name in enumerate(c["cases"])}
            tied = make_fixtures(c["evaluation_size"], eval_seed + 800, task, "ties", c["generator"])
            for method in c["methods"]:
                tick = time.perf_counter()
                model, training = fit(c, seed, task, method)
                scores, audits = {}, {}
                for name, rows in fixtures.items():
                    with torch.no_grad():
                        logits = model(torch.cat([r.x for r in rows]))
                        scores[name] = metrics(logits, torch.tensor([r.label for r in rows]))
                    audits[name] = audit_case(model, rows, c, seed)
                row = {"seed": seed, "task": task, "method": method, "training": training,
                       "metrics": scores, "audits": audits, "tie_stability": stability(model, tied, c, seed),
                       "budget_probe": budget_probe(model, fixtures["competition"], c),
                       "wall_seconds": time.perf_counter() - tick}
                threshold = c["criteria"]
                row["checks"] = {
                    "reconciliation": max(a["max_reconciliation_error"] for a in audits.values()) <= threshold["reconciliation_max_abs"],
                    "W1_reselection_material": mean([audits[name]["mean_absolute_reselection_true_stage"] for name in ("correlated", "competition")]) >= threshold["W1_reselection_effect_min"],
                    "W2_joint_dependence": mean([audits[name]["group_effect_abs_mean"] - audits[name]["individual_effect_abs_mean"] for name in ("redundant", "correlated")]) > threshold["W2_joint_minus_individual_min"],
                    "W4_eraser_agreement": mean([a["eraser_donor_spread_mean"] for a in audits.values()]) <= threshold["W4_eraser_donor_spread_max"],
                    "calibrated_gain": row["tie_stability"]["calibrated_gain"]["logit_max_abs_change_mean"] <= threshold["known_calibration_logit_difference_max"]}
                runs.append(row)
                progress(f"task={task} seed={seed} method={method} dual_accuracy={scores['dual']['accuracy']:.3f} checks={row['checks']} seconds={row['wall_seconds']:.2f}")
    comparisons = []
    for task in c["tasks"]:
        for seed in c["seeds"]:
            reference = next(r for r in runs if r["task"] == task and r["seed"] == seed and r["method"] == "q1")
            rounded = next(r for r in runs if r["task"] == task and r["seed"] == seed and r["method"] == "rounded_q1")
            instability = lambda r: mean([r["tie_stability"][name]["set_distance_mean"] for name in ("noise", "rank_jitter")])
            reduction = instability(reference) - instability(rounded)
            loss = reference["metrics"]["dual"]["accuracy"] - rounded["metrics"]["dual"]["accuracy"]
            comparisons.append({"task": task, "seed": seed, "hard_instability": instability(reference),
                                "rounded_instability": instability(rounded), "instability_reduction": reduction,
                                "dual_accuracy_loss": loss,
                                "W3_PASS": reduction >= c["criteria"]["W3_jaccard_distance_reduction_min"] and loss <= c["criteria"]["W3_clean_accuracy_loss_max"]})
    return {"version": c["version"], "config": c, "config_sha256_canonical": digest(c),
            "environment": environment("cpu"), "runs": runs, "W3_comparisons": comparisons,
            "wall_seconds": time.perf_counter() - started, "process_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "real_data_accessed": False, "reserved_accessed": False, "external_accessed": False,
            "interpretation": "Artificial codes and oracle-declared groups; neither clinical validity nor transfer is tested. Internal effects are not physiological interventions."}

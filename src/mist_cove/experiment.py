"""Reproducible CPU-only falsification harness. No real-data runner exists."""
import hashlib
import json
import math
import platform
import resource
import time

import torch

from mist_transfer.metrics import confusion_metrics
from .audit import audit_interval
from .model import CoveConfig, CoveModel
from .synthetic import make_batch, perturbations


def validate_config(config: dict) -> None:
    if (config.get("version") != "mist-cove-synthetic-v1" or config.get("scope") != "SYNTHETIC_ONLY"
            or any(config.get(k) is not False for k in ("real_data_access", "reserved_access", "external_access"))):
        raise ValueError("synthetic-only configuration required; no data authorization")
    for key in ("train_size", "evaluation_size", "optimizer_steps", "threads", "audit_examples"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("positive synthetic counts required")
    if any(config[key] < 5 or config[key] % 5 for key in ("train_size", "evaluation_size")):
        raise ValueError("synthetic train/evaluation sets require balanced five-class counts")
    if config["modes"] != ["full", "background_only", "q_one", "q_only", "fixed_event_score", "fixed_patch"]:
        raise ValueError("required controls missing")
    if config["seeds"] != [123, 456, 789] or config["coverage_grid"] != [1., .95, .9, .8]:
        raise ValueError("versioned seeds/coverage changed")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("invalid synthetic learning rate")
    required = {"clean_accuracy_min", "calibrated_gain_max_logit_difference", "erased_confidence_max",
                "witness_vs_control_delta_min", "single_loss_accuracy_min", "q_selective_risk_improvement_min",
                "donor_delta_disagreement_max"}
    if set(config["criteria"]) != required or any(not math.isfinite(v) or v < 0 for v in config["criteria"].values()):
        raise ValueError("missing/invalid predeclared falsifiers")
    CoveConfig(**config["model"])


def risk_coverage(probabilities, labels, score, grid):
    order = torch.argsort(score, descending=True, stable=True)
    correct = probabilities.argmax(-1) == labels
    result = []
    for coverage in grid:
        idx = order[:math.ceil(coverage * len(labels))]
        result.append({"requested_coverage": coverage, "coverage": len(idx) / len(labels),
                       "risk": float((~correct[idx]).float().mean()),
                       "class_coverage": [float((labels[idx] == s).sum() / (labels == s).sum())
                                          if (labels == s).any() else None for s in range(5)]})
    return result


def metrics(logits, labels):
    p = logits.softmax(-1)
    pred = p.argmax(-1)
    cm = torch.bincount(labels * 5 + pred, minlength=25).reshape(5, 5)
    result = confusion_metrics(cm.numpy())
    result.update(nll=float(torch.nn.functional.cross_entropy(logits, labels)),
                  brier=float((p - torch.nn.functional.one_hot(labels, 5)).square().sum(-1).mean()),
                  confidence=float(p.max(-1).values.mean()),
                  prediction_counts=torch.bincount(pred, minlength=5).tolist(),
                  collapsed=bool(torch.bincount(pred, minlength=5).max() / len(pred) >= .98))
    confidence = p.max(-1).values
    ece = 0.
    for i in range(15):
        mask = (confidence >= i / 15) & (confidence <= 1 if i == 14 else confidence < (i + 1) / 15)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - (pred[mask] == labels[mask]).float().mean()).abs())
    result["ece15"] = ece
    # Direct per-distinction diagnostic; not called a pairwise availability estimate.
    result["true_vs_other_stage_margins"] = {
        str(s): [None if s == other else float((logits[labels == s, s] - logits[labels == s, other]).mean())
                 for other in range(5)] for s in range(5)}
    return result


def fit(config, seed, mode):
    torch.manual_seed(seed)
    model = CoveModel(CoveConfig(**config["model"]))
    batch = make_batch(config["train_size"], seed + 1000, random_presence=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    losses = []
    for _ in range(config["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(batch.x, mode=mode), batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError("synthetic optimization failed")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite synthetic gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    return model.eval(), {"optimizer_steps": len(losses), "initial_loss": losses[0], "final_loss": losses[-1],
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "parameters_with_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None),
                          "parameters_with_nonzero_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)}


@torch.no_grad()
def evaluate(model, batch, config, mode):
    tensors = perturbations(batch)
    evidence = {name: model(x, return_evidence=True, mode=mode) for name, x in tensors.items()}
    result = {name: metrics(e.logits, batch.y) for name, e in evidence.items()}
    for name, e in evidence.items():
        result[name]["mean_max_availability"] = float(e.availability[..., 0].max(1).values.mean())
        result[name]["mean_absolute_event_contribution"] = float(e.contributions.abs().sum(1).mean())
    mix = ("clean", "both_lost", "matched_corruption")
    logits = torch.cat([evidence[name].logits for name in mix])
    y = batch.y.repeat(len(mix))
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-9).log()).sum(-1)
    q = torch.cat([evidence[name].availability[..., 0].max(1).values for name in mix])
    quality = torch.cat([tensors[name].flatten(1).std(-1) for name in mix])
    curves = {name: risk_coverage(p, y, score, config["coverage_grid"])
              for name, score in (("availability", q), ("negative_entropy", -entropy), ("global_rms", quality))}
    switch = {}
    for name in ("clean", "lose_a", "lose_b", "both_lost"):
        e = evidence[name]
        amounts = []
        for component in (0, 1):
            lo, hi = batch.intervals[:, component].unbind(-1)
            overlap = ((e.event_intervals[..., 0] < hi[:, None]) & (e.event_intervals[..., 1] > lo[:, None]) & e.valid)
            true_contribution = e.contributions.gather(2, batch.y[:, None, None].expand(-1, model.cfg.event_budget, 1))[..., 0]
            amounts.append((true_contribution * overlap).sum(1).clamp_min(0))
        denominator = (amounts[0] + amounts[1]).clamp_min(1e-8)
        switch[name] = {"a_share": float((amounts[0] / denominator).mean()),
                        "b_share": float((amounts[1] / denominator).mean())}
    return result, curves, switch, float((evidence["clean"].logits - evidence["calibrated_gain"].logits).abs().max())


@torch.no_grad()
def audit_experiment(model, batch, examples):
    records = []
    for i in range(min(examples, len(batch.y))):
        x = batch.x[i:i + 1]; stage = int(batch.y[i])
        e = model(x, return_evidence=True)
        k = int(e.contributions[0, :, stage].argmax())
        interval = tuple(e.event_intervals[0, k].tolist())
        # Known generated background is a donor oracle, not a real-data strategy.
        occupied = batch.intervals[i].tolist()
        candidates = [(s, s + 200) for s in range(0, 3000, 200)
                      if all(s + 200 <= a or s >= b for a, b in occupied)
                      and (s + 200 <= interval[0] or s >= interval[1])]
        donor = candidates[0]
        # Known nuisance has same generating amplitude/envelope/length as witnesses.
        # Also choose an energy-nearest window from non-witness physical tiles.
        target_energy = float(x[0, 0, interval[0]:interval[1]].square().mean())
        controls = [tuple(row) for row in batch.intervals[i].tolist()[2:]] + candidates
        energy_control = min(controls, key=lambda row: abs(float(x[0, 0, row[0]:row[1]].square().mean()) - target_energy))
        energy_error = abs(float(x[0, 0, energy_control[0]:energy_control[1]].square().mean()) - target_energy) / max(target_energy, 1e-8)
        for strategy in ("linear_bridge", "same_epoch_background"):
            kwargs = {"donor_interval": donor} if strategy == "same_epoch_background" else {}
            selected = audit_interval(model, x, interval, strategy=strategy, **kwargs)
            random_control = audit_interval(model, x, energy_control, strategy=strategy, **kwargs)
            records.append({"strategy": strategy, "selected_delta": float(selected.logit_delta[stage]),
                            "control_delta": float(random_control.logit_delta[stage]),
                            "claimed": float(selected.claimed_contribution[stage]),
                            "mean_signed_claim_times_delta": float((selected.claimed_contribution * selected.logit_delta).mean()),
                            "background_delta": float(selected.background_delta[stage]),
                            "energy_match_relative_error": energy_error,
                            "diagnostics": selected.diagnostics})
    mean = lambda key: sum(row[key] for row in records) / len(records)
    disagreements = [abs(records[j]["selected_delta"] - records[j + 1]["selected_delta"]) for j in range(0, len(records), 2)]
    claims = torch.tensor([r["claimed"] for r in records]); deltas = torch.tensor([r["selected_delta"] for r in records])
    return {"count": len(records), "selected_delta_mean": mean("selected_delta"),
            "control_delta_mean": mean("control_delta"), "claimed_mean": mean("claimed"),
            "signed_agreement_mean": mean("mean_signed_claim_times_delta"),
            "claim_delta_mae": float((claims - deltas).abs().mean()),
            "claim_delta_correlation": float(torch.corrcoef(torch.stack((claims, deltas)))[0, 1]) if claims.std() > 0 and deltas.std() > 0 else None,
            "donor_delta_disagreement_mean": sum(disagreements) / len(disagreements),
            "energy_match_relative_error_mean": mean("energy_match_relative_error"),
            "donors": {strategy: {"selected_delta_mean": sum(r["selected_delta"] for r in records if r["strategy"] == strategy) / (len(records) / 2),
                                  "control_delta_mean": sum(r["control_delta"] for r in records if r["strategy"] == strategy) / (len(records) / 2)}
                       for strategy in ("linear_bridge", "same_epoch_background")},
            "diagnostic_note": "Known synthetic background donors; energy-nearest controls may remain imperfectly matched."}


@torch.no_grad()
def stress_probes(model, batch, seed):
    """Exploratory controls, reported separately from predeclared checks.

    Repeated diagnostic copies reveal reselection/redundancy; equal-energy
    off-band copies provide several random non-witness control candidates.
    """
    g = torch.Generator().manual_seed(seed + 3000)
    rows = []
    for i in range(min(5, len(batch.y))):
        base = batch.x[i:i + 1].clone(); stage = int(batch.y[i])
        occupied = batch.intervals[i].tolist()
        free = [(s, s + 200) for s in range(0, 3000, 200)
                if all(s + 200 <= a or s >= b for a, b in occupied)]
        donor = free[-1]
        a0, a1 = batch.intervals[i, 0].tolist()
        n0, n1 = batch.intervals[i, 2].tolist()
        controls = [tuple(occupied[2])]
        for lo, hi in free[:3]:
            base[0, 0, lo:hi] += batch.nuisance[i, 0, n0:n1]
            controls.append((lo, hi))
        e = model(base, return_evidence=True)
        witness = tuple(e.event_intervals[0, int(e.contributions[0, :, stage].argmax())].tolist())
        # Randomized among known length/amplitude/envelope matched non-witnesses.
        selected_controls = [controls[j] for j in torch.randperm(len(controls), generator=g)[:3].tolist()]
        for strategy in ("linear_bridge", "same_epoch_background"):
            kwargs = {"donor_interval": donor} if strategy == "same_epoch_background" else {}
            effect = audit_interval(model, base, witness, strategy=strategy, **kwargs)
            deltas = [float(audit_interval(model, base, interval, strategy=strategy, **kwargs).logit_delta[stage])
                      for interval in selected_controls]
            rows.append({"donor": strategy, "selected_delta": float(effect.logit_delta[stage]),
                         "random_energy_length_matched_delta": sum(deltas) / len(deltas)})
    # Similar-looking, genuinely nonselected intervals in a redundant signal.
    x = batch.x[:1].clone(); stage = int(batch.y[0])
    occupied = batch.intervals[0].tolist()
    free = [(s, s + 200) for s in range(0, 3000, 200)
            if all(s + 200 <= a or s >= b for a, b in occupied)]
    a0, a1 = occupied[0]
    copies = free[:6]
    for lo, hi in copies:
        x[0, 0, lo:hi] = batch.x[0, 0, a0:a1]
    e = model(x, return_evidence=True)
    selected_intervals = [tuple(row) for row in e.event_intervals[0, e.valid[0]].tolist()]
    selected = tuple(e.event_intervals[0, int(e.contributions[0, :, stage].argmax())].tolist())
    nonselected = next(interval for interval in copies if interval not in selected_intervals)
    chosen = audit_interval(model, x, selected)
    similar = audit_interval(model, x, nonselected)
    return {"status": "EXPLORATORY_NOT_NEW_ACCEPTANCE_THRESHOLDS",
            "random_matched_control_comparisons": rows,
            "redundant_similar_nonselected": {
                "claimed_selected_contribution": float(chosen.claimed_contribution[stage]),
                "selected_replacement_delta": float(chosen.logit_delta[stage]),
                "similar_nonselected_replacement_delta": float(similar.logit_delta[stage]),
                "selection_changed": chosen.diagnostics["selection_changed"],
                "interpretation": "Reselection can replace a redundant witness; single-window necessity is not guaranteed."}}


def run(config, progress=print):
    validate_config(config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    runs = []
    for seed in config["seeds"]:
        heldout = make_batch(config["evaluation_size"], seed + 2000)
        for mode in config["modes"]:
            before = time.perf_counter()
            model, training = fit(config, seed, mode)
            scores, curves, switch, gain_error = evaluate(model, heldout, config, mode)
            row = {"seed": seed, "mode": mode, "training": training, "metrics": scores,
                   "risk_coverage_mixed_clean_erased_matched": curves, "witness_shares": switch,
                   "calibrated_gain_max_logit_difference": gain_error}
            if mode == "full":
                row["audit"] = audit_experiment(model, heldout, config["audit_examples"])
                row["stress_probes"] = stress_probes(model, heldout, seed)
                c = config["criteria"]
                row["predeclared_checks"] = {
                    "clean_discrimination": scores["clean"]["accuracy"] >= c["clean_accuracy_min"],
                    "benign_reversibility": gain_error <= c["calibrated_gain_max_logit_difference"],
                    "erased_confidence": scores["both_lost"]["confidence"] <= c["erased_confidence_max"],
                    "H2_selected_exceeds_control": row["audit"]["selected_delta_mean"] - row["audit"]["control_delta_mean"] >= c["witness_vs_control_delta_min"],
                    "H2_donor_stability": row["audit"]["donor_delta_disagreement_mean"] <= c["donor_delta_disagreement_max"],
                    "H3_single_loss": min(scores[name]["accuracy"] for name in ("lose_a", "lose_b")) >= c["single_loss_accuracy_min"],
                    "H3_switch": switch["lose_a"]["b_share"] > switch["clean"]["b_share"] and switch["lose_b"]["a_share"] > switch["clean"]["a_share"],
                    "H1_beats_simple_selectors": all(curves["availability"][j]["risk"] <= min(curves["negative_entropy"][j]["risk"], curves["global_rms"][j]["risk"]) - c["q_selective_risk_improvement_min"] for j in range(1, 4))}
            row["wall_seconds"] = time.perf_counter() - before
            runs.append(row)
            progress(f"seed={seed} mode={mode} clean={scores['clean']['accuracy']:.3f} erased_confidence={scores['both_lost']['confidence']:.3f} seconds={row['wall_seconds']:.2f}")
    return {"version": config["version"], "scope": "SYNTHETIC_ONLY", "config": config,
            "config_sha256_canonical": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "device": "cpu", "threads": torch.get_num_threads(), "deterministic": True},
            "runs": runs, "wall_seconds": time.perf_counter() - started,
            "process_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "real_data_accessed": False, "reserved_accessed": False, "external_accessed": False,
            "interpretation": "Mechanism tests on constructed frequency codes; no clinical or transfer evidence. Failed scientific checks are retained, not software exceptions."}

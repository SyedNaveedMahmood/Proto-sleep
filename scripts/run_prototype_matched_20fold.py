#!/usr/bin/env python3
"""Final validation-only prototype screen for the BHI one-page abstract.

A1 = plain AttnSleep.
A3_current_matched = current spherical ProtoAttnSleep with the same initialized MRCNN,
TCE and classifier as A1. The only additional initialized state is the prototype pathway.

This runner rotates all requested Sleep-EDF subjects through the existing AttnSleep fold rule,
uses three supervised seeds by default, opens only train + validation NPZ files for each fold,
and never evaluates that fold's designated test subject.

The primary endpoint is the seed-averaged, fold-level validation Macro-F1 difference
A3_current_matched - A1. Per-stage F1 and simple prototype diagnostics are saved as supporting
analysis from the same trained models; they are not a second tuning experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List


def _csv_ints(value: str) -> List[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/prototype_20fold_matched")
    p.add_argument("--folds", type=_csv_ints, default=list(range(20)))
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse any completed checkpoints and train only missing members. Safe for overnight resume.",
    )
    return p.parse_args()


def _exact_sign_p(pos: int, neg: int) -> float:
    n = int(pos + neg)
    if n == 0:
        return float("nan")
    k = min(int(pos), int(neg))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_mean_ci(x, n_boot: int = 20000, seed: int = 20260823):
    import numpy as np

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _prototype_diagnostics(model, loader, device):
    import numpy as np
    import torch
    import torch.nn.functional as F

    model = model.to(device).eval()
    usage_sum = None
    n = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            out = model(x)
            u = out["usage"].float()  # [B,K], already averaged over local positions
            usage_sum = u.sum(dim=0) if usage_sum is None else usage_sum + u.sum(dim=0)
            n += int(u.shape[0])
    if usage_sum is None or n == 0:
        raise RuntimeError("prototype diagnostic loader was empty")
    q = (usage_sum / usage_sum.sum()).clamp_min(1e-12)
    effective_k = float(torch.exp(-(q * torch.log(q)).sum()).cpu())

    p = F.normalize(model.prototype_bank.prototypes.detach().float(), dim=-1)
    gram = p @ p.t()
    eye = torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    max_offdiag = float(gram.masked_select(~eye).max().cpu())
    beta = float(torch.sigmoid(model.beta_logit.detach().float()).cpu())
    return {
        "prototype_effective_k": effective_k,
        "prototype_max_offdiag_cosine": max_offdiag,
        "prototype_beta": beta,
        "prototype_usage": np.asarray(q.cpu(), dtype=float).tolist(),
    }


def main():
    args = parse_args()

    os.environ["SLEEP_EDF_NPZ_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    os.environ["PROTOMAE_OUT"] = str(Path(args.output_dir).expanduser().resolve())
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.attnsleep import AttnSleepBaseline
    from protosleep.cache import load_model_state
    from protosleep.config import MICRO_BATCH_SIZE, PROJECT_VERSION, PROTO_TRIALS, STAGE_NAMES, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.micro import evaluate_micro_loader, train_micro_model
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.prototype_baseline import build_matched_a1_a3
    from protosleep.prototypes import ProtoAttnSleep
    from protosleep.utils import DEVICE

    if len(set(args.folds)) != len(args.folds):
        raise RuntimeError("duplicate fold IDs are not allowed")
    if any(f < 0 or f > 19 for f in args.folds):
        raise RuntimeError("Sleep-EDF-20 fold IDs must be in 0..19")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("FINAL MATCHED PROTOTYPE SCREEN: ATTNSLEEP A1 VS A3_CURRENT_MATCHED")
    print("=" * 108)
    print(f"device={DEVICE} | AMP={USE_AMP} | batch={MICRO_BATCH_SIZE}")
    print(f"folds={args.folds}")
    print(f"supervised seeds={args.seeds}")
    print("A1/A3 shared MRCNN + TCE + classifier initialization: EXACT MATCH BY CONSTRUCTION")
    print("primary endpoint: seed-averaged fold-level validation Macro-F1 A3-A1")
    print("frozen screening criterion: mean A3-A1 > 0 AND positive folds >= 12/20 when all 20 folds are run")
    print("DESIGNATED TEST SUBJECT FILES ARE NOT OPENED FOR THEIR FOLD. TEST METRICS ARE NOT COMPUTED.")
    print("=" * 108)

    rows: List[Dict[str, Any]] = []
    validation_subjects_seen: List[int] = []

    for fold in args.folds:
        split = fold_subjects_from_npz(args.data_dir, fold)
        train_subjects = split["train_subjects"]
        val_subjects = split["val_subjects"]
        test_subjects = split["test_subjects"]
        validation_subjects_seen.extend(val_subjects)

        train_recordings, train_io = load_recordings_for_subjects(args.data_dir, train_subjects)
        val_recordings, val_io = load_recordings_for_subjects(args.data_dir, val_subjects)
        opened_subjects = sorted(set(train_io["subjects"]) | set(val_io["subjects"]))
        if set(opened_subjects) & set(test_subjects):
            raise RuntimeError(
                f"Leakage barrier failure: fold {fold} test subjects {test_subjects} appear in opened subjects {opened_subjects}"
            )
        class_weights = balanced_class_weights_from_train(train_recordings)

        print("\n" + "=" * 108)
        print(f"FOLD {fold:02d} | train={train_subjects} | val={val_subjects} | test={test_subjects} (FILES NOT OPENED)")
        print(
            f"opened train recordings={train_io['n_recordings']} epochs={train_io['n_epochs']:,} | "
            f"val recordings={val_io['n_recordings']} epochs={val_io['n_epochs']:,}"
        )
        print("=" * 108)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            a1, a3, init_meta = build_matched_a1_a3(actual_seed)

            def loaders():
                return (
                    make_epoch_loader(train_recordings, MICRO_BATCH_SIZE, shuffle=True),
                    make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False),
                )

            def run_train(name, model, path, proto_cfg=None):
                reused = False
                history = None
                if args.reuse and path.exists():
                    model = load_model_state(model, path, DEVICE)
                    reused = True
                else:
                    tr, va = loaders()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with (log_dir / f"{name}.log").open("w", buffering=1) as f, contextlib.redirect_stdout(f):
                        result = train_micro_model(
                            model,
                            tr,
                            va,
                            class_weights,
                            path,
                            proto_cfg=proto_cfg,
                            seed=actual_seed,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - t0
                    model = result["model"].to(DEVICE).eval()
                    history = result["history"]
                    history.to_csv(log_dir / f"{name}_history.csv", index=False)
                    return model, evaluate_micro_loader(model, va, DEVICE), seconds, reused

                va = loaders()[1]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                metrics = evaluate_micro_loader(model, va, DEVICE)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return model, metrics, time.perf_counter() - t0, reused

            a1, m1, t1, r1 = run_train("A1", a1, ckpt_dir / "A1_attnsleep.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            a3, m3, t3, r3 = run_train(
                "A3_current_matched",
                a3,
                ckpt_dir / "A3_current_matched.pt",
                proto_cfg=PROTO_TRIALS[0],
            )
            proto_diag = _prototype_diagnostics(a3, loaders()[1], DEVICE)

            row: Dict[str, Any] = {
                "fold": int(fold),
                "seed": int(base_seed),
                "actual_seed": int(actual_seed),
                "train_subjects": ",".join(map(str, train_subjects)),
                "val_subjects": ",".join(map(str, val_subjects)),
                "test_subjects_locked": ",".join(map(str, test_subjects)),
                "test_subject_files_opened": False,
                "shared_initialization_match": True,
                "A1_f1": float(m1["macro_f1"]),
                "A3_f1": float(m3["macro_f1"]),
                "A3_minus_A1": float(m3["macro_f1"] - m1["macro_f1"]),
                "A1_accuracy": float(m1["accuracy"]),
                "A3_accuracy": float(m3["accuracy"]),
                "A1_kappa": float(m1["kappa"]),
                "A3_kappa": float(m3["kappa"]),
                "A1_seconds": float(t1),
                "A3_seconds": float(t3),
                "A1_reused": bool(r1),
                "A3_reused": bool(r3),
                "project_version": PROJECT_VERSION,
                "amp": bool(USE_AMP),
                **{k: v for k, v in init_meta.items() if k != "seed"},
                **{k: v for k, v in proto_diag.items() if k != "prototype_usage"},
            }
            for i, stage in enumerate(STAGE_NAMES):
                row[f"A1_{stage}_f1"] = float(m1["per_class_f1"][i])
                row[f"A3_{stage}_f1"] = float(m3["per_class_f1"][i])
                row[f"delta_{stage}_f1"] = float(m3["per_class_f1"][i] - m1["per_class_f1"][i])

            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "prototype_20fold_partial.csv", index=False)

            with (run_dir / "provenance.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **row,
                        "matched_initialization": init_meta,
                        "prototype_usage": proto_diag["prototype_usage"],
                        "train_io": train_io,
                        "val_io": val_io,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            print(
                f"fold={fold:02d} seed={base_seed} | A1={row['A1_f1']:.4f} | "
                f"A3={row['A3_f1']:.4f} | A3-A1={row['A3_minus_A1']:+.4f} | "
                f"beta={row['prototype_beta']:.3f} effK={row['prototype_effective_k']:.2f}"
            )

            del a1, a3
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "prototype_20fold_raw.csv", index=False)

    agg: Dict[str, Any] = {
        "A1_f1": ("A1_f1", "mean"),
        "A3_f1": ("A3_f1", "mean"),
        "A1_seed_sd": ("A1_f1", "std"),
        "A3_seed_sd": ("A3_f1", "std"),
        "prototype_beta": ("prototype_beta", "mean"),
        "prototype_effective_k": ("prototype_effective_k", "mean"),
        "prototype_max_offdiag_cosine": ("prototype_max_offdiag_cosine", "mean"),
    }
    for stage in STAGE_NAMES:
        agg[f"A1_{stage}_f1"] = (f"A1_{stage}_f1", "mean")
        agg[f"A3_{stage}_f1"] = (f"A3_{stage}_f1", "mean")

    fold_means = raw.groupby("fold", as_index=False).agg(**agg)
    fold_means["A3_minus_A1"] = fold_means["A3_f1"] - fold_means["A1_f1"]
    for stage in STAGE_NAMES:
        fold_means[f"delta_{stage}_f1"] = fold_means[f"A3_{stage}_f1"] - fold_means[f"A1_{stage}_f1"]
    fold_means.to_csv(output_root / "prototype_20fold_fold_means.csv", index=False)

    x = fold_means["A3_minus_A1"].to_numpy(float)
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    lo, hi = _bootstrap_mean_ci(x)
    effects = pd.DataFrame(
        [{
            "effect": "A3_minus_A1",
            "mean": float(np.mean(x)),
            "std_across_folds": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "median": float(np.median(x)),
            "positive_folds": pos,
            "negative_folds": neg,
            "n_folds": int(len(x)),
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "exact_sign_p_two_sided": _exact_sign_p(pos, neg),
        }]
    )
    effects.to_csv(output_root / "prototype_20fold_effects.csv", index=False)

    stage_rows = []
    for stage in STAGE_NAMES:
        sx = fold_means[f"delta_{stage}_f1"].to_numpy(float)
        slo, shi = _bootstrap_mean_ci(sx, seed=20260823 + STAGE_NAMES.index(stage) + 1)
        stage_rows.append({
            "stage": stage,
            "A1_mean_f1": float(fold_means[f"A1_{stage}_f1"].mean()),
            "A3_mean_f1": float(fold_means[f"A3_{stage}_f1"].mean()),
            "mean_delta": float(np.mean(sx)),
            "median_delta": float(np.median(sx)),
            "positive_folds": int(np.sum(sx > 0)),
            "n_folds": int(len(sx)),
            "bootstrap95_low": slo,
            "bootstrap95_high": shi,
        })
    stage_effects = pd.DataFrame(stage_rows)
    stage_effects.to_csv(output_root / "prototype_20fold_stage_effects.csv", index=False)

    all_20 = len(args.folds) == 20 and sorted(args.folds) == list(range(20))
    unique_val = len(set(validation_subjects_seen)) == len(validation_subjects_seen)
    criterion_pass = bool(all_20 and float(np.mean(x)) > 0.0 and pos >= 12)

    summary = {
        "folds": args.folds,
        "seeds": args.seeds,
        "all_20_folds": all_20,
        "validation_subjects_unique": unique_val,
        "mean_A3_minus_A1": float(np.mean(x)),
        "positive_folds": pos,
        "n_folds": int(len(x)),
        "bootstrap95": [lo, hi],
        "exact_sign_p_two_sided": float(effects.iloc[0]["exact_sign_p_two_sided"]),
        "frozen_screening_criterion": "mean A3-A1 > 0 and >=12/20 positive folds",
        "criterion_pass": criterion_pass,
        "test_metrics_computed": False,
        "test_subject_files_opened_for_own_fold": False,
    }
    (output_root / "prototype_20fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 108)
    print("SEED-AVERAGED FOLD RESULTS")
    print("=" * 108)
    print(fold_means[["fold", "A1_f1", "A3_f1", "A1_seed_sd", "A3_seed_sd", "A3_minus_A1"]].to_string(index=False))
    print("\n" + "=" * 108)
    print("PRIMARY PAIRED EFFECT")
    print("=" * 108)
    print(effects.to_string(index=False))
    print("\n" + "=" * 108)
    print("PER-STAGE SUPPORTING EFFECTS")
    print("=" * 108)
    print(stage_effects.to_string(index=False))
    print("\nFROZEN SCREENING CRITERION PASS:", "YES" if criterion_pass else "NO")
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED FOR THEIR OWN FOLD: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

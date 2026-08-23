#!/usr/bin/env python3
"""Final downstream confirmation of MorphSpec-R1 on new development folds.

The MorphSpec-R1 representation objective was designed on folds 0-4 and its frozen-probe
benefit was confirmed on folds 5-9. This script uses a *new* set of folds (default 10-14)
for the downstream-transfer question so the transfer result is not selected on the same
folds used for representation confirmation.

Within each fold/seed we build three AttnSleep models with byte-identical initialized
TCE/classifier state:

  A1_standard       random MRCNN, standard supervised AttnSleep training
  original_stage    original leakage-safe MorphMAE-v2 MRCNN, frozen 5 epochs then 1e-4 LR
  morphspec_stage   frozen MorphSpec-R1 MRCNN init, same staged transfer recipe as original

The staged recipe (MAE_STAGE_1E4) was selected before MorphSpec downstream results were
observed. No transfer hyperparameter search occurs here. Only train + validation NPZ files
are opened. The designated test subject is never evaluated.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--folds", type=_csv_ints, default=[10, 11, 12, 13, 14])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--original-checkpoint-pattern", required=True)
    p.add_argument("--morphspec-checkpoint-pattern", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/morphspec_r1_downstream_folds10_14")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--reuse", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["SLEEP_EDF_NPZ_DIR"] = str(data_dir)
    os.environ["PROTOMAE_OUT"] = str(output_root)
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.config import MICRO_BATCH_SIZE, STAGE_NAMES, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.mae_baseline import verify_mae_checkpoint_for_fold
    from protosleep.micro import evaluate_micro_loader, train_micro_model
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.morphmae_transfer import MAE_STAGE_1E4, train_attnsleep_transfer
    from protosleep.morphspec_confirm import build_matched_probe_triplet
    from protosleep.utils import DEVICE

    print("=" * 100)
    print("MORPHSPEC-R1 FINAL DOWNSTREAM TRANSFER CONFIRMATION")
    print("=" * 100)
    print(f"device={DEVICE} AMP={USE_AMP}")
    print(f"new downstream folds={args.folds} supervised seeds={args.seeds}")
    print("A1_standard: random MRCNN + standard supervised AttnSleep protocol")
    print(
        "original_stage/morphspec_stage: frozen 5 epochs, then encoder LR=1e-4 and head LR=1e-3"
    )
    print("Transfer recipe is frozen; no hyperparameter selection occurs in this run.")
    print("Primary endpoint: fold-level MorphSpec-stage minus A1-standard validation Macro-F1.")
    print("Secondary causal endpoint: MorphSpec-stage minus original-MorphMAE-stage.")
    print("DESIGNATED TEST SUBJECT FILES WILL NOT BE OPENED.")
    print("TEST METRICS WILL NOT BE COMPUTED.")
    print("=" * 100)

    rows: List[Dict[str, Any]] = []

    for fold in args.folds:
        split = fold_subjects_from_npz(data_dir, fold)
        original_ckpt = Path(args.original_checkpoint_pattern.format(fold=fold)).expanduser().resolve()
        morphspec_ckpt = Path(args.morphspec_checkpoint_pattern.format(fold=fold)).expanduser().resolve()
        original_meta = verify_mae_checkpoint_for_fold(original_ckpt, fold, split["train_subjects"])
        morphspec_meta = verify_mae_checkpoint_for_fold(morphspec_ckpt, fold, split["train_subjects"])

        train_recordings, train_io = load_recordings_for_subjects(data_dir, split["train_subjects"])
        val_recordings, val_io = load_recordings_for_subjects(data_dir, split["val_subjects"])
        opened = set(train_io["subjects"]) | set(val_io["subjects"])
        if opened & set(split["test_subjects"]):
            raise RuntimeError("Leakage barrier failure: designated test subject was opened")
        class_weights = balanced_class_weights_from_train(train_recordings)

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("train subjects:", split["train_subjects"])
        print("validation subjects:", split["val_subjects"])
        print("designated test subjects:", split["test_subjects"], "(FILES NOT OPENED)")
        print("original MorphMAE-v2:", original_ckpt)
        print("MorphSpec-R1:", morphspec_ckpt)
        print("=" * 100)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            a1_model, original_model, morphspec_model, init_meta = build_matched_probe_triplet(
                actual_seed, original_ckpt, morphspec_ckpt
            )

            def fresh_loaders():
                return (
                    make_epoch_loader(train_recordings, MICRO_BATCH_SIZE, shuffle=True),
                    make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False),
                )

            def eval_loaded(model, checkpoint: Path):
                payload = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(payload["state_dict"])
                va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
                metrics = evaluate_micro_loader(model.to(DEVICE), va, DEVICE)
                return model.cpu(), metrics, int(payload.get("best_epoch", -1))

            def run_a1(model):
                checkpoint = ckpt_dir / "A1_standard.pt"
                if args.reuse and checkpoint.is_file():
                    loaded, metrics, best_epoch = eval_loaded(model, checkpoint)
                    return loaded, metrics, best_epoch, 0.0, True
                tr, va = fresh_loaders()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with (log_dir / "A1_standard.log").open("w", buffering=1) as f, contextlib.redirect_stdout(f):
                    result = train_micro_model(
                        model,
                        tr,
                        va,
                        class_weights,
                        checkpoint,
                        proto_cfg=None,
                        seed=actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trained = result["model"].to(DEVICE).eval()
                metrics = evaluate_micro_loader(trained, va, DEVICE)
                return (
                    trained.cpu(),
                    metrics,
                    int(result["best_epoch"]),
                    float(time.perf_counter() - t0),
                    False,
                )

            def run_stage(name: str, model):
                checkpoint = ckpt_dir / f"{name}.pt"
                if args.reuse and checkpoint.is_file():
                    loaded, metrics, best_epoch = eval_loaded(model, checkpoint)
                    payload = torch.load(checkpoint, map_location="cpu")
                    return (
                        loaded,
                        metrics,
                        best_epoch,
                        float(payload.get("mrcnn_relative_drift", float("nan"))),
                        0.0,
                        True,
                    )
                tr, va = fresh_loaders()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with (log_dir / f"{name}.log").open("w", buffering=1) as f, contextlib.redirect_stdout(f):
                    result = train_attnsleep_transfer(
                        model,
                        tr,
                        va,
                        class_weights,
                        checkpoint,
                        MAE_STAGE_1E4,
                        actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["history"].to_csv(log_dir / f"{name}.csv", index=False)
                trained = result["model"].to(DEVICE).eval()
                metrics = evaluate_micro_loader(trained, va, DEVICE)
                return (
                    trained.cpu(),
                    metrics,
                    int(result["best_epoch"]),
                    float(result["mrcnn_relative_drift"]),
                    float(time.perf_counter() - t0),
                    False,
                )

            print(f"--- fold={fold} seed={base_seed} | A1_standard ---")
            a1_model, m1, e1, t1, r1 = run_a1(a1_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"--- fold={fold} seed={base_seed} | original_stage ---")
            original_model, mo, eo, drift_o, to, ro = run_stage("original_stage", original_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"--- fold={fold} seed={base_seed} | morphspec_stage ---")
            morphspec_model, mm, em, drift_m, tm, rm = run_stage("morphspec_stage", morphspec_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            row: Dict[str, Any] = {
                "fold": int(fold),
                "seed": int(base_seed),
                "actual_seed": int(actual_seed),
                "train_subjects": ",".join(map(str, split["train_subjects"])),
                "val_subjects": ",".join(map(str, split["val_subjects"])),
                "test_subjects_locked": ",".join(map(str, split["test_subjects"])),
                "test_subject_files_opened": False,
                "original_checkpoint": str(original_ckpt),
                "original_checkpoint_sha256": original_meta["sha256"],
                "morphspec_checkpoint": str(morphspec_ckpt),
                "morphspec_checkpoint_sha256": morphspec_meta["sha256"],
                "A1_standard_f1": float(m1["macro_f1"]),
                "original_stage_f1": float(mo["macro_f1"]),
                "morphspec_stage_f1": float(mm["macro_f1"]),
                "original_minus_A1": float(mo["macro_f1"] - m1["macro_f1"]),
                "morphspec_minus_A1": float(mm["macro_f1"] - m1["macro_f1"]),
                "morphspec_minus_original": float(mm["macro_f1"] - mo["macro_f1"]),
                "A1_best_epoch": int(e1),
                "original_best_epoch": int(eo),
                "morphspec_best_epoch": int(em),
                "original_mrcnn_drift": float(drift_o),
                "morphspec_mrcnn_drift": float(drift_m),
                "A1_seconds": float(t1),
                "original_seconds": float(to),
                "morphspec_seconds": float(tm),
                "A1_reused": bool(r1),
                "original_reused": bool(ro),
                "morphspec_reused": bool(rm),
                "non_mrcnn_initialization_match": True,
                "amp": bool(USE_AMP),
            }
            for idx, stage in enumerate(STAGE_NAMES):
                row[f"A1_{stage}_f1"] = float(m1["per_class_f1"][idx])
                row[f"original_{stage}_f1"] = float(mo["per_class_f1"][idx])
                row[f"morphspec_{stage}_f1"] = float(mm["per_class_f1"][idx])

            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "downstream_partial.csv", index=False)
            (run_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        **row,
                        "transfer_recipe": MAE_STAGE_1E4.__dict__,
                        "initialization": init_meta,
                        "original_metadata": original_meta,
                        "morphspec_metadata": morphspec_meta,
                        "train_io": train_io,
                        "val_io": val_io,
                    },
                    indent=2,
                    default=str,
                )
            )

            print(
                f"fold={fold} seed={base_seed} | A1={row['A1_standard_f1']:.4f} "
                f"original={row['original_stage_f1']:.4f} morphspec={row['morphspec_stage_f1']:.4f} | "
                f"MorphSpec-A1={row['morphspec_minus_A1']:+.4f} "
                f"MorphSpec-original={row['morphspec_minus_original']:+.4f}"
            )

            del a1_model, original_model, morphspec_model

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "downstream_raw.csv", index=False)

    metric_cols = [
        "A1_standard_f1",
        "original_stage_f1",
        "morphspec_stage_f1",
        "original_minus_A1",
        "morphspec_minus_A1",
        "morphspec_minus_original",
        "original_mrcnn_drift",
        "morphspec_mrcnn_drift",
    ]
    for stage in STAGE_NAMES:
        metric_cols.extend([f"A1_{stage}_f1", f"original_{stage}_f1", f"morphspec_{stage}_f1"])
    fold_means = raw.groupby("fold", as_index=False)[metric_cols].mean()
    fold_means.to_csv(output_root / "downstream_fold_means.csv", index=False)

    effect_rows = []
    for col in ("original_minus_A1", "morphspec_minus_A1", "morphspec_minus_original"):
        x = fold_means[col].to_numpy(float)
        pos = int(np.sum(x > 0))
        neg = int(np.sum(x < 0))
        lo, hi = _bootstrap_mean_ci(x)
        effect_rows.append(
            {
                "effect": col,
                "mean": float(np.mean(x)),
                "std_across_folds": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)),
                "positive_folds": pos,
                "negative_folds": neg,
                "n_folds": int(len(x)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "exact_sign_p_two_sided": _exact_sign_p(pos, neg),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output_root / "downstream_effects.csv", index=False)

    stage_rows = []
    for stage in STAGE_NAMES:
        for effect_name, left, right in (
            ("morphspec_minus_A1", f"morphspec_{stage}_f1", f"A1_{stage}_f1"),
            ("morphspec_minus_original", f"morphspec_{stage}_f1", f"original_{stage}_f1"),
        ):
            x = (fold_means[left] - fold_means[right]).to_numpy(float)
            stage_rows.append(
                {
                    "stage": stage,
                    "effect": effect_name,
                    "mean_delta_f1": float(np.mean(x)),
                    "median_delta_f1": float(np.median(x)),
                    "positive_folds": int(np.sum(x > 0)),
                    "n_folds": int(len(x)),
                }
            )
    stage_effects = pd.DataFrame(stage_rows)
    stage_effects.to_csv(output_root / "downstream_stage_effects.csv", index=False)

    primary = effects[effects["effect"] == "morphspec_minus_A1"].iloc[0]
    secondary = effects[effects["effect"] == "morphspec_minus_original"].iloc[0]
    required_positive = max(1, math.ceil(0.60 * len(fold_means)))
    primary_pass = bool(float(primary["mean"]) > 0 and int(primary["positive_folds"]) >= required_positive)
    secondary_pass = bool(
        float(secondary["mean"]) > 0 and int(secondary["positive_folds"]) >= required_positive
    )
    decision = {
        "new_downstream_folds": True,
        "representation_design_folds": [0, 1, 2, 3, 4],
        "representation_confirmation_folds": [5, 6, 7, 8, 9],
        "downstream_evaluation_folds": [int(v) for v in args.folds],
        "transfer_recipe": MAE_STAGE_1E4.__dict__,
        "primary_endpoint": "morphspec_stage_minus_A1_standard_validation_macro_f1",
        "secondary_endpoint": "morphspec_stage_minus_original_stage_validation_macro_f1",
        "required_positive_folds": int(required_positive),
        "mean_morphspec_minus_A1": float(primary["mean"]),
        "positive_folds_vs_A1": int(primary["positive_folds"]),
        "passes_primary_downstream_gate": primary_pass,
        "mean_morphspec_minus_original": float(secondary["mean"]),
        "positive_folds_vs_original": int(secondary["positive_folds"]),
        "passes_objective_downstream_gate": secondary_pass,
        "passes_full_downstream_gate": bool(primary_pass and secondary_pass),
        "gate": "mean delta > 0 and positive in at least 60% of new downstream folds",
        "test_metrics_computed": False,
        "designated_test_subject_files_opened": False,
        "external_transfer_claim_allowed": False,
    }
    (output_root / "downstream_summary.json").write_text(json.dumps(decision, indent=2))

    print("\n" + "=" * 100)
    print("SEED-AVERAGED DOWNSTREAM FOLD RESULTS")
    print("=" * 100)
    show_cols = [
        "fold",
        "A1_standard_f1",
        "original_stage_f1",
        "morphspec_stage_f1",
        "original_minus_A1",
        "morphspec_minus_A1",
        "morphspec_minus_original",
    ]
    print(fold_means[show_cols].to_string(index=False))
    print("\n" + "=" * 100)
    print("DOWNSTREAM EFFECT SUMMARY")
    print("=" * 100)
    print(effects.to_string(index=False))
    print("\n" + "=" * 100)
    print("PER-STAGE SUPPORTING EFFECTS")
    print("=" * 100)
    print(stage_effects.to_string(index=False))
    print("\n" + "=" * 100)
    print("FROZEN DOWNSTREAM DECISION")
    print("=" * 100)
    print(json.dumps(decision, indent=2))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("External-transfer conclusion allowed: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

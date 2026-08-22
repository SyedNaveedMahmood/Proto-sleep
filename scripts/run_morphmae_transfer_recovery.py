#!/usr/bin/env python3
"""Theory-driven MorphMAE transfer recovery on development folds.

This is deliberately diagnostic rather than a result-rescue grid.

Questions answered:
1. Does the frozen MorphMAE MRCNN contain more stage information than a frozen random MRCNN?
2. If yes, can we retain that representation with staged/discriminative fine-tuning instead
   of immediately training the pretrained MRCNN at the current global 1e-3 LR?

The existing matched A1 results are read from the completed A1/A2 baseline CSV. No test
subject is opened by this runner. The default folds 0..4 are development folds only; any
selected transfer recipe must later be frozen and evaluated on new folds.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
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
    p.add_argument("--folds", type=_csv_ints, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--mae-checkpoint-pattern", required=True)
    p.add_argument(
        "--baseline-csv",
        default="mist_sleep_runs/mae_baseline_stability/mae_baseline_raw.csv",
        help="Completed matched A1/A2 raw CSV; A1 provides the standard-training reference.",
    )
    p.add_argument("--output-dir", default="mist_sleep_runs/morphmae_transfer_recovery")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--reuse", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["SLEEP_EDF_NPZ_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    os.environ["PROTOMAE_OUT"] = str(Path(args.output_dir).expanduser().resolve())
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.config import MICRO_BATCH_SIZE, MICRO_LR, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.mae_baseline import build_matched_a1_a2, verify_mae_checkpoint_for_fold
    from protosleep.micro import evaluate_micro_loader
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.morphmae_transfer import (
        MAE_PROBE,
        MAE_STAGE_1E4,
        MAE_STAGE_2E5,
        train_attnsleep_transfer,
    )
    from protosleep.utils import DEVICE

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline_csv).expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline CSV not found: {baseline_path}. Run the matched A1/A2 baseline screen first."
        )
    baseline = pd.read_csv(baseline_path)
    required = {"fold", "seed", "A1_f1"}
    missing = required - set(baseline.columns)
    if missing:
        raise RuntimeError(f"Baseline CSV missing columns: {sorted(missing)}")

    key_counts = baseline.groupby(["fold", "seed"]).size()
    duplicates = key_counts[key_counts != 1]
    if len(duplicates):
        raise RuntimeError(f"Baseline CSV must have one row per fold/seed; bad keys: {duplicates.to_dict()}")

    print("=" * 100)
    print("MORPHMAE THEORY-DRIVEN TRANSFER RECOVERY")
    print("=" * 100)
    print(f"device={DEVICE} AMP={USE_AMP}")
    print(f"development folds={args.folds} seeds={args.seeds}")
    print(f"standard supervised head LR={MICRO_LR}")
    print("diagnostic 1: frozen random MRCNN vs frozen MorphMAE MRCNN")
    print("diagnostic 2: MorphMAE staged unfreeze, encoder LR=1e-4")
    print("diagnostic 3: MorphMAE staged unfreeze, encoder LR=2e-5")
    print("warmup: 5 epochs head/TCE only; frozen MRCNN BatchNorm statistics are preserved")
    print("DEVELOPMENT ONLY. Any selected recipe must be frozen before evaluation on new folds.")
    print("DESIGNATED TEST SUBJECT FILES WILL NOT BE OPENED.")
    print("TEST METRICS WILL NOT BE COMPUTED.")
    print("=" * 100)

    rows: List[Dict[str, Any]] = []

    def baseline_a1(fold: int, seed: int) -> float:
        hit = baseline[(baseline["fold"] == fold) & (baseline["seed"] == seed)]
        if len(hit) != 1:
            raise RuntimeError(f"No unique A1 baseline for fold={fold}, seed={seed}")
        return float(hit.iloc[0]["A1_f1"])

    for fold in args.folds:
        split = fold_subjects_from_npz(data_dir, fold)
        train_subjects = split["train_subjects"]
        val_subjects = split["val_subjects"]
        test_subjects = split["test_subjects"]
        mae_path = Path(args.mae_checkpoint_pattern.format(fold=fold)).expanduser().resolve()
        mae_meta = verify_mae_checkpoint_for_fold(mae_path, fold, train_subjects)

        train_recordings, train_io = load_recordings_for_subjects(data_dir, train_subjects)
        val_recordings, val_io = load_recordings_for_subjects(data_dir, val_subjects)
        opened_subjects = sorted(set(train_io["subjects"]) | set(val_io["subjects"]))
        if set(opened_subjects) & set(test_subjects):
            raise RuntimeError(
                f"Leakage barrier failure: test subjects {test_subjects} appear in opened subjects {opened_subjects}"
            )
        class_weights = balanced_class_weights_from_train(train_recordings)

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("train subjects:", train_subjects)
        print("validation subjects:", val_subjects)
        print("designated test subjects:", test_subjects, "(FILES NOT OPENED)")
        print(
            f"opened train recordings={train_io['n_recordings']} epochs={train_io['n_epochs']:,} | "
            f"val recordings={val_io['n_recordings']} epochs={val_io['n_epochs']:,}"
        )
        print("MorphMAE split metadata: MATCH")
        print("=" * 100)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            a1, a2, init_meta = build_matched_a1_a2(actual_seed, mae_path)
            models = {
                "random_probe": (copy.deepcopy(a1), MAE_PROBE),
                "mae_probe": (copy.deepcopy(a2), MAE_PROBE),
                "mae_stage_1e4": (copy.deepcopy(a2), MAE_STAGE_1E4),
                "mae_stage_2e5": (copy.deepcopy(a2), MAE_STAGE_2E5),
            }

            def run_one(name, model, recipe):
                checkpoint = ckpt_dir / f"{name}.pt"
                history_path = log_dir / f"{name}.csv"
                if args.reuse and checkpoint.is_file():
                    payload = torch.load(checkpoint, map_location="cpu")
                    model.load_state_dict(payload["state_dict"])
                    va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
                    metrics = evaluate_micro_loader(model.to(DEVICE), va, DEVICE)
                    return {
                        "model": model.cpu(),
                        "best_val_macro_f1": float(metrics["macro_f1"]),
                        "best_epoch": int(payload["best_epoch"]),
                        "mrcnn_relative_drift": float(payload["mrcnn_relative_drift"]),
                        "seconds": 0.0,
                        "reused": True,
                    }

                tr = make_epoch_loader(train_recordings, MICRO_BATCH_SIZE, shuffle=True)
                va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
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
                        recipe,
                        actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                result["history"].to_csv(history_path, index=False)
                return {
                    "model": result["model"].cpu(),
                    "best_val_macro_f1": float(result["best_val_macro_f1"]),
                    "best_epoch": int(result["best_epoch"]),
                    "mrcnn_relative_drift": float(result["mrcnn_relative_drift"]),
                    "seconds": float(seconds),
                    "reused": False,
                }

            results = {}
            for name in ("random_probe", "mae_probe", "mae_stage_1e4", "mae_stage_2e5"):
                model, recipe = models[name]
                print(f"--- fold={fold} seed={base_seed} | {name} ---")
                results[name] = run_one(name, model, recipe)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            a1_standard = baseline_a1(fold, base_seed)
            row = {
                "fold": int(fold),
                "seed": int(base_seed),
                "actual_seed": int(actual_seed),
                "train_subjects": ",".join(map(str, train_subjects)),
                "val_subjects": ",".join(map(str, val_subjects)),
                "test_subjects_locked": ",".join(map(str, test_subjects)),
                "test_subject_files_opened": False,
                "mae_checkpoint": str(mae_path),
                "mae_checkpoint_sha256": mae_meta["sha256"],
                "pretrain_split_verified": True,
                "A1_standard_f1": float(a1_standard),
                "random_probe_f1": results["random_probe"]["best_val_macro_f1"],
                "mae_probe_f1": results["mae_probe"]["best_val_macro_f1"],
                "probe_gain": results["mae_probe"]["best_val_macro_f1"] - results["random_probe"]["best_val_macro_f1"],
                "mae_stage_1e4_f1": results["mae_stage_1e4"]["best_val_macro_f1"],
                "stage_1e4_minus_A1": results["mae_stage_1e4"]["best_val_macro_f1"] - a1_standard,
                "mae_stage_2e5_f1": results["mae_stage_2e5"]["best_val_macro_f1"],
                "stage_2e5_minus_A1": results["mae_stage_2e5"]["best_val_macro_f1"] - a1_standard,
                "random_probe_drift": results["random_probe"]["mrcnn_relative_drift"],
                "mae_probe_drift": results["mae_probe"]["mrcnn_relative_drift"],
                "stage_1e4_drift": results["mae_stage_1e4"]["mrcnn_relative_drift"],
                "stage_2e5_drift": results["mae_stage_2e5"]["mrcnn_relative_drift"],
                "random_probe_best_epoch": results["random_probe"]["best_epoch"],
                "mae_probe_best_epoch": results["mae_probe"]["best_epoch"],
                "stage_1e4_best_epoch": results["mae_stage_1e4"]["best_epoch"],
                "stage_2e5_best_epoch": results["mae_stage_2e5"]["best_epoch"],
                "non_mrcnn_initialization_match": True,
                "amp": bool(USE_AMP),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "transfer_recovery_partial.csv", index=False)

            with (run_dir / "provenance.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **row,
                        "mae_metadata": mae_meta,
                        "matched_initialization": init_meta,
                        "train_io": train_io,
                        "val_io": val_io,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            print(
                f"fold={fold} seed={base_seed} | randomProbe={row['random_probe_f1']:.4f} "
                f"maeProbe={row['mae_probe_f1']:.4f} probeGain={row['probe_gain']:+.4f} | "
                f"A1std={a1_standard:.4f} stage1e4={row['mae_stage_1e4_f1']:.4f} "
                f"({row['stage_1e4_minus_A1']:+.4f}) stage2e5={row['mae_stage_2e5_f1']:.4f} "
                f"({row['stage_2e5_minus_A1']:+.4f})"
            )

            del a1, a2, models, results

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "transfer_recovery_raw.csv", index=False)

    mean_cols = [
        "A1_standard_f1",
        "random_probe_f1",
        "mae_probe_f1",
        "probe_gain",
        "mae_stage_1e4_f1",
        "stage_1e4_minus_A1",
        "mae_stage_2e5_f1",
        "stage_2e5_minus_A1",
        "stage_1e4_drift",
        "stage_2e5_drift",
    ]
    fold_means = raw.groupby("fold", as_index=False)[mean_cols].mean()
    fold_means.to_csv(output_root / "transfer_recovery_fold_means.csv", index=False)

    effect_rows = []
    for col in ("probe_gain", "stage_1e4_minus_A1", "stage_2e5_minus_A1"):
        x = fold_means[col].to_numpy(float)
        effect_rows.append(
            {
                "effect": col,
                "mean": float(np.mean(x)),
                "std_across_folds": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)),
                "positive_folds": int(np.sum(x > 0)),
                "n_folds": int(len(x)),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output_root / "transfer_recovery_effects.csv", index=False)

    staged_effects = effects[effects["effect"].isin(["stage_1e4_minus_A1", "stage_2e5_minus_A1"])]
    selected = staged_effects.sort_values(["mean", "positive_folds"], ascending=False).iloc[0]
    representation = effects[effects["effect"] == "probe_gain"].iloc[0]
    summary = {
        "development_only": True,
        "representation_probe": {
            "mean_gain": float(representation["mean"]),
            "positive_folds": int(representation["positive_folds"]),
            "passes_exploratory_gate": bool(
                representation["mean"] > 0 and representation["positive_folds"] >= 3
            ),
        },
        "selected_staged_recipe": str(selected["effect"]).replace("_minus_A1", ""),
        "selected_staged_mean_gain_vs_A1": float(selected["mean"]),
        "selected_staged_positive_folds": int(selected["positive_folds"]),
        "passes_exploratory_transfer_gate": bool(selected["mean"] > 0 and selected["positive_folds"] >= 3),
        "next_step_if_pass": "Freeze selected recipe and evaluate on new folds with newly trained fold-specific MorphMAE checkpoints.",
        "next_step_if_probe_passes_but_transfer_fails": "Representation exists; investigate retention/regularization, not a new SSL objective yet.",
        "next_step_if_probe_fails": "Redesign MorphMAE pretraining objective before more downstream tuning.",
        "test_metrics_computed": False,
        "designated_test_subject_files_opened": False,
    }
    (output_root / "transfer_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("SEED-AVERAGED DEVELOPMENT FOLD RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("DEVELOPMENT EFFECT SUMMARY")
    print("=" * 100)
    print(effects.to_string(index=False))
    print("\n" + "=" * 100)
    print("THEORY DECISION")
    print("=" * 100)
    print(json.dumps(summary, indent=2))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("CONFIRMATORY CLAIM ALLOWED FROM THIS SCRIPT: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen representation probe for a refined MorphMAE checkpoint.

The random-probe reference is read from the already completed transfer-recovery experiment.
Only the refined MorphMAE probe is trained here, so this is much cheaper than rerunning the
staged-transfer grid. The MRCNN remains frozen and in eval mode throughout; only the matched
AttnSleep TCE/classifier are optimized.
"""
from __future__ import annotations

import argparse
import contextlib
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
    p.add_argument("--refined-checkpoint-pattern", required=True)
    p.add_argument(
        "--reference-csv",
        default="mist_sleep_runs/morphmae_transfer_recovery/transfer_recovery_raw.csv",
    )
    p.add_argument("--output-dir", default="mist_sleep_runs/morphmae_morphspec_r1_probe")
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

    from protosleep.config import MICRO_BATCH_SIZE, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.mae_baseline import build_matched_a1_a2, verify_mae_checkpoint_for_fold
    from protosleep.micro import evaluate_micro_loader
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.morphmae_transfer import MAE_PROBE, train_attnsleep_transfer
    from protosleep.utils import DEVICE

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference_csv).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = pd.read_csv(reference_path)
    required = {"fold", "seed", "random_probe_f1", "mae_probe_f1"}
    missing = required - set(reference.columns)
    if missing:
        raise RuntimeError(f"Reference CSV missing columns: {sorted(missing)}")
    if (reference.groupby(["fold", "seed"]).size() != 1).any():
        raise RuntimeError("Reference CSV must contain exactly one row per fold/seed")

    def ref_value(fold: int, seed: int, col: str) -> float:
        hit = reference[(reference["fold"] == fold) & (reference["seed"] == seed)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing unique reference fold={fold} seed={seed}")
        return float(hit.iloc[0][col])

    print("=" * 100)
    print("MORPHMAE MORPHSPEC-R1 FROZEN REPRESENTATION PROBE")
    print("=" * 100)
    print(f"device={DEVICE} AMP={USE_AMP}")
    print(f"development folds={args.folds} seeds={args.seeds}")
    print("MRCNN is frozen and kept in eval mode for the entire probe.")
    print("Reference random-probe values are reused from the completed transfer-recovery audit.")
    print("DESIGNATED TEST SUBJECT FILES WILL NOT BE OPENED.")
    print("TEST METRICS WILL NOT BE COMPUTED.")
    print("=" * 100)

    rows: List[Dict[str, Any]] = []
    for fold in args.folds:
        split = fold_subjects_from_npz(data_dir, fold)
        checkpoint = Path(args.refined_checkpoint_pattern.format(fold=fold)).expanduser().resolve()
        meta = verify_mae_checkpoint_for_fold(checkpoint, fold, split["train_subjects"])

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
        print("refined checkpoint:", checkpoint)
        print("=" * 100)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            out_ckpt = ckpt_dir / "morphspec_probe.pt"

            _, refined_model, init_meta = build_matched_a1_a2(actual_seed, checkpoint)

            if args.reuse and out_ckpt.is_file():
                payload = torch.load(out_ckpt, map_location="cpu")
                refined_model.load_state_dict(payload["state_dict"])
                va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
                metrics = evaluate_micro_loader(refined_model.to(DEVICE), va, DEVICE)
                refined_f1 = float(metrics["macro_f1"])
                best_epoch = int(payload["best_epoch"])
                seconds = 0.0
                reused = True
            else:
                tr = make_epoch_loader(train_recordings, MICRO_BATCH_SIZE, shuffle=True)
                va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with (log_dir / "morphspec_probe.log").open("w", buffering=1) as f, contextlib.redirect_stdout(f):
                    result = train_attnsleep_transfer(
                        refined_model,
                        tr,
                        va,
                        class_weights,
                        out_ckpt,
                        MAE_PROBE,
                        actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                refined_f1 = float(result["best_val_macro_f1"])
                best_epoch = int(result["best_epoch"])
                result["history"].to_csv(log_dir / "morphspec_probe.csv", index=False)
                reused = False

            random_ref = ref_value(fold, base_seed, "random_probe_f1")
            original_mae = ref_value(fold, base_seed, "mae_probe_f1")
            row = {
                "fold": int(fold),
                "seed": int(base_seed),
                "actual_seed": int(actual_seed),
                "random_probe_f1": random_ref,
                "original_mae_probe_f1": original_mae,
                "morphspec_probe_f1": refined_f1,
                "morphspec_minus_random": float(refined_f1 - random_ref),
                "morphspec_minus_original_mae": float(refined_f1 - original_mae),
                "best_epoch": best_epoch,
                "seconds": float(seconds),
                "reused": bool(reused),
                "test_subject_files_opened": False,
                "pretrain_split_verified": True,
                "non_mrcnn_initialization_match": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": meta["sha256"],
                "amp": bool(USE_AMP),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "morphspec_probe_partial.csv", index=False)
            (run_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        **row,
                        "checkpoint_metadata": meta,
                        "matched_initialization": init_meta,
                        "train_io": train_io,
                        "val_io": val_io,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(
                f"fold={fold} seed={base_seed} | random={random_ref:.4f} "
                f"originalMAE={original_mae:.4f} morphspec={refined_f1:.4f} | "
                f"vsRandom={refined_f1-random_ref:+.4f} vsOriginalMAE={refined_f1-original_mae:+.4f}"
            )

            del refined_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "morphspec_probe_raw.csv", index=False)
    cols = [
        "random_probe_f1",
        "original_mae_probe_f1",
        "morphspec_probe_f1",
        "morphspec_minus_random",
        "morphspec_minus_original_mae",
    ]
    fold_means = raw.groupby("fold", as_index=False)[cols].mean()
    fold_means.to_csv(output_root / "morphspec_probe_fold_means.csv", index=False)

    effects = []
    for col in ("morphspec_minus_random", "morphspec_minus_original_mae"):
        x = fold_means[col].to_numpy(float)
        effects.append(
            {
                "effect": col,
                "mean": float(np.mean(x)),
                "std_across_folds": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)),
                "positive_folds": int(np.sum(x > 0)),
                "n_folds": int(len(x)),
            }
        )
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(output_root / "morphspec_probe_effects.csv", index=False)

    primary = effects_df[effects_df["effect"] == "morphspec_minus_random"].iloc[0]
    decision = {
        "development_only": True,
        "mean_morphspec_minus_random": float(primary["mean"]),
        "positive_folds_vs_random": int(primary["positive_folds"]),
        "passes_exploratory_representation_gate": bool(
            float(primary["mean"]) > 0 and int(primary["positive_folds"]) >= 3
        ),
        "gate": "mean MorphSpec frozen-probe gain over random > 0 and positive in at least 3/5 development folds",
        "test_metrics_computed": False,
        "designated_test_subject_files_opened": False,
    }
    (output_root / "morphspec_probe_summary.json").write_text(json.dumps(decision, indent=2))

    print("\n" + "=" * 100)
    print("SEED-AVERAGED MORPHSPEC PROBE RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("MORPHSPEC PROBE EFFECT SUMMARY")
    print("=" * 100)
    print(effects_df.to_string(index=False))
    print("\n" + "=" * 100)
    print("EXPLORATORY DECISION")
    print("=" * 100)
    print(json.dumps(decision, indent=2))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

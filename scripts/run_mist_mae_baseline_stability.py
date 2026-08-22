#!/usr/bin/env python3
"""Validation-only matched A1/A2 screen for MorphMAE initialization.

This is the missing causal isolation after the prototype A4 screens:
  A1 = AttnSleep with random MRCNN
  A2 = the same AttnSleep initialization, with only MRCNN replaced by fold-specific MorphMAE-v2

The runner uses the current supervised AttnSleep protocol so A1 is directly comparable to the
current MIST stability audit. Fold-specific test subjects are resolved from filenames only;
their NPZ arrays are never opened and no test metric is computed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List


def _csv_ints(value: str) -> List[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/mae_baseline_stability")
    p.add_argument("--folds", type=_csv_ints, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    ck = p.add_mutually_exclusive_group(required=True)
    ck.add_argument(
        "--mae-checkpoint",
        help="Single fold-specific checkpoint; allowed only when exactly one fold is requested.",
    )
    ck.add_argument(
        "--mae-checkpoint-pattern",
        help="Fold-aware path template, e.g. '.../fold_{fold:02d}/ssl_seed_1337/encoder.pt'.",
    )
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def resolve_checkpoint(args, fold: int) -> Path:
    if args.mae_checkpoint:
        if len(args.folds) != 1:
            raise RuntimeError("--mae-checkpoint can only be used with one fold")
        return Path(args.mae_checkpoint).expanduser().resolve()
    return Path(args.mae_checkpoint_pattern.format(fold=fold)).expanduser().resolve()


def main():
    args = parse_args()

    # Match the current AttnSleep training stack before importing protosleep.config/utils.
    os.environ["SLEEP_EDF_NPZ_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    os.environ["PROTOMAE_OUT"] = str(Path(args.output_dir).expanduser().resolve())
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.config import MICRO_BATCH_SIZE, PROJECT_VERSION, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.mae_baseline import build_matched_a1_a2, verify_mae_checkpoint_for_fold
    from protosleep.micro import train_micro_model
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.utils import DEVICE

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MORPHMAE ATTNSLEEP BASELINE STABILITY: MATCHED A1 VS A2")
    print("=" * 100)
    print(f"device: {DEVICE} | AMP={USE_AMP}")
    print(f"folds: {args.folds} | supervised seeds: {args.seeds}")
    print("A1 = random AttnSleep MRCNN")
    print("A2 = identical AttnSleep initialization except strict fold-specific MorphMAE-v2 MRCNN")
    print(f"current supervised protocol: batch={MICRO_BATCH_SIZE}; validation Macro-F1 selection")
    print("DESIGNATED TEST SUBJECT NPZ FILES WILL NOT BE OPENED.")
    print("TEST METRICS WILL NOT BE COMPUTED.")
    print("=" * 100)

    rows: List[Dict] = []

    for fold in args.folds:
        split = fold_subjects_from_npz(data_dir, fold)
        train_subjects = split["train_subjects"]
        val_subjects = split["val_subjects"]
        test_subjects = split["test_subjects"]
        mae_path = resolve_checkpoint(args, fold)
        mae_meta = verify_mae_checkpoint_for_fold(mae_path, fold, train_subjects)

        # Strict I/O barrier: open only train + validation subjects.
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

            def run_train(name, model, path):
                # Fresh loaders plus train_micro_model(seed=actual_seed) give both pair members
                # the same seeded shuffle/stochastic training protocol.
                tr = make_epoch_loader(train_recordings, MICRO_BATCH_SIZE, shuffle=True)
                va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
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
                        proto_cfg=None,
                        seed=actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                return result["model"].cpu(), float(result["best_val_macro_f1"]), int(result["best_epoch"]), seconds

            a1, f1_a1, e1, t1 = run_train("A1_attnsleep", a1, ckpt_dir / "A1_attnsleep.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            a2, f1_a2, e2, t2 = run_train("A2_morphmae", a2, ckpt_dir / "A2_morphmae_attnsleep.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                "A1_f1": f1_a1,
                "A2_f1": f1_a2,
                "A2_minus_A1": float(f1_a2 - f1_a1),
                "A1_best_epoch": e1,
                "A2_best_epoch": e2,
                "A1_seconds": float(t1),
                "A2_seconds": float(t2),
                "a1_random_mrcnn_sha256": init_meta["a1_random_mrcnn_sha256"],
                "a2_mae_mrcnn_sha256": init_meta["a2_mae_mrcnn_sha256"],
                "non_mrcnn_initialization_match": True,
                "project_version": PROJECT_VERSION,
                "amp": bool(USE_AMP),
                "selection_metric": "validation_macro_f1",
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "mae_baseline_partial.csv", index=False)

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
                f"fold={fold} seed={base_seed} | A1={f1_a1:.4f} | A2={f1_a2:.4f} | "
                f"A2-A1={f1_a2 - f1_a1:+.4f} | best epochs A1/A2={e1}/{e2}"
            )

            del a1, a2

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "mae_baseline_raw.csv", index=False)

    fold_rows = []
    for fold, group in raw.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "A1_f1": float(group["A1_f1"].mean()),
                "A2_f1": float(group["A2_f1"].mean()),
                "A1_seed_sd": float(group["A1_f1"].std(ddof=1)) if len(group) > 1 else np.nan,
                "A2_seed_sd": float(group["A2_f1"].std(ddof=1)) if len(group) > 1 else np.nan,
                "A2_minus_A1": float(group["A2_minus_A1"].mean()),
                "positive_seeds": int((group["A2_minus_A1"] > 0).sum()),
                "n_seeds": int(len(group)),
            }
        )
    fold_means = pd.DataFrame(fold_rows)
    fold_means.to_csv(output_root / "mae_baseline_fold_means.csv", index=False)

    effects = pd.DataFrame(
        [
            {
                "effect": "A2_minus_A1",
                "mean": float(fold_means["A2_minus_A1"].mean()),
                "std_across_folds": float(fold_means["A2_minus_A1"].std(ddof=1)) if len(fold_means) > 1 else np.nan,
                "median": float(fold_means["A2_minus_A1"].median()),
                "positive_folds": int((fold_means["A2_minus_A1"] > 0).sum()),
                "n_folds": int(len(fold_means)),
            }
        ]
    )
    effects.to_csv(output_root / "mae_baseline_effects.csv", index=False)

    print("\n" + "=" * 100)
    print("SEED-AVERAGED FOLD RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("PAIRED EFFECT SUMMARY")
    print("=" * 100)
    print(effects.to_string(index=False))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("External-transfer conclusion allowed from this script: NO")
    print(f"Results saved under: {output_root}")


if __name__ == "__main__":
    main()

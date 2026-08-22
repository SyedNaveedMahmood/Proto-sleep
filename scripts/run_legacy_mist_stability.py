#!/usr/bin/env python3
"""Validation-only matched recovery of historical WaveSleepNet A3/A4.

This runner is the next gate after the compatibility and one-step objective smokes. It uses:
- the audited historical WaveSleepNet ProtoPNet architecture,
- the exact archived OneFoldTrainer.protop_loss,
- historical Adam/batch-size/LR/weight-decay settings,
- fold-specific leakage-safe MorphMAE MRCNN initialization for A4,
- matched A3/A4 initialization everywhere outside MRCNN,
- validation Macro-F1 for checkpoint selection so the recovered mechanism experiment uses
  the same selection target as the current MIST stability audit.

The designated test subject is resolved from filenames but its NPZ files are never opened.
No test metrics are computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _csv_ints(value: str) -> List[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/legacy_stability")
    p.add_argument("--folds", type=_csv_ints, default=[0])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    ck = p.add_mutually_exclusive_group(required=True)
    ck.add_argument(
        "--mae-checkpoint",
        help="Single fold-specific MorphMAE checkpoint; allowed only when one fold is requested.",
    )
    ck.add_argument(
        "--mae-checkpoint-pattern",
        help="Path template such as '.../fold_{fold:02d}/ssl_seed_1337/encoder.pt'.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Development-only override. Omit for the historical max_epochs from config.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Development-only override. Omit for the historical early-stopping patience.",
    )
    return p.parse_args()


def resolve_checkpoint(args, fold: int) -> Path:
    if args.mae_checkpoint:
        if len(args.folds) != 1:
            raise RuntimeError("--mae-checkpoint can only be used when exactly one fold is requested")
        return Path(args.mae_checkpoint).expanduser().resolve()
    return Path(args.mae_checkpoint_pattern.format(fold=fold)).expanduser().resolve()


def main():
    args = parse_args()

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.legacy_wavesleepnet import (
        build_matched_legacy_a3_a4,
        validate_legacy_snapshot,
    )
    from protosleep.legacy_wavesleepnet_objective import (
        historical_training_config,
        import_legacy_trainer_module,
    )
    from protosleep.legacy_wavesleepnet_train import (
        historical_training_protocol,
        load_recordings_for_subjects,
        train_legacy_model,
        verify_mae_checkpoint_for_fold,
    )
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.utils import DEVICE

    legacy_root = Path(args.legacy_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot = validate_legacy_snapshot(legacy_root)
    trainer_module = import_legacy_trainer_module(legacy_root)
    cfg = historical_training_config(legacy_root)
    protocol = historical_training_protocol(cfg, args.max_epochs, args.patience)

    print("=" * 100)
    print("RECOVERED HISTORICAL WAVESLEEPNET MIST STABILITY")
    print("=" * 100)
    print(f"device: {DEVICE}")
    print(f"folds: {args.folds}")
    print(f"supervised seeds: {args.seeds}")
    print("pair: A3_legacy_recovered (random MRCNN) vs A4_legacy_recovered (MorphMAE MRCNN)")
    print("objective: exact archived OneFoldTrainer.protop_loss")
    print(
        "optimizer/config: Adam "
        f"lr={protocol['lr']} weight_decay={protocol['weight_decay']} "
        f"batch={protocol['batch_size']} FP32"
    )
    print(
        f"training limit: max_epochs={protocol['max_epochs']} patience={protocol['patience']} "
        "selection=validation_macro_f1"
    )
    if protocol["max_epochs_overridden"] or protocol["patience_overridden"]:
        print("WARNING: DEVELOPMENT OVERRIDE ACTIVE; this run is not the frozen historical-config screen.")
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

        # Only train and validation files are opened. The test subject remains filename-only metadata.
        train_recordings, train_io = load_recordings_for_subjects(data_dir, train_subjects)
        val_recordings, val_io = load_recordings_for_subjects(data_dir, val_subjects)
        opened_subjects = sorted(set(train_io["subjects"]) | set(val_io["subjects"]))
        if set(opened_subjects) & set(test_subjects):
            raise RuntimeError(
                f"Leakage barrier failure: designated test subjects {test_subjects} appear in opened subjects {opened_subjects}"
            )

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

            a3, a4, init_meta = build_matched_legacy_a3_a4(
                legacy_root,
                mae_path,
                actual_seed,
            )

            print(f"\n--- fold={fold} seed={base_seed} actual_seed={actual_seed} | A3_legacy_recovered ---")
            r3 = train_legacy_model(
                a3,
                train_recordings,
                val_recordings,
                trainer_module,
                cfg,
                DEVICE,
                actual_seed,
                ckpt_dir / "A3_legacy_recovered.pt",
                history_path=log_dir / "A3_legacy_recovered.csv",
                max_epochs_override=args.max_epochs,
                patience_override=args.patience,
            )
            a3 = r3["model"].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n--- fold={fold} seed={base_seed} actual_seed={actual_seed} | A4_legacy_recovered ---")
            r4 = train_legacy_model(
                a4,
                train_recordings,
                val_recordings,
                trainer_module,
                cfg,
                DEVICE,
                actual_seed,
                ckpt_dir / "A4_legacy_recovered.pt",
                history_path=log_dir / "A4_legacy_recovered.csv",
                max_epochs_override=args.max_epochs,
                patience_override=args.patience,
            )
            a4 = r4["model"].cpu()
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
                "A3_legacy_f1": float(r3["best_val_macro_f1"]),
                "A4_legacy_f1": float(r4["best_val_macro_f1"]),
                "A4_minus_A3": float(r4["best_val_macro_f1"] - r3["best_val_macro_f1"]),
                "A3_best_epoch": int(r3["best_epoch"]),
                "A4_best_epoch": int(r4["best_epoch"]),
                "A3_seconds": float(r3["seconds"]),
                "A4_seconds": float(r4["seconds"]),
                "a3_random_mrcnn_sha256": init_meta["a3_random_mrcnn_sha256"],
                "a4_mae_mrcnn_sha256": init_meta["a4_mae_mrcnn_sha256"],
                "objective_source_sha256": snapshot["trainer"]["sha256"],
                "selection_metric": protocol["selection_metric"],
                "optimizer": protocol["optimizer"],
                "lr": protocol["lr"],
                "weight_decay": protocol["weight_decay"],
                "batch_size": protocol["batch_size"],
                "max_epochs": protocol["max_epochs"],
                "patience": protocol["patience"],
                "precision": protocol["precision"],
                "protocol_override": bool(
                    protocol["max_epochs_overridden"] or protocol["patience_overridden"]
                ),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "legacy_stability_partial.csv", index=False)

            with (run_dir / "provenance.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **row,
                        "legacy_snapshot": snapshot,
                        "mae_metadata": mae_meta,
                        "matched_initialization": init_meta,
                        "protocol": protocol,
                        "train_io": train_io,
                        "val_io": val_io,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            print(
                f"fold={fold} seed={base_seed} | "
                f"A3_legacy={row['A3_legacy_f1']:.4f} | "
                f"A4_legacy={row['A4_legacy_f1']:.4f} | "
                f"A4-A3={row['A4_minus_A3']:+.4f} | "
                f"epochs A3/A4={row['A3_best_epoch']}/{row['A4_best_epoch']}"
            )

            del a3, a4
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "legacy_stability_raw.csv", index=False)

    fold_rows = []
    for fold, g in raw.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "A3_legacy_f1": float(g["A3_legacy_f1"].mean()),
                "A4_legacy_f1": float(g["A4_legacy_f1"].mean()),
                "A3_seed_sd": float(g["A3_legacy_f1"].std(ddof=1)) if len(g) > 1 else np.nan,
                "A4_seed_sd": float(g["A4_legacy_f1"].std(ddof=1)) if len(g) > 1 else np.nan,
                "A4_minus_A3": float(g["A4_minus_A3"].mean()),
                "positive_seeds": int((g["A4_minus_A3"] > 0).sum()),
                "n_seeds": int(len(g)),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_root / "legacy_stability_fold_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("RECOVERED HISTORICAL A3/A4 FOLD SUMMARY")
    print("=" * 100)
    print(fold_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("PAIRED EFFECT SUMMARY")
    print("=" * 100)
    effects = fold_summary["A4_minus_A3"].to_numpy(dtype=float)
    print(f"mean fold-level A4-A3: {float(np.mean(effects)):+.6f}")
    print(f"median fold-level A4-A3: {float(np.median(effects)):+.6f}")
    print(f"positive folds: {int(np.sum(effects > 0))}/{len(effects)}")
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("External-transfer conclusion allowed from this script: NO")
    if protocol["max_epochs_overridden"] or protocol["patience_overridden"]:
        print("PROTOCOL OVERRIDE ACTIVE: YES (development-only result)")
    else:
        print("PROTOCOL OVERRIDE ACTIVE: NO")
    print(f"Results saved under: {output_root}")


if __name__ == "__main__":
    main()

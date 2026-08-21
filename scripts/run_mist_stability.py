#!/usr/bin/env python3
"""
Validation-only MIST mechanism stability audit.

This is the scientifically valid step before external transfer: repeat A1 and a matched
prototype A3/A4 comparison across supervised seeds/folds while leaving each fold's test
subject untouched.

Important naming note
---------------------
The repository's current ProtoAttnSleep implementation is used for the prototype pathway.
Therefore the output labels are A3_current and A4_current rather than claiming byte-for-byte
identity with the historical WaveSleepNet-derived A3/A4 implementation described in the
original proposal. The causal manipulation is nevertheless clean: A3_current and A4_current
are identical at initialization except that A4_current receives an exact MAE-pretrained MRCNN.
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
    p.add_argument("--data-dir", required=True, help="Sleep-EDF NPZ directory.")
    p.add_argument("--output-dir", default="./mist_sleep_runs")
    p.add_argument("--folds", type=_csv_ints, default=[0], help="Example: 0 or 0,1,2")
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789], help="Supervised seeds.")
    ck = p.add_mutually_exclusive_group(required=True)
    ck.add_argument(
        "--mae-checkpoint",
        help="Single MAE checkpoint. Allowed only when one fold is requested.",
    )
    ck.add_argument(
        "--mae-checkpoint-pattern",
        help=(
            "Fold/seed-aware path template, e.g. '/x/fold_{fold:02d}/seed_{seed}/encoder.pt'. "
            "If {seed} is omitted, the same fold-specific SSL checkpoint is reused across supervised seeds."
        ),
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--allow-unverified-pretrain-split",
        action="store_true",
        help=(
            "Development-only escape hatch when the old checkpoint lacks train-subject metadata. "
            "The run will be marked split_verified=false."
        ),
    )
    p.add_argument("--reuse", action="store_true", help="Reuse completed A1/A3/A4 checkpoints.")
    return p.parse_args()


def resolve_checkpoint(args, fold: int, seed: int) -> Path:
    if args.mae_checkpoint:
        if len(args.folds) != 1:
            raise RuntimeError("--mae-checkpoint can only be used with a single fold")
        return Path(args.mae_checkpoint).expanduser().resolve()
    return Path(args.mae_checkpoint_pattern.format(fold=fold, seed=seed)).expanduser().resolve()


def main():
    args = parse_args()

    # Set package-level runtime config before importing protosleep modules.
    os.environ["SLEEP_EDF_NPZ_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    os.environ["PROTOMAE_OUT"] = str(Path(args.output_dir).expanduser().resolve())
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"
    os.environ["PROTOSLEEP_REUSE"] = "1" if args.reuse else "0"

    import numpy as np
    import pandas as pd
    import torch

    from protosleep.attnsleep import AttnSleepBaseline
    from protosleep.cache import load_model_state
    from protosleep.config import (
        ENABLE_MICRO_MASK,
        MICRO_BATCH_SIZE,
        PROJECT_VERSION,
        PROTO_TRIALS,
        USE_AMP,
    )
    from protosleep.data import (
        balanced_class_weights_from_train,
        load_sleep_edf_recordings,
        make_epoch_loader,
        split_subjects,
    )
    from protosleep.micro import evaluate_micro_loader, train_micro_model
    from protosleep.mist import build_a1, build_matched_a3_a4, extract_mrcnn_state_dict
    from protosleep.prototypes import ProtoAttnSleep
    from protosleep.utils import DEVICE

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    recordings = load_sleep_edf_recordings(args.data_dir)
    rows: List[Dict] = []

    print("=" * 100)
    print("MIST MECHANISM STABILITY AUDIT")
    print(f"folds={args.folds} | seeds={args.seeds} | AMP={USE_AMP}")
    print("VALIDATION ONLY. DESIGNATED TEST SUBJECTS ARE NOT LOADED INTO ANY EVALUATION LOADER.")
    print("=" * 100)

    for fold in args.folds:
        split = split_subjects(recordings, fold)
        class_weights = balanced_class_weights_from_train(split["train"])

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("train subjects:", split["train_subjects"])
        print("validation subjects:", split["val_subjects"])
        print("designated test subjects:", split["test_subjects"], "(UNTOUCHED)")
        print("=" * 100)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            mae_path = resolve_checkpoint(args, fold, base_seed)
            _, mae_meta = extract_mrcnn_state_dict(mae_path)

            declared_subjects = None
            for key in ("train_subjects", "pretrain_subjects", "subjects"):
                if key in mae_meta:
                    declared_subjects = sorted(int(x) for x in mae_meta[key])
                    break
            split_verified = declared_subjects is not None and declared_subjects == sorted(split["train_subjects"])
            if declared_subjects is not None and not split_verified:
                raise RuntimeError(
                    f"MAE checkpoint subjects {declared_subjects} do not equal fold-{fold} training subjects "
                    f"{sorted(split['train_subjects'])}. Refusing a leakage-prone A4 run."
                )
            if declared_subjects is None and not args.allow_unverified_pretrain_split:
                raise RuntimeError(
                    f"Checkpoint {mae_path} has no train/pretrain subject metadata. "
                    "Cannot verify subject-disjoint MAE pretraining. Inspect the checkpoint first, or rerun with "
                    "--allow-unverified-pretrain-split for a clearly marked DEVELOPMENT-ONLY pilot."
                )

            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            def loaders():
                return (
                    make_epoch_loader(split["train"], MICRO_BATCH_SIZE, shuffle=True),
                    make_epoch_loader(split["val"], MICRO_BATCH_SIZE, shuffle=False),
                )

            def run_train(name, model, path, proto_cfg=None):
                if args.reuse and path.exists():
                    if isinstance(model, ProtoAttnSleep):
                        fresh = ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)
                    else:
                        fresh = AttnSleepBaseline()
                    loaded = load_model_state(fresh, path, DEVICE)
                    metrics = evaluate_micro_loader(loaded, loaders()[1], DEVICE)
                    return loaded.cpu(), metrics, 0.0, True

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
                trained = result["model"].to(DEVICE).eval()
                metrics = evaluate_micro_loader(trained, va, DEVICE)
                return trained.cpu(), metrics, seconds, False

            # A1 is a baseline architecture. A3/A4 are the matched prototype causal pair.
            a1 = build_a1(actual_seed)
            a3, a4, init_meta = build_matched_a3_a4(actual_seed, mae_path)

            a1, m1, t1, r1 = run_train("A1", a1, ckpt_dir / "A1_attnsleep.pt")
            a3, m3, t3, r3 = run_train(
                "A3_current",
                a3,
                ckpt_dir / "A3_current_proto_random_mrcnn.pt",
                proto_cfg=PROTO_TRIALS[0],
            )
            a4, m4, t4, r4 = run_train(
                "A4_current",
                a4,
                ckpt_dir / "A4_current_proto_mae_mrcnn.pt",
                proto_cfg=PROTO_TRIALS[0],
            )

            row = {
                "fold": fold,
                "seed": base_seed,
                "actual_seed": actual_seed,
                "val_subjects": ",".join(map(str, split["val_subjects"])),
                "test_subjects_locked": ",".join(map(str, split["test_subjects"])),
                "mae_checkpoint": str(mae_path),
                "mae_checkpoint_sha256": mae_meta["sha256"],
                "pretrain_split_verified": bool(split_verified),
                "A1_f1": float(m1["macro_f1"]),
                "A3_current_f1": float(m3["macro_f1"]),
                "A4_current_f1": float(m4["macro_f1"]),
                "A3_minus_A1": float(m3["macro_f1"] - m1["macro_f1"]),
                "A4_minus_A3": float(m4["macro_f1"] - m3["macro_f1"]),
                "A4_minus_A1": float(m4["macro_f1"] - m1["macro_f1"]),
                "A1_seconds": t1,
                "A3_seconds": t3,
                "A4_seconds": t4,
                "A1_reused": r1,
                "A3_reused": r3,
                "A4_reused": r4,
                "a3_random_mrcnn_sha256": init_meta["a3_random_mrcnn_sha256"],
                "a4_mae_mrcnn_sha256": init_meta["a4_mae_mrcnn_sha256"],
                "project_version": PROJECT_VERSION,
                "amp": bool(USE_AMP),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "mist_stability_partial.csv", index=False)

            with (run_dir / "provenance.json").open("w") as f:
                json.dump({**row, "mae_metadata": mae_meta, "init_metadata": init_meta}, f, indent=2, default=str)

            print(
                f"fold={fold} seed={base_seed} | A1={row['A1_f1']:.4f} | "
                f"A3_current={row['A3_current_f1']:.4f} | A4_current={row['A4_current_f1']:.4f} | "
                f"A4-A3={row['A4_minus_A3']:+.4f}"
            )

            del a1, a3, a4
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "mist_stability_raw.csv", index=False)

    # Average supervised seeds within each fold first; folds/subjects are the experimental units.
    fold_means = (
        raw.groupby("fold", as_index=False)
        .agg(
            A1_f1=("A1_f1", "mean"),
            A3_current_f1=("A3_current_f1", "mean"),
            A4_current_f1=("A4_current_f1", "mean"),
            A1_seed_sd=("A1_f1", "std"),
            A3_seed_sd=("A3_current_f1", "std"),
            A4_seed_sd=("A4_current_f1", "std"),
        )
    )
    fold_means["A3_minus_A1"] = fold_means["A3_current_f1"] - fold_means["A1_f1"]
    fold_means["A4_minus_A3"] = fold_means["A4_current_f1"] - fold_means["A3_current_f1"]
    fold_means["A4_minus_A1"] = fold_means["A4_current_f1"] - fold_means["A1_f1"]
    fold_means.to_csv(output_root / "mist_stability_fold_means.csv", index=False)

    effect_rows = []
    for col in ("A3_minus_A1", "A4_minus_A3", "A4_minus_A1"):
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
    effects.to_csv(output_root / "mist_stability_effects.csv", index=False)

    print("\n" + "=" * 100)
    print("SEED-AVERAGED FOLD RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("PAIRED EFFECT SUMMARY")
    print("=" * 100)
    print(effects.to_string(index=False))
    print("\nTEST METRICS COMPUTED: NO")
    print("External-transfer conclusion allowed from this script: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

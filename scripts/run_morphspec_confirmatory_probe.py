#!/usr/bin/env python3
"""Held-out-fold confirmation of the frozen MorphSpec-R1 representation.

This script is intentionally narrower than the exploratory transfer-recovery runner.
It evaluates one frozen recipe on folds that were not used to select MorphSpec-R1:

  random_probe       random MRCNN, frozen
  original_mae_probe leakage-safe MorphMAE-v2 MRCNN, frozen
  morphspec_probe    frozen MorphSpec-R1 MRCNN

All three models have byte-identical initialized TCE/classifier state within each fold/seed.
Only train + validation NPZ files are opened. The designated test subject is never evaluated.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--folds", type=_csv_ints, default=[5, 6, 7, 8, 9])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--original-checkpoint-pattern", required=True)
    p.add_argument("--morphspec-checkpoint-pattern", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/morphspec_r1_confirm_folds5_9")
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

    from protosleep.config import MICRO_BATCH_SIZE, USE_AMP
    from protosleep.data import balanced_class_weights_from_train, make_epoch_loader
    from protosleep.legacy_wavesleepnet_train import load_recordings_for_subjects
    from protosleep.mae_baseline import verify_mae_checkpoint_for_fold
    from protosleep.micro import evaluate_micro_loader
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.morphmae_transfer import MAE_PROBE, train_attnsleep_transfer
    from protosleep.morphspec_confirm import build_matched_probe_triplet
    from protosleep.utils import DEVICE

    print("=" * 100)
    print("MORPHSPEC-R1 HELD-OUT-FOLD CONFIRMATORY FROZEN PROBE")
    print("=" * 100)
    print(f"device={DEVICE} AMP={USE_AMP}")
    print(f"held-out folds={args.folds} supervised seeds={args.seeds}")
    print("Recipe is frozen from folds 0-4; no hyperparameter selection occurs here.")
    print("All MRCNNs stay frozen and in eval mode; only matched TCE/classifier heads train.")
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
        print("original MorphMAE:", original_ckpt)
        print("MorphSpec-R1:", morphspec_ckpt)
        print("=" * 100)

        for base_seed in args.seeds:
            actual_seed = int(base_seed + 10000 * fold)
            run_dir = output_root / f"fold_{fold:02d}" / f"seed_{base_seed}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            random_model, original_model, morphspec_model, init_meta = build_matched_probe_triplet(
                actual_seed, original_ckpt, morphspec_ckpt
            )
            models = {
                "random_probe": random_model,
                "original_mae_probe": original_model,
                "morphspec_probe": morphspec_model,
            }

            def run_one(name: str, model):
                checkpoint = ckpt_dir / f"{name}.pt"
                history_path = log_dir / f"{name}.csv"
                if args.reuse and checkpoint.is_file():
                    payload = torch.load(checkpoint, map_location="cpu")
                    model.load_state_dict(payload["state_dict"])
                    va = make_epoch_loader(val_recordings, MICRO_BATCH_SIZE, shuffle=False)
                    metrics = evaluate_micro_loader(model.to(DEVICE), va, DEVICE)
                    return {
                        "f1": float(metrics["macro_f1"]),
                        "best_epoch": int(payload["best_epoch"]),
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
                        MAE_PROBE,
                        actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["history"].to_csv(history_path, index=False)
                return {
                    "f1": float(result["best_val_macro_f1"]),
                    "best_epoch": int(result["best_epoch"]),
                    "seconds": float(time.perf_counter() - t0),
                    "reused": False,
                }

            results = {}
            for name in ("random_probe", "original_mae_probe", "morphspec_probe"):
                print(f"--- fold={fold} seed={base_seed} | {name} ---")
                results[name] = run_one(name, models[name])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            random_f1 = results["random_probe"]["f1"]
            original_f1 = results["original_mae_probe"]["f1"]
            morphspec_f1 = results["morphspec_probe"]["f1"]
            row = {
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
                "random_probe_f1": random_f1,
                "original_mae_probe_f1": original_f1,
                "morphspec_probe_f1": morphspec_f1,
                "original_minus_random": float(original_f1 - random_f1),
                "morphspec_minus_random": float(morphspec_f1 - random_f1),
                "morphspec_minus_original": float(morphspec_f1 - original_f1),
                "random_best_epoch": results["random_probe"]["best_epoch"],
                "original_best_epoch": results["original_mae_probe"]["best_epoch"],
                "morphspec_best_epoch": results["morphspec_probe"]["best_epoch"],
                "non_mrcnn_initialization_match": True,
                "amp": bool(USE_AMP),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "confirmatory_partial.csv", index=False)
            (run_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        **row,
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
                f"fold={fold} seed={base_seed} | random={random_f1:.4f} original={original_f1:.4f} "
                f"morphspec={morphspec_f1:.4f} | MorphSpec-random={morphspec_f1-random_f1:+.4f} "
                f"MorphSpec-original={morphspec_f1-original_f1:+.4f}"
            )

            del random_model, original_model, morphspec_model, models, results

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "confirmatory_raw.csv", index=False)

    cols = [
        "random_probe_f1",
        "original_mae_probe_f1",
        "morphspec_probe_f1",
        "original_minus_random",
        "morphspec_minus_random",
        "morphspec_minus_original",
    ]
    fold_means = raw.groupby("fold", as_index=False)[cols].mean()
    fold_means.to_csv(output_root / "confirmatory_fold_means.csv", index=False)

    effects = []
    for col in ("original_minus_random", "morphspec_minus_random", "morphspec_minus_original"):
        x = fold_means[col].to_numpy(float)
        pos = int(np.sum(x > 0))
        neg = int(np.sum(x < 0))
        effects.append(
            {
                "effect": col,
                "mean": float(np.mean(x)),
                "std_across_folds": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)),
                "positive_folds": pos,
                "negative_folds": neg,
                "n_folds": int(len(x)),
                "exact_sign_p_two_sided": _exact_sign_p(pos, neg),
            }
        )
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(output_root / "confirmatory_effects.csv", index=False)

    primary = effects_df[effects_df["effect"] == "morphspec_minus_random"].iloc[0]
    n_folds = int(primary["n_folds"])
    required_positive = max(1, math.ceil(0.60 * n_folds))
    passes = bool(float(primary["mean"]) > 0 and int(primary["positive_folds"]) >= required_positive)
    decision = {
        "confirmatory_held_out_folds": True,
        "recipe_selected_on_folds": [0, 1, 2, 3, 4],
        "evaluation_folds": [int(x) for x in args.folds],
        "mean_morphspec_minus_random": float(primary["mean"]),
        "positive_folds_vs_random": int(primary["positive_folds"]),
        "required_positive_folds": int(required_positive),
        "passes_frozen_confirmation_gate": passes,
        "gate": "mean MorphSpec-random > 0 and positive in at least 60% of held-out folds",
        "test_metrics_computed": False,
        "designated_test_subject_files_opened": False,
        "external_transfer_claim_allowed": False,
    }
    (output_root / "confirmatory_summary.json").write_text(json.dumps(decision, indent=2))

    print("\n" + "=" * 100)
    print("SEED-AVERAGED HELD-OUT FOLD RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("CONFIRMATORY EFFECT SUMMARY")
    print("=" * 100)
    print(effects_df.to_string(index=False))
    print("\n" + "=" * 100)
    print("FROZEN CONFIRMATION DECISION")
    print("=" * 100)
    print(json.dumps(decision, indent=2))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("External-transfer conclusion allowed: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

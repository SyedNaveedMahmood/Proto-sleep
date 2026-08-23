#!/usr/bin/env python3
"""Final parameter-efficient downstream confirmation of MorphSpec-R1.

The preceding staged-unfreeze experiment on folds 10-14 showed that both original
MorphMAE-v2 and MorphSpec-R1 lose their advantage when the MRCNN is unfrozen under the
frozen staged recipe. This runner tests the next mechanistic hypothesis on the last
untouched rotating validation folds (default 15-19): the confirmed MorphSpec
representation should be *retained* as a frozen feature extractor while only the
AttnSleep TCE/classifier are trained.

Within each fold/seed four models start from matched non-MRCNN initialization:

  A1_standard       random MRCNN, fully trainable standard AttnSleep
  random_frozen     random MRCNN frozen, train TCE/classifier only
  original_frozen   MorphMAE-v2 MRCNN frozen, train TCE/classifier only
  morphspec_frozen  MorphSpec-R1 MRCNN frozen, train TCE/classifier only

No transfer hyperparameter selection occurs here. Only train + validation NPZ files are
opened; the designated test subject is never evaluated.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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


def _passes_gate(mean_delta: float, positive_folds: int, n_folds: int) -> bool:
    required = max(1, math.ceil(0.60 * int(n_folds)))
    return bool(float(mean_delta) > 0 and int(positive_folds) >= required)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--folds", type=_csv_ints, default=[15, 16, 17, 18, 19])
    p.add_argument("--seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--original-checkpoint-pattern", required=True)
    p.add_argument("--morphspec-checkpoint-pattern", required=True)
    p.add_argument("--output-dir", default="mist_sleep_runs/morphspec_r1_frozen_transfer_folds15_19")
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
    from protosleep.morphmae_transfer import MAE_PROBE, train_attnsleep_transfer
    from protosleep.morphspec_confirm import build_matched_probe_triplet
    from protosleep.utils import DEVICE

    print("=" * 100)
    print("MORPHSPEC-R1 FINAL FROZEN-ENCODER DOWNSTREAM CONFIRMATION")
    print("=" * 100)
    print(f"device={DEVICE} AMP={USE_AMP}")
    print(f"untouched downstream folds={args.folds} supervised seeds={args.seeds}")
    print("A1_standard: fully trainable random-initialized AttnSleep")
    print("random/original/MorphSpec frozen: MRCNN kept in eval mode; only TCE/classifier train")
    print("No transfer hyperparameter selection occurs in this run.")
    print("Primary endpoint: MorphSpec-frozen minus A1-standard validation Macro-F1.")
    print("Secondary endpoint: MorphSpec-frozen minus original-MorphMAE-frozen validation Macro-F1.")
    print("Representation control: MorphSpec-frozen minus random-frozen validation Macro-F1.")
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

            random_template, original_model, morphspec_model, init_meta = build_matched_probe_triplet(
                actual_seed, original_ckpt, morphspec_ckpt
            )
            a1_model = copy.deepcopy(random_template)
            random_frozen_model = random_template

            total_params = int(sum(p.numel() for p in a1_model.parameters()))
            mrcnn_params = int(sum(p.numel() for p in a1_model.mrcnn.parameters()))
            frozen_trainable_params = int(total_params - mrcnn_params)

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
                return trained.cpu(), metrics, int(result["best_epoch"]), float(time.perf_counter() - t0), False

            def run_frozen(name: str, model):
                checkpoint = ckpt_dir / f"{name}.pt"
                if args.reuse and checkpoint.is_file():
                    loaded, metrics, best_epoch = eval_loaded(model, checkpoint)
                    return loaded, metrics, best_epoch, 0.0, True
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
                        MAE_PROBE,
                        actual_seed,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["history"].to_csv(log_dir / f"{name}.csv", index=False)
                trained = result["model"].to(DEVICE).eval()
                metrics = evaluate_micro_loader(trained, va, DEVICE)
                return trained.cpu(), metrics, int(result["best_epoch"]), float(time.perf_counter() - t0), False

            print(f"--- fold={fold} seed={base_seed} | A1_standard ---")
            a1_model, m1, e1, t1, r1 = run_a1(a1_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"--- fold={fold} seed={base_seed} | random_frozen ---")
            random_frozen_model, mr, er, tr_sec, rr = run_frozen("random_frozen", random_frozen_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"--- fold={fold} seed={base_seed} | original_frozen ---")
            original_model, mo, eo, to_sec, ro = run_frozen("original_frozen", original_model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"--- fold={fold} seed={base_seed} | morphspec_frozen ---")
            morphspec_model, mm, em, tm_sec, rm = run_frozen("morphspec_frozen", morphspec_model)
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
                "random_frozen_f1": float(mr["macro_f1"]),
                "original_frozen_f1": float(mo["macro_f1"]),
                "morphspec_frozen_f1": float(mm["macro_f1"]),
                "random_frozen_minus_A1": float(mr["macro_f1"] - m1["macro_f1"]),
                "original_frozen_minus_A1": float(mo["macro_f1"] - m1["macro_f1"]),
                "morphspec_frozen_minus_A1": float(mm["macro_f1"] - m1["macro_f1"]),
                "morphspec_minus_random_frozen": float(mm["macro_f1"] - mr["macro_f1"]),
                "morphspec_minus_original_frozen": float(mm["macro_f1"] - mo["macro_f1"]),
                "A1_best_epoch": int(e1),
                "random_frozen_best_epoch": int(er),
                "original_frozen_best_epoch": int(eo),
                "morphspec_frozen_best_epoch": int(em),
                "A1_seconds": float(t1),
                "random_frozen_seconds": float(tr_sec),
                "original_frozen_seconds": float(to_sec),
                "morphspec_frozen_seconds": float(tm_sec),
                "A1_reused": bool(r1),
                "random_frozen_reused": bool(rr),
                "original_frozen_reused": bool(ro),
                "morphspec_frozen_reused": bool(rm),
                "total_model_params": total_params,
                "mrcnn_params_frozen": mrcnn_params,
                "frozen_transfer_trainable_params": frozen_trainable_params,
                "non_mrcnn_initialization_match": True,
                "amp": bool(USE_AMP),
            }
            for idx, stage in enumerate(STAGE_NAMES):
                row[f"A1_{stage}_f1"] = float(m1["per_class_f1"][idx])
                row[f"random_frozen_{stage}_f1"] = float(mr["per_class_f1"][idx])
                row[f"original_frozen_{stage}_f1"] = float(mo["per_class_f1"][idx])
                row[f"morphspec_frozen_{stage}_f1"] = float(mm["per_class_f1"][idx])

            rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "frozen_transfer_partial.csv", index=False)
            (run_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        **row,
                        "frozen_recipe": MAE_PROBE.__dict__,
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
                f"randomFrozen={row['random_frozen_f1']:.4f} originalFrozen={row['original_frozen_f1']:.4f} "
                f"morphspecFrozen={row['morphspec_frozen_f1']:.4f} | "
                f"MorphSpec-A1={row['morphspec_frozen_minus_A1']:+.4f} "
                f"MorphSpec-randomFrozen={row['morphspec_minus_random_frozen']:+.4f} "
                f"MorphSpec-originalFrozen={row['morphspec_minus_original_frozen']:+.4f}"
            )

            del a1_model, random_frozen_model, original_model, morphspec_model

    raw = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    raw.to_csv(output_root / "frozen_transfer_raw.csv", index=False)

    metric_cols = [
        "A1_standard_f1",
        "random_frozen_f1",
        "original_frozen_f1",
        "morphspec_frozen_f1",
        "random_frozen_minus_A1",
        "original_frozen_minus_A1",
        "morphspec_frozen_minus_A1",
        "morphspec_minus_random_frozen",
        "morphspec_minus_original_frozen",
    ]
    for stage in STAGE_NAMES:
        metric_cols.extend(
            [
                f"A1_{stage}_f1",
                f"random_frozen_{stage}_f1",
                f"original_frozen_{stage}_f1",
                f"morphspec_frozen_{stage}_f1",
            ]
        )
    fold_means = raw.groupby("fold", as_index=False)[metric_cols].mean()
    fold_means.to_csv(output_root / "frozen_transfer_fold_means.csv", index=False)

    effects = []
    effect_cols = (
        "random_frozen_minus_A1",
        "original_frozen_minus_A1",
        "morphspec_frozen_minus_A1",
        "morphspec_minus_random_frozen",
        "morphspec_minus_original_frozen",
    )
    for col in effect_cols:
        x = fold_means[col].to_numpy(float)
        pos = int(np.sum(x > 0))
        neg = int(np.sum(x < 0))
        lo, hi = _bootstrap_mean_ci(x)
        effects.append(
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
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(output_root / "frozen_transfer_effects.csv", index=False)

    stage_rows = []
    for stage in STAGE_NAMES:
        for label, lhs, rhs in (
            ("morphspec_frozen_minus_A1", f"morphspec_frozen_{stage}_f1", f"A1_{stage}_f1"),
            (
                "morphspec_minus_random_frozen",
                f"morphspec_frozen_{stage}_f1",
                f"random_frozen_{stage}_f1",
            ),
            (
                "morphspec_minus_original_frozen",
                f"morphspec_frozen_{stage}_f1",
                f"original_frozen_{stage}_f1",
            ),
        ):
            x = (fold_means[lhs] - fold_means[rhs]).to_numpy(float)
            stage_rows.append(
                {
                    "stage": stage,
                    "effect": label,
                    "mean_delta_f1": float(np.mean(x)),
                    "median_delta_f1": float(np.median(x)),
                    "positive_folds": int(np.sum(x > 0)),
                    "n_folds": int(len(x)),
                }
            )
    stage_effects = pd.DataFrame(stage_rows)
    stage_effects.to_csv(output_root / "frozen_transfer_stage_effects.csv", index=False)

    def effect_row(name: str):
        return effects_df[effects_df["effect"] == name].iloc[0]

    primary = effect_row("morphspec_frozen_minus_A1")
    objective = effect_row("morphspec_minus_original_frozen")
    representation = effect_row("morphspec_minus_random_frozen")
    required_positive = max(1, math.ceil(0.60 * len(fold_means)))

    decision = {
        "final_untouched_rotating_folds": True,
        "representation_design_folds": [0, 1, 2, 3, 4],
        "representation_confirmation_folds": [5, 6, 7, 8, 9],
        "staged_downstream_folds": [10, 11, 12, 13, 14],
        "frozen_transfer_evaluation_folds": [int(x) for x in args.folds],
        "frozen_transfer_recipe": MAE_PROBE.__dict__,
        "primary_endpoint": "morphspec_frozen_minus_A1_standard_validation_macro_f1",
        "secondary_endpoint": "morphspec_frozen_minus_original_frozen_validation_macro_f1",
        "representation_control": "morphspec_frozen_minus_random_frozen_validation_macro_f1",
        "required_positive_folds": int(required_positive),
        "mean_morphspec_frozen_minus_A1": float(primary["mean"]),
        "positive_folds_vs_A1": int(primary["positive_folds"]),
        "passes_primary_frozen_transfer_gate": _passes_gate(
            float(primary["mean"]), int(primary["positive_folds"]), int(primary["n_folds"])
        ),
        "mean_morphspec_minus_original_frozen": float(objective["mean"]),
        "positive_folds_vs_original_frozen": int(objective["positive_folds"]),
        "passes_objective_frozen_transfer_gate": _passes_gate(
            float(objective["mean"]), int(objective["positive_folds"]), int(objective["n_folds"])
        ),
        "mean_morphspec_minus_random_frozen": float(representation["mean"]),
        "positive_folds_vs_random_frozen": int(representation["positive_folds"]),
        "passes_representation_control_gate": _passes_gate(
            float(representation["mean"]), int(representation["positive_folds"]), int(representation["n_folds"])
        ),
        "gate": "mean delta > 0 and positive in at least 60% of untouched folds",
        "test_metrics_computed": False,
        "designated_test_subject_files_opened": False,
        "external_transfer_claim_allowed": False,
    }
    decision["passes_full_frozen_transfer_gate"] = bool(
        decision["passes_primary_frozen_transfer_gate"]
        and decision["passes_objective_frozen_transfer_gate"]
        and decision["passes_representation_control_gate"]
    )
    (output_root / "frozen_transfer_summary.json").write_text(json.dumps(decision, indent=2))

    print("\n" + "=" * 100)
    print("SEED-AVERAGED UNTOUCHED-FOLD RESULTS")
    print("=" * 100)
    print(fold_means.to_string(index=False))
    print("\n" + "=" * 100)
    print("FROZEN-TRANSFER EFFECT SUMMARY")
    print("=" * 100)
    print(effects_df.to_string(index=False))
    print("\n" + "=" * 100)
    print("PER-STAGE SUPPORTING EFFECTS")
    print("=" * 100)
    print(stage_effects.to_string(index=False))
    print("\n" + "=" * 100)
    print("FROZEN TRANSFER DECISION")
    print("=" * 100)
    print(json.dumps(decision, indent=2))
    print("TEST METRICS COMPUTED: NO")
    print("DESIGNATED TEST SUBJECT FILES OPENED: NO")
    print("External-transfer conclusion allowed: NO")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

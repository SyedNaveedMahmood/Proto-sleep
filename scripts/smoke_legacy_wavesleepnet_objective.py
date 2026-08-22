#!/usr/bin/env python3
"""One-step gradient gate for the recovered historical WaveSleepNet prototype objective.

This script performs two real optimizer steps (one matched A3, one matched A4) on the same
TRAIN-ONLY mini-batch. It does not run validation, test evaluation, early stopping, or a full
training experiment. The loss itself is executed from the frozen archived train_mtcl.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mae-checkpoint", required=True)
    p.add_argument("--report", default="legacy_wavesleepnet_objective_smoke.json")
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from protosleep.data import EpochDataset, load_sleep_edf_recordings, split_subjects
    from protosleep.legacy_wavesleepnet import (
        build_matched_legacy_a3_a4,
        validate_legacy_snapshot,
    )
    from protosleep.legacy_wavesleepnet_objective import (
        historical_training_config,
        import_legacy_trainer_module,
        one_historical_optimizer_step,
    )
    from protosleep.mist import extract_mrcnn_state_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.legacy_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    mae_path = Path(args.mae_checkpoint).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    print("=" * 100)
    print("HISTORICAL WAVESLEEPNET OBJECTIVE / BACKWARD SMOKE")
    print("=" * 100)
    print("TWO ONE-STEP TRAINING UPDATES ONLY. NO VALIDATION OR TEST EVALUATION.")
    print("device:", device)

    snapshot = validate_legacy_snapshot(root)
    cfg = historical_training_config(root)
    trainer_module = import_legacy_trainer_module(root)

    recordings = load_sleep_edf_recordings(str(data_dir))
    split = split_subjects(recordings, int(args.fold))

    _, mae_meta = extract_mrcnn_state_dict(mae_path)
    declared = None
    for key in ("train_subjects", "pretrain_subjects", "subjects"):
        if key in mae_meta:
            declared = sorted(int(x) for x in mae_meta[key])
            break
    expected_train = sorted(int(x) for x in split["train_subjects"])
    if declared is None:
        raise RuntimeError("MorphMAE checkpoint has no declared train/pretrain subjects")
    if declared != expected_train:
        raise RuntimeError(
            f"MorphMAE checkpoint subjects {declared} do not equal fold-{args.fold} train subjects {expected_train}"
        )
    if "fold" in mae_meta and int(mae_meta["fold"]) != int(args.fold):
        raise RuntimeError(
            f"MorphMAE checkpoint fold={mae_meta['fold']} does not match requested fold={args.fold}"
        )

    gen = torch.Generator()
    gen.manual_seed(int(args.seed) + 1701)
    train_loader = DataLoader(
        EpochDataset(split["train"]),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    x, y = next(iter(train_loader))

    train_subjects = set(split["train_subjects"])
    val_subjects = set(split["val_subjects"])
    test_subjects = set(split["test_subjects"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)

    a3, a4, pair_meta = build_matched_legacy_a3_a4(root, mae_path, int(args.seed))

    # Match stochastic layers (if any) across the two one-step updates.
    def reseed():
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    reseed()
    a3_step = one_historical_optimizer_step(a3, x, y, trainer_module, cfg, device)
    reseed()
    a4_step = one_historical_optimizer_step(a4, x, y, trainer_module, cfg, device)

    print("\nLeakage gate:")
    print("  fold:", args.fold)
    print("  train subjects:", expected_train)
    print("  validation subjects NOT LOADED:", split["val_subjects"])
    print("  designated test subjects NOT LOADED:", split["test_subjects"])
    print("  MorphMAE train-subject metadata: MATCH")

    print("\nExact historical objective source:")
    print("  trainer sha256:", snapshot["trainer"]["sha256"])
    print("  method: OneFoldTrainer.protop_loss")
    print("  optimizer: Adam")
    print("  lr:", cfg["training_params"]["lr"])
    print("  weight_decay:", cfg["training_params"]["weight_decay"])
    print("  precision: fp32")

    def show(name, result):
        print(f"\n{name} one-step result:")
        print(f"  loss: {result['loss']:.6f}")
        print(f"  terms: {json.dumps(result['terms'], sort_keys=True)}")
        print(f"  grad_norm: {result['grad_norm']:.6f}")
        print(f"  tensors with grad: {result['n_parameter_tensors_with_grad']}")
        print(f"  tensors changed: {result['n_parameter_tensors_changed']}")
        print("  logits:", result["logit_shape"])
        print("  FINITE FORWARD/BACKWARD/STEP: PASS")

    show("A3_legacy", a3_step)
    show("A4_legacy", a4_step)

    report = {
        "fold": int(args.fold),
        "seed": int(args.seed),
        "device": str(device),
        "train_subjects": expected_train,
        "val_subjects_not_loaded": split["val_subjects"],
        "test_subjects_not_loaded": split["test_subjects"],
        "mae_checkpoint": str(mae_path),
        "mae_metadata": mae_meta,
        "pair_metadata": pair_meta,
        "legacy_snapshot": snapshot,
        "historical_training_params": cfg["training_params"],
        "A3_legacy": a3_step,
        "A4_legacy": a4_step,
        "validation_metrics_computed": False,
        "test_metrics_computed": False,
        "full_training_started": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print("HISTORICAL OBJECTIVE / BACKWARD COMPATIBILITY: PASS")
    print("=" * 100)
    print("report:", report_path)
    print("FULL TRAINING STARTED: NO")
    print("TEST METRICS COMPUTED: NO")
    print("Next gate: matched A3_legacy/A4_legacy validation-only stability training.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict no-training compatibility gate for the historical WaveSleepNet MIST A3/A4 recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from protosleep.legacy_wavesleepnet import (
    build_matched_legacy_a3_a4,
    sample_train_only_fold_batch,
    tensor_leaves,
    tensor_tree_shapes,
    validate_legacy_snapshot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate the frozen historical WaveSleepNet snapshot, apply only the audited 27->30 "
            "MorphMAE bridge patch in memory, build a matched A3/A4 pair, strictly load MorphMAE "
            "into A4.mrcnn, and run one train-only forward pass. No training is performed."
        )
    )
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mae-checkpoint", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    p.add_argument("--report", default="legacy_wavesleepnet_smoke.json")
    p.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="Development-only: allow audited legacy file hashes to differ. Do not use for confirmatory runs.",
    )
    return p.parse_args()


def _assert_class_logits_exist(output) -> None:
    leaves = list(tensor_leaves(output))
    if not leaves:
        raise RuntimeError("Legacy ProtoPNet forward produced no tensor outputs")
    if not any(t.ndim >= 2 and int(t.shape[-1]) == 5 for t in leaves):
        raise RuntimeError(
            "Legacy ProtoPNet forward produced tensors, but none has final class dimension 5: "
            + json.dumps(tensor_tree_shapes(output))
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 100)
    print("HISTORICAL WAVESLEEPNET MIST COMPATIBILITY SMOKE")
    print("=" * 100)
    print("NO TRAINING WILL BE PERFORMED.")
    print("device:", device)

    snapshot = validate_legacy_snapshot(args.legacy_root, allow_source_drift=args.allow_source_drift)
    print("\nFrozen legacy source snapshot:")
    for name, row in snapshot.items():
        status = "MATCH" if row["sha256"] == row["expected_sha256"] else "DRIFT-ALLOWED"
        print(f"  {name:12s} {status} {row['sha256'][:12]}  {row['path']}")

    a3, a4, pair_meta = build_matched_legacy_a3_a4(
        legacy_root=args.legacy_root,
        mae_checkpoint=args.mae_checkpoint,
        seed=args.seed,
    )
    print("\nMatched historical pair:")
    print("  ProtoPNet constructor:", pair_meta["constructor_signature"])
    print("  prototype_vectors:", pair_meta["prototype_shape"])
    print("  A3 random MRCNN sha256:", pair_meta["a3_random_mrcnn_sha256"])
    print("  A4 MorphMAE MRCNN sha256:", pair_meta["a4_mae_mrcnn_sha256"])
    print("  STRICT MRCNN LOAD: PASS")
    print("  NON-MRCNN INITIALIZATION MATCH: PASS")

    x, y, split_meta = sample_train_only_fold_batch(
        data_dir=args.data_dir,
        fold=args.fold,
        batch_size=args.batch_size,
    )
    print("\nSmoke batch:")
    print("  fold:", split_meta["fold"])
    print("  train subjects:", split_meta["train_subjects"])
    print("  validation subjects NOT loaded:", split_meta["val_subjects"])
    print("  designated test subjects NOT loaded:", split_meta["test_subjects"])
    print("  x:", list(x.shape), x.dtype)
    print("  y:", list(y.shape), y.dtype, "labels=", y.tolist())

    a3 = a3.to(device).eval()
    a4 = a4.to(device).eval()
    x = x.to(device)

    with torch.no_grad():
        h3 = a3.mrcnn(x)
        h4 = a4.mrcnn(x)
        if list(h3.shape) != [args.batch_size, 30, 80]:
            raise RuntimeError(f"Legacy A3 MRCNN output {list(h3.shape)}, expected [{args.batch_size}, 30, 80]")
        if list(h4.shape) != [args.batch_size, 30, 80]:
            raise RuntimeError(f"Legacy A4 MRCNN output {list(h4.shape)}, expected [{args.batch_size}, 30, 80]")
        out3 = a3(x)
        out4 = a4(x)

    _assert_class_logits_exist(out3)
    _assert_class_logits_exist(out4)

    print("\nForward compatibility:")
    print("  A3 MRCNN:", list(h3.shape))
    print("  A4 MRCNN:", list(h4.shape))
    print("  A3 output:", json.dumps(tensor_tree_shapes(out3), sort_keys=True))
    print("  A4 output:", json.dumps(tensor_tree_shapes(out4), sort_keys=True))
    print("  FORWARD PASS: PASS")

    report = {
        "status": "pass",
        "training_started": False,
        "device": str(device),
        "legacy_snapshot": snapshot,
        "fold_split": split_meta,
        "pair": pair_meta,
        "batch_shape": list(x.shape),
        "labels": y.tolist(),
        "a3_mrcnn_shape": list(h3.shape),
        "a4_mrcnn_shape": list(h4.shape),
        "a3_output": tensor_tree_shapes(out3),
        "a4_output": tensor_tree_shapes(out4),
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print("HISTORICAL WAVESLEEPNET COMPATIBILITY: PASS")
    print("=" * 100)
    print("report:", report_path)
    print("NO TRAINING STARTED: YES")
    print("Next gate: implement/run the matched historical objective only after this smoke passes.")


if __name__ == "__main__":
    main()

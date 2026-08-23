#!/usr/bin/env python3
"""Exploratory label-free MorphMAE representation refinement.

This is the next gate after the frozen MorphMAE representation probe failed. It does not
change downstream hyperparameters. Instead it starts from each leakage-safe MorphMAE-v2
encoder and performs a second self-supervised refinement stage on the same fold training
subjects only.

The objective predicts stage-relevant but label-free morphology targets computed from the
clean EEG epoch while the encoder receives a patch-masked view. Targets comprise full-epoch
relative band power, within-epoch band variability/maxima across 5-second segments, and
normalized line length. Stage labels are neither read nor used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List


def _csv_ints(value: str) -> List[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--folds", type=_csv_ints, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--source-checkpoint-pattern",
        required=True,
        help="Fold-aware MorphMAE-v2 checkpoint path, e.g. '.../fold_{fold:02d}/ssl_seed_1337/encoder.pt'.",
    )
    p.add_argument("--output-dir", default="mist_sleep_runs/morphmae_morphspec_r1")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--reuse", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep package-level paths explicit before importing project modules.
    os.environ["SLEEP_EDF_NPZ_DIR"] = str(data_dir)
    os.environ["PROTOMAE_OUT"] = str(output_root)

    from protosleep.mae_baseline import verify_mae_checkpoint_for_fold
    from protosleep.morphmae_bridge import fold_subjects_from_npz
    from protosleep.morphmae_representation import (
        MORPHSPEC_R1,
        load_unlabeled_recordings,
        refine_morphmae_encoder,
    )
    from protosleep.mist import extract_mrcnn_state_dict
    from protosleep.utils import DEVICE

    print("=" * 100)
    print("MORPHMAE REPRESENTATION REFINEMENT: MORPHSPEC-R1")
    print("=" * 100)
    print(f"device: {DEVICE}")
    print(f"folds: {args.folds}")
    print("source: leakage-safe MorphMAE-v2")
    print("objective: predict label-free spectral/morphology targets from a patch-masked epoch")
    print("stage labels read: NO")
    print("stage labels used: NO")
    print("validation/test subject NPZ files opened: NO")
    print("This is exploratory development. It is not a confirmatory result.")
    print("=" * 100)

    manifest = []
    for fold in args.folds:
        split = fold_subjects_from_npz(data_dir, fold)
        source = Path(args.source_checkpoint_pattern.format(fold=fold)).expanduser().resolve()
        verify_mae_checkpoint_for_fold(source, fold, split["train_subjects"])

        run_dir = output_root / f"fold_{fold:02d}" / f"refine_seed_{args.seed}"
        out_ckpt = run_dir / "encoder.pt"
        manifest_path = run_dir / "manifest.json"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.reuse and out_ckpt.exists():
            _, meta = extract_mrcnn_state_dict(out_ckpt)
            declared = sorted(int(v) for v in meta.get("train_subjects", []))
            if declared != sorted(split["train_subjects"]):
                raise RuntimeError(
                    f"Existing refined checkpoint split mismatch for fold {fold}: {declared}"
                )
            print(f"fold={fold}: REUSE {out_ckpt}")
            if manifest_path.exists():
                manifest.append(json.loads(manifest_path.read_text()))
            else:
                manifest.append({"fold": fold, "checkpoint": str(out_ckpt), "reused": True})
            continue

        recordings, io_meta = load_unlabeled_recordings(data_dir, split["train_subjects"])
        if set(io_meta["subjects"]) & (set(split["val_subjects"]) | set(split["test_subjects"])):
            raise RuntimeError("Leakage barrier failure in unlabeled refinement loader")

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("train subjects:", split["train_subjects"])
        print("validation subjects NOT OPENED:", split["val_subjects"])
        print("designated test subjects NOT OPENED:", split["test_subjects"])
        print(f"opened train recordings={io_meta['n_recordings']} epochs={io_meta['n_epochs']:,}")
        print("stage label arrays read:", io_meta["stage_label_arrays_read"])
        print("source checkpoint:", source)
        print("=" * 100)

        result = refine_morphmae_encoder(
            source_checkpoint=source,
            train_recordings=recordings,
            output_checkpoint=out_ckpt,
            fold=fold,
            train_subjects=split["train_subjects"],
            val_subjects=split["val_subjects"],
            test_subjects=split["test_subjects"],
            recipe=MORPHSPEC_R1,
            seed=args.seed,
        )
        result["train_io"] = io_meta
        result["reused"] = False
        manifest_path.write_text(json.dumps(result, indent=2, default=str))
        manifest.append(result)
        print(
            f"fold={fold}: COMPLETE final_loss={result['final_loss']:.6f} checkpoint={out_ckpt}"
        )

        del recordings

    summary = {
        "development_only": True,
        "recipe": MORPHSPEC_R1.__dict__,
        "folds": args.folds,
        "runs": manifest,
        "stage_labels_used": False,
        "stage_label_arrays_read": False,
        "validation_subject_files_opened": False,
        "test_subject_files_opened": False,
        "next_gate": "Frozen representation probe against the previously measured random-probe reference.",
    }
    (output_root / "representation_refine_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\n" + "=" * 100)
    print("MORPHSPEC-R1 REFINEMENT COMPLETE")
    print("=" * 100)
    print("stage labels used: NO")
    print("stage label arrays read: NO")
    print("validation subject files opened: NO")
    print("designated test subject files opened: NO")
    print("Next gate: frozen representation probe only; do not tune downstream transfer yet.")
    print("Results saved under:", output_root)


if __name__ == "__main__":
    main()

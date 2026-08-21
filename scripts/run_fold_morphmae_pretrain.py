#!/usr/bin/env python3
"""
Leakage-safe fold-specific MorphMAE-v2 pretraining using the verified historical codebase.

The historical implementation is executed unchanged. This wrapper changes only:
  * the NPZ root -> a symlink view containing current-fold TRAIN subjects only;
  * output directory;
  * seed.

Validation/test NPZ files are absent from the legacy process's configured data root.
After training, a canonical AttnSleep-MRCNN checkpoint with explicit split provenance is exported
for scripts/run_mist_stability.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from protosleep.mist import extract_mrcnn_state_dict
from protosleep.morphmae_bridge import (
    create_train_only_npz_view,
    fold_subjects_from_npz,
    load_prepare_v2_config,
    render_fold_launcher,
    sha256_file,
    sha256_legacy_source_tree,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", required=True, help="Historical MorphMAE_Sleep_Codebase directory.")
    p.add_argument("--data-dir", required=True, help="Current Sleep-EDF-20 NPZ directory.")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=1337, help="MorphMAE SSL seed; historical v2 default is 1337.")
    p.add_argument(
        "--base-config",
        default=None,
        help="Historical v2 YAML. Default: <legacy-root>/configs/mae_npz_edf78_v2.yaml",
    )
    p.add_argument("--output-dir", default="mist_sleep_runs/morphmae_pretrain")
    p.add_argument("--prepare-only", action="store_true", help="Build/verify the train-only view and generated config, but do not train.")
    p.add_argument("--reuse-existing", action="store_true", help="If best_morphmae.pt already exists, skip legacy training and only verify/export it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    base_config = (
        Path(args.base_config).expanduser().resolve()
        if args.base_config
        else legacy_root / "configs" / "mae_npz_edf78_v2.yaml"
    )
    if not legacy_root.is_dir():
        raise FileNotFoundError(legacy_root)
    if not base_config.is_file():
        raise FileNotFoundError(base_config)

    split = fold_subjects_from_npz(data_dir, args.fold)
    run_dir = Path(args.output_dir).expanduser().resolve() / f"fold_{args.fold:02d}" / f"ssl_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_view = run_dir / "train_npz_view"
    legacy_output = run_dir / "legacy_output"
    generated_config = run_dir / "morphmae_v2_fold_config.yaml"
    copied_launcher = run_dir / "legacy_pretrain_launcher.sh"
    manifest_path = run_dir / "pretrain_manifest.json"
    canonical_path = run_dir / "encoder.pt"

    linked = create_train_only_npz_view(data_dir, train_view, split["train_subjects"])
    cfg = load_prepare_v2_config(base_config, train_view, legacy_output, args.seed)
    write_yaml(cfg, generated_config)
    source_launcher, copied_launcher = render_fold_launcher(legacy_root, generated_config, copied_launcher)

    base_config_sha = sha256_file(base_config)
    generated_config_sha = sha256_file(generated_config)
    legacy_tree_sha = sha256_legacy_source_tree(legacy_root)

    manifest = {
        "status": "prepared",
        "fold": int(args.fold),
        "seed": int(args.seed),
        "data_dir": str(data_dir),
        "train_subjects": split["train_subjects"],
        "val_subjects_excluded_from_ssl": split["val_subjects"],
        "test_subjects_excluded_from_ssl": split["test_subjects"],
        "n_train_npz_files": len(linked),
        "train_npz_view": str(train_view),
        "legacy_root": str(legacy_root),
        "legacy_source_sha256": legacy_tree_sha,
        "historical_base_config": str(base_config),
        "historical_base_config_sha256": base_config_sha,
        "generated_config": str(generated_config),
        "generated_config_sha256": generated_config_sha,
        "source_launcher": str(source_launcher),
        "copied_launcher": str(copied_launcher),
        "legacy_output": str(legacy_output),
        "canonical_checkpoint": str(canonical_path),
        "test_signal_arrays_opened_by_wrapper": False,
        "validation_signal_arrays_opened_by_wrapper": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 100)
    print("FOLD-SPECIFIC MORPHMAE-v2 PRETRAIN")
    print("=" * 100)
    print("fold:", args.fold)
    print("train subjects:", split["train_subjects"])
    print("validation subject excluded from SSL:", split["val_subjects"])
    print("designated test subject excluded from SSL:", split["test_subjects"])
    print("train-only NPZ files:", len(linked))
    print("base config:", base_config)
    print("generated config:", generated_config)
    print("historical launcher:", source_launcher)
    print("legacy source sha256:", legacy_tree_sha)
    print("IMPORTANT: the legacy process receives only the train_npz_view directory.")

    if args.prepare_only:
        print("PREPARE-ONLY: no training started.")
        return

    best_ckpt = legacy_output / "best_morphmae.pt"
    if best_ckpt.exists() and args.reuse_existing:
        print("[checkpoint-hit]", best_ckpt)
    else:
        if best_ckpt.exists() and not args.reuse_existing:
            raise FileExistsError(
                f"{best_ckpt} already exists. Use a new output directory or --reuse-existing; "
                "the wrapper will not silently overwrite an SSL run."
            )
        legacy_output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(legacy_root) + (os.pathsep + old_pp if old_pp else "")
        print("\nStarting historical MorphMAE-v2 trainer...")
        subprocess.run(
            ["bash", str(copied_launcher)],
            cwd=str(legacy_root),
            env=env,
            check=True,
        )

    if not best_ckpt.is_file():
        raise FileNotFoundError(
            f"Historical trainer finished but {best_ckpt} was not produced. "
            "Inspect the legacy launcher/training log before proceeding."
        )

    mrcnn_state, legacy_meta = extract_mrcnn_state_dict(best_ckpt)
    source_ckpt_sha = sha256_file(best_ckpt)
    payload = {
        "state_dict": mrcnn_state,
        "train_subjects": split["train_subjects"],
        "val_subjects": split["val_subjects"],
        "test_subjects": split["test_subjects"],
        "fold": int(args.fold),
        "seed": int(args.seed),
        "recipe": "historical_morphmae_v2_train_only_view",
        "source_checkpoint": str(best_ckpt),
        "source_checkpoint_sha256": source_ckpt_sha,
        "historical_base_config": str(base_config),
        "historical_base_config_sha256": base_config_sha,
        "generated_config_sha256": generated_config_sha,
        "legacy_source_sha256": legacy_tree_sha,
    }
    torch.save(payload, canonical_path)

    # Verify our own exported artifact through the same strict bridge used by A4.
    _, canonical_meta = extract_mrcnn_state_dict(canonical_path)
    if sorted(canonical_meta.get("train_subjects", [])) != sorted(split["train_subjects"]):
        raise RuntimeError("Canonical checkpoint lost/changed train_subject provenance")

    manifest.update(
        {
            "status": "complete",
            "legacy_best_checkpoint": str(best_ckpt),
            "legacy_best_checkpoint_sha256": source_ckpt_sha,
            "legacy_checkpoint_mapping": legacy_meta,
            "canonical_checkpoint_sha256": sha256_file(canonical_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print("MORPHMAE PRETRAIN COMPLETE")
    print("=" * 100)
    print("legacy checkpoint:", best_ckpt)
    print("canonical A4 checkpoint:", canonical_path)
    print("manifest:", manifest_path)
    print("STRICT MRCNN COMPATIBILITY: PASS")
    print("VALIDATION SUBJECT USED BY SSL: NO")
    print("DESIGNATED TEST SUBJECT USED BY SSL: NO")


if __name__ == "__main__":
    main()

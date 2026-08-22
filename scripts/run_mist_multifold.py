#!/usr/bin/env python3
"""Sequential leakage-safe MorphMAE pretraining + matched multifold MIST stability audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from protosleep.mist_multifold import (
    canonical_morphmae_checkpoint,
    morphmae_checkpoint_pattern,
    parse_int_csv,
    validate_fold_morphmae_checkpoint,
    validate_multifold_checkpoints,
)


def _csv_ints(value: str) -> List[int]:
    try:
        return parse_int_csv(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create/verify fold-specific MorphMAE-v2 checkpoints sequentially, then run the "
            "validation-only A1/A3_current/A4_current stability audit."
        )
    )
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--folds", type=_csv_ints, default=[0, 1, 2, 3, 4])
    p.add_argument("--ssl-seed", type=int, default=1337)
    p.add_argument("--supervised-seeds", type=_csv_ints, default=[123, 456, 789])
    p.add_argument("--pretrain-output-dir", default="mist_sleep_runs/morphmae_pretrain")
    p.add_argument("--stability-output-dir", default="mist_sleep_runs/multifold_stability")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/verify missing fold configs and train-only views, but start no training.",
    )
    p.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Create/verify fold-specific MorphMAE checkpoints, then stop before supervised stability training.",
    )
    p.add_argument(
        "--stability-only",
        action="store_true",
        help="Require all fold-specific checkpoints to exist and run only the supervised stability audit.",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable AMP for A1/A3/A4 supervised training only.")
    p.add_argument(
        "--reuse-stability",
        action="store_true",
        help="Reuse completed A1/A3/A4 checkpoints in the stability output directory.",
    )
    args = p.parse_args()
    modes = int(args.prepare_only) + int(args.pretrain_only) + int(args.stability_only)
    if modes > 1:
        p.error("--prepare-only, --pretrain-only, and --stability-only are mutually exclusive")
    return args


def _run(cmd: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    pretrain_root = Path(args.pretrain_output_dir).expanduser().resolve()
    stability_root = Path(args.stability_output_dir).expanduser().resolve()
    stability_root.mkdir(parents=True, exist_ok=True)

    manifest_path = stability_root / "multifold_orchestration.json"
    manifest = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "folds": args.folds,
        "ssl_seed": int(args.ssl_seed),
        "supervised_seeds": args.supervised_seeds,
        "legacy_root": str(legacy_root),
        "data_dir": str(data_dir),
        "pretrain_output_dir": str(pretrain_root),
        "stability_output_dir": str(stability_root),
        "checkpoints": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 100)
    print("MIST MULTIFOLD ORCHESTRATION")
    print("=" * 100)
    print("folds:", args.folds)
    print("SSL seed:", args.ssl_seed)
    print("supervised seeds:", args.supervised_seeds)
    print("MorphMAE pretraining is sequential; no concurrent GPU jobs are launched.")
    print("Designated test subjects remain excluded from both SSL and validation-only model evaluation.")

    if not args.stability_only:
        pretrain_script = repo_root / "scripts" / "run_fold_morphmae_pretrain.py"
        for fold in args.folds:
            checkpoint = canonical_morphmae_checkpoint(pretrain_root, fold, args.ssl_seed)
            if checkpoint.is_file():
                report = validate_fold_morphmae_checkpoint(
                    checkpoint, data_dir, fold, args.ssl_seed
                )
                print(
                    f"[verified-checkpoint] fold={fold} sha256={report['sha256'][:12]} "
                    f"path={checkpoint}"
                )
                continue

            partial_best = checkpoint.parent / "legacy_output" / "best_morphmae.pt"
            if partial_best.exists() and not args.prepare_only:
                raise RuntimeError(
                    f"Fold {fold} has a legacy best checkpoint but no canonical encoder.pt: {partial_best}. "
                    "This looks like a partial previous run. Inspect it first; the multifold driver will not "
                    "silently reuse an incompletely verified SSL run."
                )

            cmd = [
                sys.executable,
                str(pretrain_script),
                "--legacy-root",
                str(legacy_root),
                "--data-dir",
                str(data_dir),
                "--fold",
                str(fold),
                "--seed",
                str(args.ssl_seed),
                "--output-dir",
                str(pretrain_root),
            ]
            if args.prepare_only:
                cmd.append("--prepare-only")
            _run(cmd, repo_root)

            if not args.prepare_only:
                validate_fold_morphmae_checkpoint(checkpoint, data_dir, fold, args.ssl_seed)

        if args.prepare_only:
            manifest["status"] = "prepared_only"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print("\nPREPARE-ONLY COMPLETE: no MorphMAE or supervised training was started.")
            return

    reports = validate_multifold_checkpoints(
        pretrain_root=pretrain_root,
        data_dir=data_dir,
        folds=args.folds,
        ssl_seed=args.ssl_seed,
    )
    manifest["checkpoints"] = reports
    manifest["status"] = "pretrain_complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nAll requested fold-specific MorphMAE checkpoints passed strict split/provenance checks:")
    for report in reports:
        print(
            f"  fold={report['fold']} val={report['val_subjects']} test={report['test_subjects']} "
            f"sha256={report['sha256'][:12]}"
        )

    if args.pretrain_only:
        manifest["status"] = "pretrain_only_complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("\nPRETRAIN-ONLY COMPLETE: supervised A1/A3/A4 stability training was not started.")
        return

    stability_script = repo_root / "scripts" / "run_mist_stability.py"
    pattern = morphmae_checkpoint_pattern(pretrain_root, args.ssl_seed)
    cmd = [
        sys.executable,
        str(stability_script),
        "--data-dir",
        str(data_dir),
        "--folds",
        ",".join(str(x) for x in args.folds),
        "--seeds",
        ",".join(str(x) for x in args.supervised_seeds),
        "--mae-checkpoint-pattern",
        pattern,
        "--output-dir",
        str(stability_root),
    ]
    if args.no_amp:
        cmd.append("--no-amp")
    if args.reuse_stability:
        cmd.append("--reuse")

    manifest["status"] = "running_stability"
    manifest["checkpoint_pattern"] = pattern
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _run(cmd, repo_root)

    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nMULTIFOLD MIST AUDIT COMPLETE")
    print("Summary:", stability_root / "mist_stability_effects.csv")
    print("Fold means:", stability_root / "mist_stability_fold_means.csv")
    print("Raw runs:", stability_root / "mist_stability_raw.csv")


if __name__ == "__main__":
    main()

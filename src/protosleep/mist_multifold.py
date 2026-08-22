from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .mist import extract_mrcnn_state_dict
from .morphmae_bridge import fold_subjects_from_npz


def parse_int_csv(value: str) -> List[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate integers are not allowed: {values}")
    return values


def canonical_morphmae_checkpoint(
    pretrain_root: Path | str,
    fold: int,
    ssl_seed: int,
) -> Path:
    return (
        Path(pretrain_root).expanduser().resolve()
        / f"fold_{int(fold):02d}"
        / f"ssl_seed_{int(ssl_seed)}"
        / "encoder.pt"
    )


def morphmae_checkpoint_pattern(pretrain_root: Path | str, ssl_seed: int) -> str:
    root = Path(pretrain_root).expanduser().resolve()
    return str(root / "fold_{fold:02d}" / f"ssl_seed_{int(ssl_seed)}" / "encoder.pt")


def validate_fold_morphmae_checkpoint(
    checkpoint: Path | str,
    data_dir: Path | str,
    fold: int,
    ssl_seed: int,
) -> Dict[str, Any]:
    """Fail closed unless a canonical MorphMAE checkpoint matches the requested fold exactly."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    _, metadata = extract_mrcnn_state_dict(path)
    expected = fold_subjects_from_npz(data_dir, int(fold))

    declared_train = metadata.get("train_subjects")
    if declared_train is None:
        raise RuntimeError(f"{path} has no train_subjects metadata")
    declared_train = sorted(int(x) for x in declared_train)
    expected_train = sorted(int(x) for x in expected["train_subjects"])
    if declared_train != expected_train:
        raise RuntimeError(
            f"{path}: train_subjects={declared_train} but fold-{fold} requires {expected_train}"
        )

    declared_fold = metadata.get("fold")
    if declared_fold is None or int(declared_fold) != int(fold):
        raise RuntimeError(f"{path}: fold metadata {declared_fold!r} does not match requested fold {fold}")

    declared_seed = metadata.get("seed")
    if declared_seed is None or int(declared_seed) != int(ssl_seed):
        raise RuntimeError(
            f"{path}: SSL seed metadata {declared_seed!r} does not match requested seed {ssl_seed}"
        )

    forbidden = set(int(x) for x in expected["val_subjects"] + expected["test_subjects"])
    overlap = sorted(forbidden.intersection(declared_train))
    if overlap:
        raise RuntimeError(f"{path}: validation/test subjects leaked into SSL train metadata: {overlap}")

    return {
        "checkpoint": str(path),
        "sha256": metadata["sha256"],
        "fold": int(fold),
        "ssl_seed": int(ssl_seed),
        "train_subjects": expected_train,
        "val_subjects": [int(x) for x in expected["val_subjects"]],
        "test_subjects": [int(x) for x in expected["test_subjects"]],
        "strict_mrcnn_compatible": True,
    }


def validate_multifold_checkpoints(
    pretrain_root: Path | str,
    data_dir: Path | str,
    folds: Iterable[int],
    ssl_seed: int,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for fold in folds:
        path = canonical_morphmae_checkpoint(pretrain_root, int(fold), int(ssl_seed))
        reports.append(
            validate_fold_morphmae_checkpoint(
                checkpoint=path,
                data_dir=data_dir,
                fold=int(fold),
                ssl_seed=int(ssl_seed),
            )
        )
    return reports

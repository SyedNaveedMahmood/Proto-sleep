from pathlib import Path

import torch

from protosleep.attnsleep import MRCNN
from protosleep.mist_multifold import (
    canonical_morphmae_checkpoint,
    morphmae_checkpoint_pattern,
    parse_int_csv,
    validate_fold_morphmae_checkpoint,
)
from protosleep.morphmae_bridge import fold_subjects_from_npz


def _fake_sleep_edf20(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subject in range(20):
        # Only the filename is inspected by fold_subjects_from_npz.
        (root / f"abc{subject:02d}_night.npz").touch()


def test_multifold_paths_and_csv(tmp_path):
    root = tmp_path / "pretrain"
    assert parse_int_csv("0,1,2,3,4") == [0, 1, 2, 3, 4]
    assert canonical_morphmae_checkpoint(root, 3, 1337) == (
        root.resolve() / "fold_03" / "ssl_seed_1337" / "encoder.pt"
    )
    assert morphmae_checkpoint_pattern(root, 1337).endswith(
        "fold_{fold:02d}/ssl_seed_1337/encoder.pt"
    )


def test_validate_fold_checkpoint_accepts_exact_split(tmp_path):
    data_dir = tmp_path / "data"
    _fake_sleep_edf20(data_dir)
    split = fold_subjects_from_npz(data_dir, 0)

    ckpt = tmp_path / "encoder.pt"
    torch.save(
        {
            "state_dict": MRCNN(30).state_dict(),
            "train_subjects": split["train_subjects"],
            "val_subjects": split["val_subjects"],
            "test_subjects": split["test_subjects"],
            "fold": 0,
            "seed": 1337,
        },
        ckpt,
    )

    report = validate_fold_morphmae_checkpoint(ckpt, data_dir, fold=0, ssl_seed=1337)
    assert report["strict_mrcnn_compatible"] is True
    assert report["val_subjects"] == [5]
    assert report["test_subjects"] == [14]
    assert 5 not in report["train_subjects"]
    assert 14 not in report["train_subjects"]


def test_validate_fold_checkpoint_rejects_wrong_fold_metadata(tmp_path):
    data_dir = tmp_path / "data"
    _fake_sleep_edf20(data_dir)
    split = fold_subjects_from_npz(data_dir, 0)

    ckpt = tmp_path / "encoder.pt"
    torch.save(
        {
            "state_dict": MRCNN(30).state_dict(),
            "train_subjects": split["train_subjects"],
            "fold": 1,
            "seed": 1337,
        },
        ckpt,
    )

    try:
        validate_fold_morphmae_checkpoint(ckpt, data_dir, fold=0, ssl_seed=1337)
    except RuntimeError as exc:
        assert "fold metadata" in str(exc)
    else:
        raise AssertionError("wrong fold metadata should be rejected")

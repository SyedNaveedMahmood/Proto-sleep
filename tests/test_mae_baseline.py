from __future__ import annotations

import pytest
import torch

from protosleep.attnsleep import MRCNN
from protosleep.mae_baseline import build_matched_a1_a2, verify_mae_checkpoint_for_fold


def _write_checkpoint(path, train_subjects, fold=0):
    model = MRCNN(30)
    state = {}
    for key, value in model.state_dict().items():
        value = value.detach().cpu().clone()
        if value.is_floating_point():
            value = value + 0.25
        state[key] = value
    torch.save(
        {
            "state_dict": state,
            "train_subjects": list(train_subjects),
            "fold": int(fold),
            "seed": 1337,
        },
        path,
    )


def test_build_matched_a1_a2_changes_only_mrcnn(tmp_path):
    ckpt = tmp_path / "encoder.pt"
    subjects = [0, 1, 2, 3]
    _write_checkpoint(ckpt, subjects, fold=0)

    a1, a2, meta = build_matched_a1_a2(123, ckpt)

    assert meta["a1_random_mrcnn_sha256"] != meta["a2_mae_mrcnn_sha256"]
    assert meta["non_mrcnn_initialization_match"] is True
    for key, value in a1.state_dict().items():
        if key.startswith("mrcnn."):
            continue
        assert torch.equal(value, a2.state_dict()[key])


def test_verify_mae_checkpoint_for_fold_requires_exact_subjects_and_fold(tmp_path):
    ckpt = tmp_path / "encoder.pt"
    _write_checkpoint(ckpt, [0, 1, 2, 3], fold=2)

    meta = verify_mae_checkpoint_for_fold(ckpt, 2, [3, 2, 1, 0])
    assert meta["fold"] == 2

    with pytest.raises(RuntimeError, match="do not equal"):
        verify_mae_checkpoint_for_fold(ckpt, 2, [0, 1, 2, 4])

    with pytest.raises(RuntimeError, match="declares fold"):
        verify_mae_checkpoint_for_fold(ckpt, 1, [0, 1, 2, 3])

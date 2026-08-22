from __future__ import annotations

import copy

import torch

from protosleep.attnsleep import AttnSleepBaseline
from protosleep.morphmae_transfer import (
    MAE_PROBE,
    MAE_STAGE_1E4,
    MAE_STAGE_2E5,
    mrcnn_relative_drift,
    set_mrcnn_trainable,
)
from protosleep.utils import cpu_state_dict


def test_transfer_recipes_are_theory_locked():
    assert MAE_PROBE.freeze_encoder is True
    assert MAE_PROBE.encoder_lr == 0.0

    assert MAE_STAGE_1E4.freeze_encoder is False
    assert MAE_STAGE_1E4.warmup_epochs == 5
    assert MAE_STAGE_1E4.encoder_lr == 1e-4
    assert MAE_STAGE_1E4.head_lr == 1e-3

    assert MAE_STAGE_2E5.freeze_encoder is False
    assert MAE_STAGE_2E5.warmup_epochs == 5
    assert MAE_STAGE_2E5.encoder_lr == 2e-5
    assert MAE_STAGE_2E5.head_lr == 1e-3


def test_set_mrcnn_trainable_only_changes_encoder_flags():
    model = AttnSleepBaseline()
    head_flags_before = {n: p.requires_grad for n, p in model.named_parameters() if not n.startswith("mrcnn.")}

    set_mrcnn_trainable(model, False)
    assert all(not p.requires_grad for p in model.mrcnn.parameters())
    assert {n: p.requires_grad for n, p in model.named_parameters() if not n.startswith("mrcnn.")} == head_flags_before

    set_mrcnn_trainable(model, True)
    assert all(p.requires_grad for p in model.mrcnn.parameters())


def test_mrcnn_relative_drift_zero_then_positive():
    model = AttnSleepBaseline()
    initial = cpu_state_dict(model.mrcnn)
    assert mrcnn_relative_drift(initial, model) == 0.0

    changed = copy.deepcopy(model)
    with torch.no_grad():
        first = next(changed.mrcnn.parameters())
        first.view(-1)[0].add_(0.25)
    assert mrcnn_relative_drift(initial, changed) > 0.0

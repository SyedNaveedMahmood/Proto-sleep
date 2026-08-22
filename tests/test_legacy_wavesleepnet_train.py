from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protosleep.data import Recording
from protosleep.legacy_wavesleepnet_train import (
    historical_training_protocol,
    load_recordings_for_subjects,
    train_legacy_model,
)


def _write_npz(path, subject_value: float = 0.0):
    x = np.full((3, 3000, 1), subject_value, dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    np.savez(path, x=x, y=y, fs=np.asarray([100], dtype=np.int64))


def test_subject_filtered_loader_never_opens_unrequested_npz(tmp_path):
    good = tmp_path / "SC4001E0.npz"  # filename[3:5] -> subject 00
    bad = tmp_path / "SC4011E0.npz"   # filename[3:5] -> subject 01
    _write_npz(good)
    bad.write_bytes(b"this is deliberately not an npz file")

    recordings, meta = load_recordings_for_subjects(tmp_path, [0])
    assert {r.subject_id for r in recordings} == {0}
    assert meta["subjects"] == [0]
    assert meta["opened_files"] == [str(good)]

    try:
        load_recordings_for_subjects(tmp_path, [1])
    except Exception:
        pass
    else:
        raise AssertionError("Expected corrupt requested subject file to be opened and fail")


def test_historical_training_protocol_records_overrides():
    cfg = {
        "training_params": {
            "batch_size": 64,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "max_epochs": 5000,
            "early_stopping": {"mode": "max", "patience": 50},
        }
    }
    frozen = historical_training_protocol(cfg)
    assert frozen["batch_size"] == 64
    assert frozen["max_epochs"] == 5000
    assert frozen["patience"] == 50
    assert frozen["selection_metric"] == "validation_macro_f1"
    assert frozen["max_epochs_overridden"] is False
    assert frozen["patience_overridden"] is False

    dev = historical_training_protocol(cfg, max_epochs_override=2, patience_override=1)
    assert dev["max_epochs"] == 2
    assert dev["patience"] == 1
    assert dev["max_epochs_overridden"] is True
    assert dev["patience_overridden"] is True


class DummyTrainer:
    def protop_loss(self, outputs, labels):
        ce = F.cross_entropy(outputs, labels)
        weight = 0.01 * self.model.module.fc.weight.abs().sum()
        self.loss_ensemble["cross_entropy"] = ce
        self.loss_ensemble["weight_loss"] = weight
        return ce + weight

    def _diversity_cal(self, x):
        return x.square().mean() + 1.0


class DummyLegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prototype_vectors = nn.Parameter(torch.rand(2, 1, 1))
        self.fc = nn.Linear(1, 5)

    def forward(self, x):
        pooled = x.mean(dim=-1)
        return self.fc(pooled)


def _recording(name: str, x: np.ndarray, y: np.ndarray) -> Recording:
    return Recording(
        path=name,
        recording_id=name,
        subject_id=0,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        fs=100,
    )


def test_train_legacy_model_runs_validation_only_selection(tmp_path):
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(8, 1, 3000)).astype(np.float32)
    val_x = rng.normal(size=(4, 1, 3000)).astype(np.float32)
    train_y = np.asarray([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
    val_y = np.asarray([0, 1, 2, 3], dtype=np.int64)
    train = [_recording("train", train_x, train_y)]
    val = [_recording("val", val_x, val_y)]

    cfg = {
        "classifier": {},
        "training_params": {
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "early_stopping": {"mode": "max", "patience": 1},
        },
    }
    trainer_module = SimpleNamespace(OneFoldTrainer=DummyTrainer)
    ckpt = tmp_path / "dummy.pt"
    history = tmp_path / "history.csv"

    result = train_legacy_model(
        DummyLegacyModel(),
        train,
        val,
        trainer_module,
        cfg,
        "cpu",
        seed=123,
        checkpoint_path=ckpt,
        history_path=history,
    )

    assert ckpt.is_file()
    assert history.is_file()
    assert 1 <= result["best_epoch"] <= 2
    assert np.isfinite(result["best_val_macro_f1"])
    assert result["protocol"]["selection_metric"] == "validation_macro_f1"

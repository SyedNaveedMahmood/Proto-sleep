"""In-memory training views and exact-state probes; no real loader invocation."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from mist_cove.model import CoveConfig
from mist_evidence.model import ModelConfig
from mist_evidence.runtime import TrainConfig, seed_all, load_trusted
from .adapter import build_adapter
from .provenance import environment
from .training import configure_precision, sealed_train


@dataclass
class MockRecording:
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    scoring_mask: np.ndarray
    subject: str
    split: str
    recording_id: str
    file_hash: str

    def segments(self):
        valid = np.flatnonzero(self.scoring_mask)
        if not len(valid):
            return []
        cuts = np.r_[0, np.flatnonzero((np.diff(valid) != 1) | (np.diff(self.indices[valid]) != 1)) + 1, len(valid)]
        return [(int(valid[a]), int(valid[b - 1] + 1)) for a, b in zip(cuts[:-1], cuts[1:])]


def mock_recordings():
    rng = np.random.default_rng(714)
    rows = []
    for role in ("train", "val"):
        x = rng.normal(size=(10, 1, 3000)).astype(np.float32)
        y = np.arange(10, dtype=np.int64) % 5
        y[3] = -1
        indices = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        rows.append(MockRecording(x, y, indices, y >= 0, "synthetic_" + role, role,
                                  "synthetic_record_" + role, hashlib.sha256(x.tobytes()).hexdigest()))
    return rows


def synthetic_bank():
    cfg = ModelConfig(radius=0, use_crf=False)
    g = torch.Generator().manual_seed(714)
    return {"config": cfg.dictionary(), "waveforms": [torch.randn(cfg.prototypes_per_scale, scale, generator=g) for scale in cfg.scales],
            "amplitude_stats": [0., 1.], "scope": "SYNTHETIC_MOCK_ONLY_NOT_A_REAL_TRAIN_BANK"}


def assert_exact(a, b, path="state"):
    if torch.is_tensor(a):
        if not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError("tensor resume mismatch: " + path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError("NumPy resume mismatch: " + path)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("state keys mismatch: " + path)
        for key in a:
            assert_exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError("state lengths mismatch: " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_exact(x, y, f"{path}[{i}]")
    elif a != b:
        raise AssertionError("scalar resume mismatch: " + path)


def exact_resume_probe(output, method="q1", device="cpu"):
    configure_precision(device)
    train_row, val_row = mock_recordings()
    bank = synthetic_bank() if method in {"evidence", "raw_context"} else None
    cfg = TrainConfig(epochs=2, patience=12, core_epochs=4, encode_batch=4, seed=123)
    normalization = {"mean_uV": 0., "std_uV": 1.}

    def fresh():
        seed_all(123, True)
        return build_adapter(method, normalization, bank)

    output = Path(output)
    kwargs = {"provenance": {"scope": "SYNTHETIC_MOCK_ONLY", "method": method}, "device": device, "bank": bank}
    sealed_train(fresh(), [train_row], [val_row], cfg, output / "full", **kwargs)
    try:
        sealed_train(fresh(), [train_row], [val_row], cfg, output / "resumed", interrupt_after_epoch=1, **kwargs)
    except InterruptedError:
        pass
    else:
        raise AssertionError("test interruption did not occur")
    sealed_train(fresh(), [train_row], [val_row], cfg, output / "resumed", resume=True, **kwargs)
    left = load_trusted(output / "full/last.pt")
    right = load_trusted(output / "resumed/last.pt")
    for key in ("state_dict", "optimizer", "rng", "epoch", "best_score", "best_epoch", "stale"):
        assert_exact(left[key], right[key], key)
    semantic = ["epoch", "train_nll", "val_nll", "val_subject_macro_f1", "val_pooled_macro_f1", "val_accuracy", "best_epoch", "stale", "optimizer_steps", "last_grad_norm"]
    assert_exact([{k: h[k] for k in semantic} for h in left["history"]], [{k: h[k] for k in semantic} for h in right["history"]], "validation_history")
    # A completed resume must validate hashes and reevaluate without optimization.
    sealed_train(fresh(), [train_row], [val_row], cfg, output / "resumed", resume=True, **kwargs)
    return {"method": method, "device": str(device), "model_optimizer_rng_history_exact": True,
            "epochs": left["epoch"], "optimizer_steps": sum(h["optimizer_steps"] for h in left["history"]),
            "parameters": sum(v.numel() for k, v in left["state_dict"].items() if not k.endswith(("mean_uV", "std_uV"))),
            "peak_gpu_allocated_bytes": max(h["peak_gpu_allocated_bytes"] for h in left["history"]),
            "peak_gpu_reserved_bytes": max(h["peak_gpu_reserved_bytes"] for h in left["history"]),
            "environment": environment(device), "real_data_accessed": False}

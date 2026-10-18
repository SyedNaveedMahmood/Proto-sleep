"""Mock source adaptation, sealed checkpoints, exact CPU/CUDA state recovery."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mist_evidence.data import blocks
from mist_evidence.runtime import TrainConfig, atomic_torch_save, seed_all, load_trusted
from mist_transfer.experiment import evaluate_source
from mist_cove.phase2.adapter import SourceAdapter, build_adapter, adapter_logits
from mist_cove.phase2.provenance import ROOT, file_sha, digest, implementation_snapshot
from mist_cove.phase2.source import validate_source_config, training_config, load_role, run_source, validate_normalization, validate_launch_receipt
from mist_cove.phase2.synthetic_runtime import mock_recordings, synthetic_bank, exact_resume_probe
from mist_cove.phase2.training import sealed_train, validate_views


@pytest.fixture(autouse=True)
def cpu_threads(): torch.set_num_threads(2)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/mist_cove_source_prepared_v1.json").read_text())


@pytest.mark.parametrize("method", ["q1", "availability", "background", "rounded_q1", "raw_context", "evidence"])
def test_adapter_full_physical_validation_masks_and_finite_gradients(method):
    train, val = mock_recordings(); bank = synthetic_bank() if method in {"raw_context", "evidence"} else None
    seed_all(123); model = build_adapter(method, {"mean_uV": 2., "std_uV": 10.}, bank)
    x = torch.from_numpy(train.x[:5]); labels = torch.from_numpy(train.y[:5])[None]; mask = labels >= 0
    logits = model.emissions(model.encode(x)[None]); loss = model.loss(logits, labels, mask)
    torch.testing.assert_close(adapter_logits(model, x), logits[0], rtol=0, atol=0)
    assert logits.shape == (1, 5, 5) and torch.isfinite(loss)
    loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    metrics, rows = evaluate_source(model, [val], torch.device("cpu"), 3)
    assert len(rows) == 10 and metrics["n_epochs"] == 9 and metrics["unscored_epochs"] == 1
    assert rows[3]["truth"] is None and rows[4]["original_epoch_index"] == 5
    assert metrics["mean_subject_macro_f1"] == metrics["macro_f1"]
    covered = [i for _, _, _, lo, hi in blocks([train], 16, 0) for i in range(lo, hi)]
    assert covered == [0, 1, 2, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("method", ["q1", "rounded_q1", "background", "raw_context", "evidence"])
def test_exact_cpu_interrupted_resume(tmp_path, method):
    result = exact_resume_probe(tmp_path / method, method, "cpu")
    assert result["model_optimizer_rng_history_exact"] and result["epochs"] == 2
    assert result["peak_gpu_allocated_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")
@pytest.mark.parametrize("method", ["q1", "availability", "background", "rounded_q1", "raw_context", "evidence"])
def test_exact_cuda_interrupted_resume(tmp_path, method):
    result = exact_resume_probe(tmp_path / method, method, "cuda")
    assert result["model_optimizer_rng_history_exact"]
    assert result["environment"]["gpu_name"] and result["peak_gpu_allocated_bytes"] > 0


@pytest.mark.parametrize("change", ["recipe", "model", "provenance", "indices", "labels", "bank"])
def test_resume_refuses_identity_changes_before_deserialization(tmp_path, change):
    train, val = mock_recordings(); cfg = TrainConfig(epochs=2, core_epochs=4, encode_batch=4)
    seed_all(123); model = SourceAdapter()
    with pytest.raises(InterruptedError):
        sealed_train(model, [train], [val], cfg, tmp_path / "run", provenance={"scope": "synthetic"}, interrupt_after_epoch=1)
    seed_all(123); model = SourceAdapter()
    provenance = {"scope": "synthetic"}; bank = None
    if change == "recipe": cfg = replace(cfg, lr=.001)
    if change == "model": model = SourceAdapter("rounded_q1")
    if change == "provenance": provenance["scope"] = "changed"
    if change == "indices": train.indices[-1] += 1
    if change == "labels": train.y[0] = 2
    if change == "bank": bank = {"waveforms": [], "type": "changed"}
    with patch("mist_evidence.runtime.load_trusted", side_effect=AssertionError("unverified deserialize")):
        with pytest.raises(ValueError, match="identity changed"):
            sealed_train(model, [train], [val], cfg, tmp_path / "run", provenance=provenance, resume=True, bank=bank)


def test_checkpoint_tamper_rejected_and_output_not_overwritten(tmp_path):
    train, val = mock_recordings(); cfg = TrainConfig(epochs=2, core_epochs=4, encode_batch=4)
    with pytest.raises(InterruptedError):
        sealed_train(SourceAdapter(), [train], [val], cfg, tmp_path / "run", provenance={}, interrupt_after_epoch=1)
    with pytest.raises(FileExistsError):
        sealed_train(SourceAdapter(), [train], [val], cfg, tmp_path / "run", provenance={})
    (tmp_path / "run/last.pt").write_bytes(b"untrusted corrupted checkpoint")
    with patch("mist_evidence.runtime.load_trusted", side_effect=AssertionError("unsafe deserialize")):
        with pytest.raises(ValueError, match="digest mismatch"):
            sealed_train(SourceAdapter(), [train], [val], cfg, tmp_path / "run", provenance={}, resume=True)


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "state.pt"; atomic_torch_save({"x": torch.tensor([1.])}, path)
    old = path.read_bytes()
    with patch("mist_evidence.runtime.os.replace", side_effect=OSError("simulated interruption before replace")):
        with pytest.raises(OSError): atomic_torch_save({"x": torch.tensor([2.])}, path)
    assert path.read_bytes() == old


@pytest.mark.parametrize("role", ["test", "reserved", "ST", "SHHS"])
def test_roles_rejected_before_any_file_access(role):
    train, val = mock_recordings(); train.split = role
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        with pytest.raises(ValueError, match="before any file"): validate_views([train], [val])
        with pytest.raises(ValueError, match="prohibited"): load_role({}, role)


def test_source_authorization_and_approval_receipt_fail_closed_before_file_access():
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        with pytest.raises(ValueError, match="authorization required"): run_source("unused", "unused", "unused")
        with pytest.raises(ValueError, match="launch receipt required"):
            run_source("unused", "unused", "unused", authorize_source_train_val=True)


@pytest.mark.parametrize("change", [{"lr": .001}, {"max_epochs": 1}, {"patience": 1}, {"amp": True}, {"tf32": True},
                                    {"core_epochs": 8}, {"seeds": [123]}, {"reserved_access": True},
                                    {"variants": ["q1"]}, {"temporal_radius": 1}, {"split_file_sha256": "bad"}])
def test_frozen_source_recipe_is_metadata_only(config, change):
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        validate_source_config(config)
        recipe = training_config(config, 123)
        assert recipe.epochs == 60 and recipe.patience == 12 and recipe.lr == .0003
        config.update(change)
        with pytest.raises(ValueError, match="configuration violated"): validate_source_config(config)


def test_masks_and_subject_overlap_rejected():
    train, val = mock_recordings(); val.subject = train.subject
    with pytest.raises(ValueError, match="leakage"): validate_views([train], [val])
    train, val = mock_recordings(); val.indices[1] = val.indices[0]
    with pytest.raises(ValueError, match="physical-grid"): validate_views([train], [val])
    train, val = mock_recordings(); train.scoring_mask[3] = True
    with pytest.raises(ValueError, match="physical-grid"): validate_views([train], [val])


def test_synthetic_readiness_receipt_cannot_authorize_source(config):
    with pytest.raises(ValueError, match="BLOCKED"):
        validate_launch_receipt({"scope": "SYNTHETIC_ONLY_NOT_SOURCE_AUTHORIZATION", "status": "PASS"}, config)

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from protosleep.evidence.cli import synthetic_manifest
from protosleep.evidence.explain import explain_epoch
from protosleep.evidence.model import EvidenceConfig
from protosleep.evidence.train import (TrainConfig, fit_context, fit_local, safe_load)


def configs():
    return (EvidenceConfig(scales_samples=(100,), anchors_per_class=1, embedding_dim=8, pool_per_subject=20),
            TrainConfig(epochs=2, patience=2, batch_size=10, log_every=0))


def test_full_pipeline_safe_serialization_and_completeness(tmp_path):
    torch.set_num_threads(2)
    manifest = synthetic_manifest(tmp_path / "data")
    mc, tc = configs()
    done = fit_local(manifest, tmp_path / "local", mc, tc)
    assert done["status"] == "complete"
    repeated = fit_local(manifest, tmp_path / "local", mc, tc, resume=True)
    assert repeated == done
    with pytest.raises(ValueError, match="fingerprint"):
        fit_local(manifest, tmp_path / "local", mc, replace(tc, lr=1e-3), resume=True)
    fit_context(manifest, tmp_path / "local/best.pt", tmp_path / "context", epochs=2, patience=2)
    report = explain_epoch(manifest, tmp_path / "local/best.pt", tmp_path / "report",
                           context_checkpoint=tmp_path / "context/best.pt", top_k=1, perturb=True)
    assert report["completeness_error"] < 1e-4
    assert (tmp_path / "report/explanation.html").stat().st_size > 1000
    assert "input_intervention" in report
    with pytest.raises(ValueError):
        fit_context(manifest, tmp_path / "local/best.pt", tmp_path / "context", epochs=3, patience=2, resume=True)


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import protosleep.evidence.train as train
    torch.set_num_threads(2)
    manifest = synthetic_manifest(tmp_path / "data")
    mc, tc = configs()
    fit_local(manifest, tmp_path / "uninterrupted", mc, tc)
    original_write = train.atomic_torch
    def interrupt(path, payload):
        original_write(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise RuntimeError("simulated power loss AFTER atomic epoch commit")
    monkeypatch.setattr(train, "atomic_torch", interrupt)
    with pytest.raises(RuntimeError, match="power loss"):
        fit_local(manifest, tmp_path / "resumed", mc, tc)
    assert not (tmp_path / "resumed/COMPLETE.json").exists()
    monkeypatch.setattr(train, "atomic_torch", original_write)
    fit_local(manifest, tmp_path / "resumed", mc, tc, resume=True)
    a = safe_load(tmp_path / "uninterrupted/last.pt")
    b = safe_load(tmp_path / "resumed/last.pt")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    assert a["best"] == b["best"]

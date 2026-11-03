import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mist_evidence.cli import synthetic_manifest, variant_config
from mist_evidence.data import (blocks, build_anchors, edf_manifest, load_split,
                               validate_manifest)
from mist_evidence.explain import export_explanation
from mist_evidence.model import EvidenceModel, ModelConfig
from mist_evidence.runtime import TrainConfig, load_trusted, seed_all, tensor_digest, train


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def data(tmp_path):
    manifest = synthetic_manifest(tmp_path / "signals")
    return manifest, load_split(manifest, "train"), load_split(manifest, "val")


def test_selective_loading_reserved_test_is_never_opened(tmp_path):
    manifest, tr, va = data(tmp_path)
    assert len(tr) == 2 and len(va) == 1
    with pytest.raises(ValueError, match="test access"):
        load_split(manifest, "test")
    manifest["records"][-1]["subject"] = tr[0].subject
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(manifest)


def test_discontinuous_indices_blocks_and_continuity_guard(tmp_path):
    manifest = synthetic_manifest(tmp_path / "signals")
    p = Path(manifest["records"][0]["path"])
    with np.load(p) as z:
        x, y = z["x"], z["y"]
    indices = np.r_[np.arange(5), np.arange(10,15)]
    np.savez(p, x=x, y=y, fs=100, epoch_indices=indices)
    tr = load_split(manifest, "train")
    first = tr[0]
    assert first.segments() == [(0,5), (5,10)]
    covered = []
    for ri, a, b, lo, hi in blocks([first], 3, 2):
        assert (a < 5 and b <= 5) or a >= 5
        covered.extend(range(lo,hi))
    assert covered == list(range(10))
    np.savez(p, x=x, y=y, fs=100)
    with pytest.raises(ValueError, match="epoch_indices"):
        load_split(manifest, "train")
    assert load_split(manifest, "train", assume_contiguous=True)[0].continuity_assumed


def test_strict_npz_schema_and_fs(tmp_path):
    manifest = synthetic_manifest(tmp_path / "signals")
    p = Path(manifest["records"][0]["path"])
    with np.load(p) as z:
        x,y,idx = z["x"],z["y"],z["epoch_indices"]
    np.savez(p,x=x,y=y+0.2,fs=100,epoch_indices=idx)
    with pytest.raises(ValueError,match="labels"):
        load_split(manifest,"train")
    np.savez(p,x=x,y=y,fs=99.9,epoch_indices=idx)
    with pytest.raises(ValueError,match="100 Hz"):
        load_split(manifest,"train")


def test_edf_filename_split_without_opening_files(tmp_path):
    for sid in range(20):
        (tmp_path/f"SC4{sid:02d}1E0.npz").write_bytes(b"not an npz; do not open")
    manifest = edf_manifest(tmp_path,0)
    validate_manifest(manifest)
    assert [e["subject"] for e in manifest["records"] if e["split"]=="val"] == ["5"]
    assert [e["subject"] for e in manifest["records"] if e["split"]=="test"] == ["14"]


def test_actual_training_anchors_and_matching_initialization(tmp_path):
    _, tr, _ = data(tmp_path)
    cfg = ModelConfig(scales=(100,200),prototypes_per_scale=2,embedding_dim=8,radius=0)
    bank = build_anchors(tr,cfg,per_subject=8)
    records = {r.path:r for r in tr}
    for scale,waves in enumerate(bank["waveforms"]):
        for wave,m in zip(waves,bank["metadata"][scale]):
            r = records[m["source_path"]]
            segment = r.x[m["array_epoch"],0,m["start_sample"]:m["start_sample"]+m["length_samples"]]
            assert np.array_equal(wave.numpy(),segment)
            assert m["subject"] in bank["training_subjects"]
    seed_all(123); a = EvidenceModel(cfg,bank["waveforms"])
    seed_all(123); b = EvidenceModel(variant_config("raw_context",cfg),bank["waveforms"])
    assert tensor_digest(a.banks[0].encoder.state_dict()) == tensor_digest(b.banks[0].encoder.state_dict())


def test_resume_is_exact_and_completion_is_not_best_so_far(tmp_path):
    _, tr, va = data(tmp_path)
    cfg = ModelConfig(scales=(100,),prototypes_per_scale=2,embedding_dim=8,radius=0,use_crf=False)
    bank = build_anchors(tr,cfg,per_subject=8)
    recipe = TrainConfig(epochs=2,patience=5,core_epochs=5,encode_batch=5)
    def fresh():
        seed_all(123)
        return EvidenceModel(cfg,bank["waveforms"])
    train(fresh(),tr,va,recipe,tmp_path/"full",bank,{"code":"test"},torch.device("cpu"))
    with pytest.raises(InterruptedError):
        train(fresh(),tr,va,recipe,tmp_path/"resume",bank,{"code":"test"},torch.device("cpu"),interrupt_after_epoch=1)
    assert (tmp_path/"resume"/"best.pt").exists()
    assert not (tmp_path/"resume"/"COMPLETE.json").exists()
    model, metrics, _ = train(fresh(),tr,va,recipe,tmp_path/"resume",bank,{"code":"test"},torch.device("cpu"),resume=True)
    a = load_trusted(tmp_path/"full"/"last.pt")["state_dict"]
    b = load_trusted(tmp_path/"resume"/"last.pt")["state_dict"]
    assert tensor_digest(a) == tensor_digest(b)
    assert (tmp_path/"resume"/"COMPLETE.json").exists()
    train(fresh(),tr,va,recipe,tmp_path/"resume",bank,{"code":"test"},torch.device("cpu"),resume=True)
    with pytest.raises(ValueError,match="signature"):
        train(fresh(),tr,va,recipe,tmp_path/"resume",bank,{"code":"changed"},torch.device("cpu"),resume=True)
    report = export_explanation(model,va[0],3,bank,tmp_path/"explain",torch.device("cpu"),5)
    assert report["score_reconstruction_max_abs_error"] < 2e-5
    assert (tmp_path/"explain"/"index.html").exists()
    total = (report["center_epoch_evidence_difference"] + report["neighbor_evidence_difference"]
             + report["bias_difference"] + report["crf_neighbor_transition_difference"])
    assert abs(total-report["conditional_path_score_difference"]) < 1e-5


def test_independent_epochs_do_not_invent_continuity(tmp_path):
    manifest = synthetic_manifest(tmp_path / "signals")
    p = Path(manifest["records"][0]["path"])
    with np.load(p) as z:
        x,y = z["x"],z["y"]
    np.savez(p,x=x,y=y,fs=100)
    rec = load_split(manifest,"train",independent_epochs=True)[0]
    assert not rec.indices_known and not rec.continuity_assumed
    assert rec.segments() == [(i,i+1) for i in range(len(y))]


def test_reference_defaults_match_typed_configuration():
    from dataclasses import asdict
    root = Path(__file__).resolve().parents[2]
    defaults = json.loads((root/"configs"/"mist_evidence_v1.json").read_text())
    assert defaults["model"] == json.loads(json.dumps(asdict(ModelConfig())))
    assert defaults["training"] == json.loads(json.dumps(asdict(TrainConfig())))

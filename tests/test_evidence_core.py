"""Numerical contracts, not a claim of validation on real sleep recordings."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from protosleep.evidence.cli import synthetic_manifest
from protosleep.evidence.crf import EvidenceCRF
from protosleep.evidence.data import (Epochs, contiguous_runs, load_split, validate_manifest, write_json)
from protosleep.evidence.model import (BANDS, CropEncoder, EvidenceConfig, EvidenceNet, build_bank, morphology)


@pytest.fixture(autouse=True)
def small_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def dataset(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    return manifest, load_split(manifest, "train"), load_split(manifest, "val")


@pytest.fixture
def candidate(dataset):
    cfg = EvidenceConfig(scales_samples=(100, 200), anchors_per_class=1, embedding_dim=8, pool_per_subject=20)
    bank = build_bank(dataset[1], cfg, 17)
    torch.manual_seed(17)
    return EvidenceNet(cfg, bank), bank


@pytest.mark.parametrize("shape", [(7, 1, 100), (2, 1, 200), (1, 1, 400)])
def test_descriptors_finite_for_flat_signal(shape):
    d = morphology(torch.zeros(shape))
    assert d.shape == (shape[0], 8)
    assert torch.isfinite(d).all()
    torch.testing.assert_close(d[:, :5].exp().sum(-1), torch.ones(shape[0]))


@pytest.mark.parametrize("freq,band", [(2, 0), (6, 1), (10, 2), (14, 3), (22, 4)])
def test_measured_sinusoid_band(freq, band):
    t = torch.arange(400) / 100
    d = morphology(torch.sin(2*torch.pi*freq*t)[None, None])
    assert int(d[0, :5].argmax()) == band


def test_amplitude_measurement_not_silently_erased():
    x = torch.randn(2, 1, 400)
    a, b = morphology(x), morphology(3*x)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b[:, 5]-a[:, 5], torch.full((2,), np.log(3)), atol=1e-5, rtol=1e-5)


def test_bank_contains_exact_training_crops(candidate, dataset):
    model, bank = candidate
    nights = {n.key: n for n in dataset[1]}
    assert bank["train_subjects"] == ["0", "1", "2"]
    for scale in bank["scales"]:
        for raw, ref in zip(scale["raw"], scale["refs"]):
            n = nights[ref["recording"]]
            wanted = n.x[ref["epoch_row"], :, ref["start_sample"]:ref["end_sample"]]
            np.testing.assert_array_equal(raw.numpy(), wanted)
            assert ref["clinical_event"] == "unreviewed"
    assert not any(name.startswith("anchors_") for name, _ in model.named_parameters())


def test_val_cannot_construct_bank(dataset):
    with pytest.raises(ValueError, match="training subjects only"):
        build_bank(dataset[2], EvidenceConfig(), 1)


def test_frozen_test_file_is_never_read(dataset):
    # Fixture test file is corrupt on purpose. Loading permitted roles already succeeded.
    assert len(Epochs(dataset[1])) == 30
    with pytest.raises(ValueError, match="cannot open test"):
        load_split(dataset[0], "test")


def test_subject_leak_rejected(dataset):
    manifest = dataset[0]
    d = json.loads(manifest.read_text())
    d["records"][-1]["subject"] = d["records"][0]["subject"]
    write_json(manifest, d)
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(manifest)


def test_fractional_labels_are_rejected(dataset):
    manifest, train, _ = dataset
    d = json.loads(manifest.read_text())
    p = Path(d["data_root"]) / train[0].key
    np.savez(p, x=train[0].x, y=train[0].y.astype(float)+0.1, fs=100)
    with pytest.raises(ValueError, match="integers"):
        load_split(manifest, "train")


@pytest.mark.parametrize("problem", ["fs", "shape", "nan"])
def test_signal_contract_rejected(dataset, problem):
    manifest, train, _ = dataset
    d = json.loads(manifest.read_text())
    p = Path(d["data_root"]) / train[0].key
    x = train[0].x.copy()
    fs = 100.1 if problem == "fs" else 100
    if problem == "shape":
        x = x[..., :2000]
    if problem == "nan":
        x[0, 0, 5] = np.nan
    np.savez(p, x=x, y=train[0].y, fs=fs)
    with pytest.raises(ValueError):
        load_split(manifest, "train")


def test_unknown_labels_and_index_gaps_break_context(dataset):
    n = dataset[1][0]
    n.y[2] = -1
    n.epoch_index[6:] += 2
    runs = contiguous_runs(n)
    assert [len(r) for r in runs] == [2, 3, 4]
    n.continuity_verified = False
    with pytest.raises(ValueError, match="continuity not verified"):
        contiguous_runs(n)


def test_no_cross_crop_batch_statistics():
    torch.manual_seed(99)
    encoder = CropEncoder(8).train()
    a, b = torch.randn(1, 1, 200), torch.randn(3, 1, 200)*100
    torch.testing.assert_close(encoder(a)[0], encoder(torch.cat([a, b]))[0], atol=2e-6, rtol=2e-6)


def test_local_ledger_and_feature_ablation_are_exact(candidate):
    model, _ = candidate
    out = model(torch.randn(3, 1, 3000), details=True)
    assert out["logits"].shape == (3, 5)
    for c, r in ((0, 1), (4, 2)):
        contributions, bias, margin = model.margin_ledger(out["features"], c, r)
        torch.testing.assert_close(margin, out["logits"][:, c]-out["logits"][:, r], atol=2e-6, rtol=2e-6)
        changed = out["features"].clone()
        changed[:, 0] = 0
        changed_logits = model.classifier(changed)
        new_margin = changed_logits[:, c]-changed_logits[:, r]
        torch.testing.assert_close(margin-new_margin, contributions[:, 0], atol=2e-6, rtol=2e-6)


def test_identity_exemplar_similarity(candidate):
    model, _ = candidate
    x = torch.randn(1, 1, 3000)
    x[:, :, :100] = model.anchors_0[:1]
    result = model(x, details=True)
    assert float(result["details"][0]["similarity"][0, 0, 0].detach()) > 0.99999


@pytest.mark.parametrize("arm", ["evidence", "dense", "descriptor"])
def test_finite_optimizer_update(dataset, arm):
    cfg = EvidenceConfig(scales_samples=(100,), anchors_per_class=1, embedding_dim=8, pool_per_subject=20, arm=arm)
    model = EvidenceNet(cfg, build_bank(dataset[1], cfg, 7))
    x = torch.from_numpy(dataset[1][0].x[:5])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.classifier.weight.detach().clone()
    out = model(x)
    loss = F.cross_entropy(out["logits"], torch.arange(5))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)


def test_auxiliary_branch_is_real_and_optional(dataset):
    cfg = EvidenceConfig(scales_samples=(100,), anchors_per_class=1, embedding_dim=8, pool_per_subject=20, aux_morph_weight=.1)
    model = EvidenceNet(cfg, build_bank(dataset[1], cfg, 7))
    result = model(torch.randn(2, 1, 3000))
    result["aux_loss"].backward()
    assert model.morph_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("length", [1, 2, 4])
def test_crf_matches_exhaustive_enumeration(length):
    torch.manual_seed(length)
    crf = EvidenceCRF(classes=3)
    with torch.no_grad():
        for p in crf.parameters():
            p.copy_(torch.randn_like(p)*.2)
    emissions = torch.randn(length, 3, dtype=torch.float64, requires_grad=True)
    a, start, end = crf.potentials()
    paths = list(itertools.product(range(3), repeat=length))
    energies = []
    for seq in paths:
        score = start[seq[0]] + end[seq[-1]] + sum(emissions[t, c] for t, c in enumerate(seq))
        score = score + sum(a[seq[t-1], seq[t]] for t in range(1, length))
        energies.append(score)
    energy = torch.stack(energies)
    z = energy.logsumexp(0)
    torch.testing.assert_close(z, crf.messages(emissions)[2])
    marginals = crf.log_marginals(emissions).exp()
    for t in range(length):
        for c in range(3):
            expected = energy[[seq[t] == c for seq in paths]].logsumexp(0).sub(z).exp()
            torch.testing.assert_close(marginals[t, c], expected)
    y = torch.tensor(paths[0])
    loss = crf.nll(emissions, y)
    torch.testing.assert_close(loss, z-energy[0])
    loss.backward()
    assert torch.isfinite(emissions.grad).all()
    for c, r in ((0, 1), (2, 0)):
        for t in range(length):
            ledger = crf.margin_ledger(emissions, t, c, r)
            torch.testing.assert_close(ledger["margin"], marginals[t, c].log()-marginals[t, r].log())


def test_zero_crf_is_independent_classifier_and_handles_large_scores():
    crf = EvidenceCRF()
    e = torch.randn(1500, 5, dtype=torch.float64)*5
    torch.testing.assert_close(crf.log_marginals(e), e.log_softmax(-1), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("bad", [torch.zeros(0, 5), torch.full((2, 5), float("nan"))])
def test_bad_crf_input_rejected(bad):
    with pytest.raises(ValueError):
        EvidenceCRF().messages(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present in this test environment")
def test_cuda_forward_backward(candidate):
    model, _ = candidate
    model = model.cuda()
    result = model(torch.randn(2, 1, 3000, device="cuda"))
    result["logits"].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

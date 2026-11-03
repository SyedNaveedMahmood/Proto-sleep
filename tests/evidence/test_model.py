import itertools
from dataclasses import replace

import pytest
import torch

from mist_evidence.model import (AdditiveTemporal, EvidenceModel, LinearCRF, ModelConfig,
                                RawTemporal, WaveformBank, epoch_summary, normalize_wave)


def small_model(**kw):
    torch.manual_seed(8)
    cfg = ModelConfig(scales=(100, 200), prototypes_per_scale=2, embedding_dim=8, radius=2, **kw)
    return EvidenceModel(cfg, [torch.randn(2, s) for s in cfg.scales])


def test_crf_partition_decode_and_marginals_against_enumeration():
    torch.manual_seed(4)
    crf = LinearCRF(3)
    for p in crf.parameters():
        p.data.normal_()
    emissions = torch.randn(1, 3, 3, requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    paths = torch.tensor(list(itertools.product(range(3), repeat=3)))
    scores = torch.stack([crf.path_score(emissions, y[None], valid)[0] for y in paths])
    assert torch.allclose(crf.log_partition(emissions, valid)[0], scores.logsumexp(0), atol=1e-6)
    best = crf.decode(emissions, valid)[0]
    assert torch.equal(best, paths[int(scores.argmax())])
    marginal = crf.marginals(emissions[0])
    expected = torch.zeros(3, 3)
    for prob, path in zip(scores.softmax(0), paths):
        for t, state in enumerate(path):
            expected[t, state] += prob
    assert torch.allclose(marginal, expected, atol=1e-6)
    assert torch.allclose(marginal.sum(-1), torch.ones(3), atol=1e-6)
    loss = crf.loss(emissions, paths[:1], valid)
    loss.backward()
    assert loss >= 0 and torch.isfinite(emissions.grad).all()


def test_crf_padding_and_bad_masks():
    crf = LinearCRF()
    e = torch.randn(2, 5, 5)
    y = torch.randint(0, 5, (2, 5))
    mask = torch.tensor([[1,1,1,1,1], [1,1,0,0,0]], dtype=torch.bool)
    z = crf.log_partition(e, mask)
    e[1,2:] = 1000
    y[1,2:] = -100
    assert torch.allclose(z, crf.log_partition(e, mask))
    assert torch.isfinite(crf.loss(e, y, mask))
    assert (crf.decode(e, mask)[1,2:] == -1).all()
    mask[1,3] = True
    with pytest.raises(ValueError, match="contiguous"):
        crf.loss(e, y, mask)


@pytest.mark.parametrize("radius", [0, 2, 10])
def test_additive_contributions_and_feature_intervention(radius):
    torch.manual_seed(2)
    head = AdditiveTemporal(7, radius)
    x = torch.rand(1, 25, 7)
    scores = head(x)
    for t in (0, 12, 24):
        parts = head.explain_at(x[0], t)
        assert torch.allclose(scores[0,t], parts.sum((1,2))+head.bias, atol=2e-6)
    t, f, lag = 12, 2, min(2, radius)
    parts = head.explain_at(x[0], t)
    modified = x.clone(); modified[0,t+lag,f] = 0
    delta = scores[0,t] - head(modified)[0,t]
    assert torch.allclose(delta, parts[:,radius+lag,f], atol=2e-6)


@pytest.mark.parametrize("raw", [False, True])
def test_halo_scores_equal_full_record_scores(raw):
    torch.manual_seed(2)
    head = RawTemporal(7, 10) if raw else AdditiveTemporal(7, 10)
    x = torch.rand(1, 60, 7)
    assert torch.allclose(head(x)[0,20:30], head(x[:,10:40])[0,10:20], atol=2e-6)


def test_locality_batch_invariance_and_immutable_anchors():
    model = small_model().eval()
    bank = model.banks[0]
    x = torch.randn(2, 3000)
    _, details = bank(x, detail=True)
    changed = x.clone(); changed[:,1000:] += 50
    _, details2 = bank(changed, detail=True)
    # Windows ending before 1000 cannot change due to remote samples.
    assert torch.allclose(details["scores"][:,:19], details2["scores"][:,:19], atol=1e-6)
    assert torch.allclose(model.encode(x[:,None]), torch.cat([model.encode(v[None,None]) for v in x]), atol=2e-6)
    saved = bank.waveforms.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = model.encode(x[:,None])
    scores = model.emissions(features[None])
    loss = model.loss(scores, torch.tensor([[1,2]]), torch.ones(1,2,dtype=torch.bool))
    loss.backward(); optimizer.step()
    assert torch.equal(bank.waveforms, saved)
    assert bank.waveforms.grad is None
    assert bank.encoder.net[0].weight.grad.abs().sum() > 0


def test_similarity_bounds_identical_window_and_affine_invariance():
    torch.manual_seed(7)
    wave = torch.randn(2, 100)
    bank = WaveformBank(wave, 8, 50, 0.5)
    x = torch.randn(1, 3000); x[0,:100] = wave[0]
    features, detail = bank(x, detail=True)
    assert abs(float(detail["scores"][0,0,0].detach())-1) < 1e-5
    assert torch.all((detail["scores"] >= 0) & (detail["scores"] <= 1+1e-6))
    assert torch.allclose(features, bank(x*2 + 4), atol=1e-5)
    flat = bank(torch.zeros(1,3000))
    assert flat.eq(0).all() and torch.isfinite(flat).all()
    summary = epoch_summary(torch.zeros(1,3000))
    assert summary.shape == (1,16) and torch.isfinite(summary).all()


def test_invalid_shapes_config_and_flat_anchor():
    with pytest.raises(ValueError): ModelConfig(fs=128)
    with pytest.raises(ValueError): ModelConfig(neural_fraction_cap=0.9)
    with pytest.raises(ValueError): WaveformBank(torch.zeros(2,100), 8, 50, 0.5)
    with pytest.raises(ValueError): small_model().encode(torch.rand(1,2,3000))


def test_ablation_shapes_and_finite_backward():
    torch.manual_seed(9)
    base = ModelConfig(scales=(100,), prototypes_per_scale=2, embedding_dim=8, radius=0)
    for update in ({}, {"summary_only":True}, {"use_summary":False}, {"neural_fraction_cap":0.0},
                   {"use_crf":False}, {"raw_control":True}):
        cfg = replace(base, **update)
        model = EvidenceModel(cfg, [torch.randn(2,100)])
        x = torch.randn(3,1,3000)
        z = model.encode(x)
        assert z.shape == (3, len(model.feature_names))
        loss = model.loss(model.emissions(z[None]), torch.tensor([[0,1,2]]), torch.ones(1,3,dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_forward_backward_smoke():
    model = small_model().cuda()
    x = torch.randn(4,1,3000,device="cuda")
    loss = model.loss(model.emissions(model.encode(x)[None]), torch.tensor([[0,1,2,3]],device="cuda"),
                      torch.ones(1,4,dtype=torch.bool,device="cuda"))
    loss.backward()
    assert torch.isfinite(loss)


def test_flat_input_gradients_and_crf_low_precision_input():
    x = torch.zeros(2,100,requires_grad=True)
    z,valid = normalize_wave(x)
    z.sum().backward()
    assert torch.isfinite(x.grad).all() and not valid.any()
    crf = LinearCRF()
    emissions = torch.randn(1,4,5,dtype=torch.bfloat16,requires_grad=True)
    labels = torch.tensor([[0,1,2,3]])
    loss = crf.loss(emissions,labels,torch.ones(1,4,dtype=torch.bool))
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(emissions.grad).all()


def test_observed_components_have_mandatory_lower_bound():
    torch.manual_seed(3)
    bank = WaveformBank(torch.randn(3,100),8,50,0.5)
    bank.observed_logits.data = torch.tensor([[20.,-20.,0.],[-20.,20.,0.],[0.,-20.,20.]])
    bank.neural_logits.data.fill_(20.)
    _, details = bank(torch.randn(2,3000),detail=True)
    weights = details["observed_weights"]
    assert torch.all(weights >= 0.1)
    assert torch.allclose(weights.sum(-1),torch.ones(3),atol=1e-6)
    distance = -details["scores"].log()/details["inverse_temperature"]
    assert torch.all(distance + 1e-6 >= 0.05*details["distance_components"].max(-1).values)

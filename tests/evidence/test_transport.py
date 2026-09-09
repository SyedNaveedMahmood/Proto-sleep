from dataclasses import replace

import torch

from mist_evidence.model import AdditiveTemporal, ModelConfig
from mist_evidence.transport import (
    TransportEvidenceModel,
    TransportModelConfig,
    semi_unbalanced_transport,
)


def test_transport_exact_source_marginal_and_finite_gradient():
    torch.manual_seed(3)
    cost = torch.rand(2, 7, 4, requires_grad=True)
    valid = torch.ones(2, 7, dtype=torch.bool)
    plan = semi_unbalanced_transport(
        cost,
        valid,
        epsilon=0.08,
        target_rho=0.20,
        iterations=40,
        background_cost=0.40,
        background_prior=0.35,
    )
    assert plan.shape == (2, 7, 5)
    assert torch.isfinite(plan).all() and (plan >= 0).all()
    expected = torch.full((2, 7), 1.0 / 7.0)
    assert torch.allclose(plan.sum(-1), expected, atol=2e-6, rtol=2e-6)
    loss = (plan[..., :-1] * torch.arange(1, 5, dtype=plan.dtype)).sum()
    loss.backward()
    assert cost.grad is not None and torch.isfinite(cost.grad).all()


def test_transport_uses_background_for_flat_or_high_cost_windows():
    cost = torch.full((1, 6, 3), 1.2)
    valid = torch.ones(1, 6, dtype=torch.bool)
    plan = semi_unbalanced_transport(
        cost,
        valid,
        epsilon=0.08,
        target_rho=0.08,
        iterations=50,
        background_cost=0.10,
        background_prior=0.35,
    )
    background = plan[..., -1].sum()
    assert background > plan[..., 0].sum()

    valid[:, :2] = False
    plan2 = semi_unbalanced_transport(
        cost,
        valid,
        epsilon=0.08,
        target_rho=0.08,
        iterations=50,
        background_cost=0.10,
        background_prior=0.35,
    )
    assert torch.all(plan2[0, :2, -1] > plan2[0, :2, :-1].max(-1).values)


def small_transport_model(radius=0):
    cfg = TransportModelConfig(
        scales=(100,),
        prototypes_per_scale=2,
        embedding_dim=8,
        radius=radius,
        use_crf=False,
        transport_iterations=20,
    )
    anchors = [torch.randn(2, 100)]
    return TransportEvidenceModel(cfg, anchors, amplitude_stats=(0.0, 1.0))


def test_transport_model_named_features_backward_and_anchor_immutability():
    torch.manual_seed(4)
    model = small_transport_model()
    # 3*K transport variables + one background variable + 16 summaries + amplitude.
    assert len(model.feature_names) == 3 * 2 + 1 + 16 + 1
    assert model.feature_names[:2] == ["ot_s100_p00_mass", "ot_s100_p01_mass"]
    assert model.feature_names[6] == "ot_s100_background_mass"

    x = torch.randn(3, 1, 3000)
    features, details = model.encode_with_details(x)
    assert features.shape == (3, len(model.feature_names))
    assert torch.isfinite(features).all()
    assert len(details) == 1
    expected = torch.full((3, details[0]["transport_plan"].shape[1]),
                          1.0 / details[0]["transport_plan"].shape[1])
    assert torch.allclose(details[0]["transport_row_mass"], expected, atol=2e-5, rtol=2e-5)

    before = model.banks[0].waveforms.clone()
    emissions = model.emissions(features[None])
    labels = torch.tensor([[0, 1, 2]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    loss = model.loss(emissions, labels, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.banks[0].encoder.net[0].weight.grad is not None
    assert torch.isfinite(model.banks[0].encoder.net[0].weight.grad).all()
    assert torch.equal(before, model.banks[0].waveforms)
    assert model.banks[0].waveforms.grad is None


def test_transport_additive_score_accounting():
    torch.manual_seed(8)
    model = small_transport_model(radius=2).eval()
    x = torch.randn(9, 1, 3000)
    features = model.encode(x)
    emissions = model.emissions(features[None])[0]
    assert isinstance(model.temporal, AdditiveTemporal)
    for t in (0, 4, 8):
        parts = model.temporal.explain_at(features, t)
        reconstructed = parts.sum((1, 2)) + model.temporal.bias
        assert torch.allclose(reconstructed, emissions[t], atol=1e-4, rtol=2e-5)


def test_transport_config_validation_and_dictionary_contains_mechanism():
    cfg = TransportModelConfig(scales=(100,), prototypes_per_scale=2, embedding_dim=8, radius=0, use_crf=False)
    d = cfg.dictionary()
    assert d["transport_epsilon"] == cfg.transport_epsilon
    assert d["transport_background_prior"] == cfg.transport_background_prior
    try:
        replace(cfg, transport_iterations=2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid transport_iterations accepted")

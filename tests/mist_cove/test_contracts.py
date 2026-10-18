"""Synthetic contracts, signed counterfactuals, and fail-closed Phase I access."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mist_cove import CoveConfig, CoveModel, SyntheticTrace, interval_ledger, masked_loss
from mist_cove.audit import audit_interval, validate_replacement
from mist_cove.experiment import evaluate, fit, risk_coverage, stress_probes, validate_config
from mist_cove.model import MODES
from mist_cove.synthetic import make_batch, perturbations

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(19)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/mist_cove_synthetic_v1.json").read_text())


@pytest.mark.parametrize("mode", sorted(MODES))
def test_five_class_finite_signed_reconciliation(mode):
    model = CoveModel().eval()
    e = model(make_batch(5, 7).x, mode=mode, return_evidence=True)
    assert e.logits.shape == (5, 5)
    assert torch.isfinite(e.logits).all()
    torch.testing.assert_close(e.logits.softmax(-1).sum(-1), torch.ones(5))
    torch.testing.assert_close(e.logits, e.background_logits + e.contributions.sum(1), rtol=0, atol=0)
    torch.testing.assert_close(e.contributions, e.availability * e.signed_evidence, rtol=0, atol=0)
    torch.testing.assert_close(e.background_logits, e.background_contributions.sum(1) + e.background_bias, rtol=0, atol=0)
    assert ((e.availability >= 0) & (e.availability <= 1)).all()
    if mode in {"full", "q_one", "fixed_patch"}:
        assert (e.signed_evidence < 0).any() and (e.signed_evidence > 0).any()
    else:
        assert not e.signed_evidence.any()
    if mode == "q_only":
        assert not e.background_logits.any()


def test_eventless_path_has_no_fabricated_intervals():
    model = CoveModel().eval()
    for value in (0., 4.):
        x = torch.full((2, 1, 3000), value)
        e = model(x, return_evidence=True)
        assert not e.valid.any()
        assert not e.contributions.any() and not e.availability.any()
        assert not e.event_intervals.any()
        assert interval_ledger(e, [SyntheticTrace.from_wave(x[i], recording_id="a", night="1", original_epoch_index=i)
                                   for i in range(2)]) == []


def test_boundary_windows_and_nonoverlap():
    x = torch.zeros(1, 1, 3000)
    wave = torch.sin(2 * torch.pi * 10 * torch.arange(200) / 100)
    x[0, 0, :200] = 20 * wave
    x[0, 0, 2800:] = 30 * wave
    e = CoveModel().eval()(x, return_evidence=True)
    intervals = e.event_intervals[0, e.valid[0]].tolist()
    assert [0, 200] in intervals and [2800, 3000] in intervals
    for i, (a, b) in enumerate(intervals):
        assert 0 <= a < b <= 3000
        assert all(b <= c or a >= d for c, d in intervals[i + 1:])


def test_offsets_hashes_nights_and_unscored_gaps():
    batch = make_batch(3, 1)
    model = CoveModel().eval()
    e = model(batch.x, return_evidence=True)
    traces = [SyntheticTrace.from_wave(batch.x[i], recording_id="night-a" if i < 2 else "night-b",
                                      night="1" if i < 2 else "2", original_epoch_index=[2, 9, 0][i],
                                      scoring_mask=i != 1) for i in range(3)]
    rows = interval_ledger(e, traces)
    assert len(rows) == 12 and any(not r["scoring_mask"] for r in rows)
    for row in rows:
        assert row["absolute_start_sample"] == row["original_epoch_index"] * 3000 + row["start_sample"]
        assert row["absolute_end_sample"] - row["absolute_start_sample"] == 200
    # A batched epoch never uses its neighbor, gap, scoring status, or night.
    single = torch.cat([model(batch.x[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(single, e.logits, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="one trace"):
        interval_ledger(e, traces[:1])
    traces[0] = SyntheticTrace("changed", "1", 2, "a" * 64)
    with pytest.raises(ValueError, match="does not match"):
        interval_ledger(e, traces)


@pytest.mark.parametrize("role", ["train", "val", "test", "reserved", "ST", "SHHS"])
def test_phase_one_trace_rejects_real_roles(role):
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        with pytest.raises(ValueError, match="synthetic traces only"):
            SyntheticTrace("unused", "1", 0, "a" * 64, role=role)


@pytest.mark.parametrize("role", ["test", "reserved", "ST", "SHHS"])
def test_existing_reserved_loader_rejects_before_access(role):
    from mist_transfer.protocol import load_development
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        with pytest.raises(ValueError, match="prohibited"):
            load_development({}, role)


def test_masks_never_compress_predictions_or_supervise_unscored():
    logits = CoveModel()(make_batch(3, 2).x)
    labels = torch.tensor([1, -1, 3]); mask = torch.tensor([True, False, True])
    loss = masked_loss(logits, labels, mask)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits[[0, 2]], labels[[0, 2]]))
    assert logits.shape == (3, 5)
    zero = masked_loss(logits, labels, torch.zeros(3, dtype=torch.bool))
    assert zero == 0 and zero.requires_grad
    with pytest.raises(ValueError, match="invalid scored"):
        masked_loss(logits, labels, torch.ones(3, dtype=torch.bool))


def test_physical_calibration_inverse_and_state_roundtrip():
    physical = make_batch(5, 3).x
    torch.manual_seed(12); reference = CoveModel().eval()
    scaled = CoveModel(mean_uV=4.5, std_uV=12.3).eval()
    # Copy heads, preserving the two distinct calibration buffers.
    scaled.background_head.load_state_dict(reference.background_head.state_dict())
    scaled.event_head.load_state_dict(reference.event_head.state_dict())
    normalized = (physical - 4.5) / 12.3
    torch.testing.assert_close(scaled.physical_uV(normalized), physical, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reference(physical), scaled(normalized), rtol=1e-5, atol=1e-5)
    clone = CoveModel(); clone.load_state_dict(scaled.state_dict())
    torch.testing.assert_close(clone(normalized), scaled(normalized), rtol=0, atol=0)


@pytest.mark.parametrize("x", [torch.zeros(1, 3000), torch.zeros(1, 2, 3000), torch.zeros(1, 1, 2999),
                               torch.zeros(0, 1, 3000), torch.zeros(1, 1, 3000).double(),
                               torch.full((1, 1, 3000), float("nan"))])
def test_invalid_inputs_fail_closed(x):
    with pytest.raises(ValueError, match="FP32"):
        CoveModel()(x)


@pytest.mark.parametrize("kwargs", [{"samples": 2999}, {"fs": 128}, {"event_budget": 9},
                                    {"stride": 0}, {"window": 600}, {"noise_scale_uV": 0},
                                    {"concentration_floor": 1}, {"event_scale": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CoveConfig(**kwargs)


def test_availability_not_a_learned_label_bypass_and_gain_is_physical():
    model = CoveModel().eval(); batch = make_batch(5, 5)
    original = model(batch.x, return_evidence=True)
    with torch.no_grad():
        model.event_head.weight.mul_(100)
        model.background_head.weight.mul_(100)
    changed_heads = model(batch.x, return_evidence=True)
    assert torch.equal(original.availability, changed_heads.availability)
    assert torch.equal(original.event_intervals, changed_heads.event_intervals)
    torch.testing.assert_close(original.availability[..., :1].expand_as(original.availability), original.availability)
    # Arbitrary physical gain is not incorrectly declared invariant.
    amplified = model(batch.x * .01, return_evidence=True)
    assert amplified.availability.max() < original.availability.max()


def test_perturbations_separate_removal_from_equal_distortion_and_calibration():
    batch = make_batch(10, 9); variants = perturbations(batch)
    distortion = lambda name: (variants[name] - batch.x).flatten(1).norm(dim=1)
    torch.testing.assert_close(distortion("lose_a"), distortion("matched_corruption"), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(variants["calibrated_gain"], batch.x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(variants["both_lost"], batch.background + batch.nuisance, rtol=1e-5, atol=1e-5)
    model = CoveModel().eval()
    clean = model(batch.x, return_evidence=True)
    erased = model(variants["both_lost"], return_evidence=True)
    assert clean.availability.mean() > erased.availability.mean()


def test_deterministic_seeds_and_finite_optimization(config):
    config.update(train_size=15, optimizer_steps=5)
    a, first = fit(config, 123, "full")
    b, second = fit(config, 123, "full")
    c, _ = fit(config, 456, "full")
    x = make_batch(5, 67).x
    assert first == second
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), c(x))
    assert first["final_loss"] < first["initial_loss"]


def audit_signal():
    x = torch.zeros(1, 1, 3000)
    x[0, 0, 600:800] = 30 * torch.hann_window(200, periodic=False) * torch.sin(2 * torch.pi * 10 * torch.arange(200) / 100)
    model = CoveModel().eval()
    with torch.no_grad():
        model.background_head.weight.zero_(); model.background_head.bias.zero_()
        model.event_head.weight.zero_(); model.event_head.bias.zero_()
        model.event_head.weight[0, 2] = 1.
        model.event_head.weight[1, 2] = -1.
    return model, x


@pytest.mark.parametrize("strategy", ["linear_bridge", "same_epoch_background"])
def test_signed_counterfactuals_recompute_model_and_match_accounting(strategy):
    model, x = audit_signal()
    kwargs = {"donor_interval": (1000, 1200)} if strategy == "same_epoch_background" else {}
    audit = audit_interval(model, x, (600, 800), strategy=strategy, **kwargs)
    assert audit.claimed_contribution[0] > 0 and audit.logit_delta[0] > 0
    assert audit.claimed_contribution[1] < 0 and audit.logit_delta[1] < 0
    torch.testing.assert_close(audit.logit_delta, audit.background_delta + audit.event_delta, rtol=1e-5, atol=1e-6)
    assert audit.diagnostics["endpoint_change_uV"] == 0
    assert audit.diagnostics["physiological_validity"] == "NOT_ESTABLISHED"
    control = audit_interval(model, x, (1400, 1600), strategy=strategy, **kwargs)
    assert audit.logit_delta[0] > control.logit_delta[0] + .1


@pytest.mark.parametrize("donor", [(600, 800), (-1, 199), (2900, 3100), (1000, 1100)])
def test_invalid_donor_geometry(donor):
    model, x = audit_signal()
    with pytest.raises(ValueError, match="donor"):
        audit_interval(model, x, (600, 800), strategy="same_epoch_background", donor_interval=donor)


def test_donor_rejects_energy_and_nonbackground_and_training_mode():
    model, x = audit_signal()
    x[:, :, 1000:1200] = x[:, :, 600:800] * 100
    with pytest.raises(ValueError, match="background compatible"):
        audit_interval(model, x, (600, 800), strategy="same_epoch_background", donor_interval=(1000, 1200))
    for bad in (torch.full((200,), 1e8), torch.full((200,), float("nan"))):
        with pytest.raises(ValueError):
            validate_replacement(torch.ones(200), bad)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        audit_interval(model, x, (600, 800))


def test_duplicate_witnesses_are_bounded_and_never_double_count_samples():
    model, x = audit_signal()
    single = model(x, return_evidence=True)
    x[:, :, 1600:1800] = x[:, :, 600:800]
    repeated = model(x, return_evidence=True)
    assert repeated.contributions[0, :, 0].sum() > single.contributions[0, :, 0].sum()
    assert repeated.contributions[0, :, 0].sum() <= model.cfg.event_scale
    # Repetition can still boost logits: bounded is not an independence proof.
    assert repeated.valid[0].sum() == 2


@pytest.mark.parametrize("flag", ["real_data_access", "reserved_access", "external_access"])
def test_config_cannot_authorize_real_data(config, flag):
    config[flag] = True
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        with pytest.raises(ValueError, match="synthetic-only"):
            validate_config(config)


def test_valid_config_validation_is_metadata_only(config):
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        validate_config(config)


def test_risk_coverage_reports_class_retention_and_full_coverage():
    p = torch.tensor([[.8, .05, .05, .05, .05], [.05, .8, .05, .05, .05]])
    curve = risk_coverage(p, torch.tensor([0, 0]), torch.tensor([1., 0.]), [1., .5])
    assert curve[0]["risk"] == .5 and curve[0]["coverage"] == 1
    assert curve[1]["risk"] == 0 and curve[1]["class_coverage"] == [.5, None, None, None, None]


def test_experiment_reports_negative_controls_and_class_collapse(config):
    config.update(train_size=15, evaluation_size=15, optimizer_steps=2)
    model, _ = fit(config, 123, "q_only")
    scores, curves, switch, gain = evaluate(model, make_batch(15, 500), config, "q_only")
    assert scores["clean"]["accuracy"] <= .6
    assert set(curves) == {"availability", "negative_entropy", "global_rms"}
    assert set(switch) == {"clean", "lose_a", "lose_b", "both_lost"}
    assert all("prediction_counts" in row and "collapsed" in row for row in scores.values())


def test_exploratory_probes_report_redundancy_and_matched_random_controls(config):
    config.update(train_size=15, optimizer_steps=10)
    model, _ = fit(config, 123, "full")
    probe = stress_probes(model, make_batch(15, 2123), 123)
    assert len(probe["random_matched_control_comparisons"]) == 10
    redundant = probe["redundant_similar_nonselected"]
    assert redundant["selection_changed"]
    assert redundant["claimed_selected_contribution"] > redundant["selected_replacement_delta"]

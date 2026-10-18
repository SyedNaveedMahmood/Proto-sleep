"""Phase II arithmetic, intervention semantics, known-component and trace tests."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mist_cove.model import CoveConfig, CoveModel
from mist_cove.phase2.model import make_model, WitnessSet
from mist_cove.phase2.audit import audit_group, replace_group, union_intervals, jaccard_distance
from mist_cove.phase2.experiment import fit, audit_case, stability, validate_config
from mist_cove.phase2.synthetic import make_fixtures, transformed
from mist_cove.phase2.provenance import ROOT, verify_preregistration
from mist_cove.phase2.traces import EpochIdentity, source_ledger
from mist_cove.phase2.report import local_evidence_report


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2); torch.manual_seed(17)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/mist_cove_phase2_synthetic_v1.json").read_text())


@pytest.mark.parametrize("method,mode", [("q1", "q_one"), ("availability", "full"), ("background", "background_only"), ("fixed_patch", "fixed_patch")])
def test_phase1_decision_functions_are_preserved_exactly(config, method, mode):
    x = torch.cat([r.x for r in make_fixtures(5, 21, "substitutable", "dual", config["generator"])])
    original = CoveModel().eval(); current = make_model(method).eval()
    current.load_state_dict(original.state_dict())
    old = original(x, mode=mode, return_evidence=True); new = current(x, return_evidence=True)
    for field in ("logits", "background_logits", "signed_evidence", "availability", "contributions", "event_intervals", "valid"):
        torch.testing.assert_close(getattr(old, field), getattr(new, field), rtol=0, atol=0)


@pytest.mark.parametrize("method", ["q1", "availability", "background", "fixed_patch", "rounded_q1"])
def test_additive_accounting_finiteness_masks_and_determinism(config, method):
    model = make_model(method).eval()
    x = torch.cat([r.x for r in make_fixtures(5, 3, "complementary", "dual", config["generator"])])
    e = model(x, return_evidence=True)
    assert sum(p.numel() for p in model.parameters()) == 75
    torch.testing.assert_close(e.logits, model(x), atol=0, rtol=0)
    torch.testing.assert_close(e.logits, e.background_logits + e.contributions.sum(1), atol=0, rtol=0)
    torch.testing.assert_close(e.contributions, e.availability * e.signed_evidence, atol=0, rtol=0)
    torch.testing.assert_close(e.logits.softmax(-1).sum(-1), torch.ones(5))
    model(x).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if method in {"q1", "rounded_q1"}:
        assert (e.availability[e.valid] == 1).all()


@pytest.mark.parametrize("method", ["q1", "availability", "rounded_q1"])
def test_fixed_ids_are_internal_interventions_and_eventless(config, method):
    model = make_model(method).eval()
    x = make_fixtures(1, 8, "substitutable", "single", config["generator"])[0].x
    original = model(x, return_evidence=True)
    flat = torch.zeros_like(x)
    dynamic = model(flat, return_evidence=True)
    fixed = model(flat, return_evidence=True, witnesses=WitnessSet.from_evidence(original))
    assert not dynamic.valid.any() and not dynamic.contributions.any()
    assert torch.equal(original.valid, fixed.valid)
    assert torch.equal(original.event_intervals, fixed.event_intervals)
    assert fixed.acquisition_observability_metadata["fixed_witness_internal_intervention"]
    assert not dynamic.event_intervals.any()


@pytest.mark.parametrize("bad", ["overlap", "off_grid", "length", "negative", "dtype", "valid_shape"])
def test_invalid_fixed_identities_rejected(config, bad):
    m = make_model("q1").eval()
    x = torch.randn(1, 1, 3000)
    e = m(x, return_evidence=True)
    intervals, valid = e.event_intervals.clone(), e.valid.clone()
    if bad == "overlap": intervals[0, 1] = intervals[0, 0]
    if bad == "off_grid": intervals[0, 0] += 1
    if bad == "length": intervals[0, 0, 1] -= 1
    if bad == "negative": intervals[0, 0] -= 3000
    if bad == "dtype": intervals = intervals.float()
    if bad == "valid_shape": valid = valid[:, :1]
    with pytest.raises(ValueError): m(x, witnesses=WitnessSet(intervals, valid))


@pytest.mark.parametrize("case", ["single", "dual", "redundant", "correlated", "distractor", "competition", "overlap"])
@pytest.mark.parametrize("strategy", ["bridge", "background", "noise", "phase_scramble"])
def test_counterfactual_components_reconcile_all_stages(config, case, strategy):
    m = make_model("q1").eval()
    row = make_fixtures(1, 47, "substitutable", case, config["generator"])[0]
    kwargs = {"donor_interval": row.donor_interval} if strategy == "background" else {}
    audit = audit_group(m, row.x, row.diagnostic_intervals, strategy=strategy, **kwargs)
    torch.testing.assert_close(audit.full_effect, audit.background_effect + audit.fixed_event_path_effect + audit.reselection_effect, atol=1e-5, rtol=0)
    torch.testing.assert_close(audit.fixed_witness_effect, audit.background_effect + audit.fixed_event_path_effect, atol=1e-5, rtol=0)
    torch.testing.assert_close(audit.additive_contribution, audit.score_deletion_effect, atol=1e-5, rtol=0)
    assert not audit.diagnostics["internal_paths_are_physiological_counterfactuals"]
    assert all(d["physiological_validity"] == "NOT_ESTABLISHED" for d in audit.diagnostics["replacements"])


def test_joint_replacement_is_order_independent_union_not_sequential(config):
    m = make_model("q1").eval(); row = make_fixtures(1, 27, "substitutable", "overlap", config["generator"])[0]
    a = audit_group(m, row.x, row.diagnostic_intervals, strategy="noise", seed=7)
    b = audit_group(m, row.x, list(reversed(row.diagnostic_intervals)), strategy="noise", seed=7)
    assert a.intervals == b.intervals and len(a.intervals) == 1
    torch.testing.assert_close(a.full_effect, b.full_effect, atol=0, rtol=0)
    assert union_intervals([(0, 200), (100, 300), (0, 200)]) == [(0, 300)]


@pytest.mark.parametrize("intervals", [[], [(-1, 200)], [(1, 3001)], [(200, 100)], [(0, 10)]])
def test_bad_group_intervals_fail_closed(intervals):
    with pytest.raises(ValueError): union_intervals(intervals)


def test_donors_cannot_reuse_group_or_accept_unknown_recipe(config):
    m = make_model("q1").eval(); row = make_fixtures(1, 27, "substitutable", "dual", config["generator"])[0]
    with pytest.raises(ValueError, match="disjoint"):
        audit_group(m, row.x, row.diagnostic_intervals, strategy="background", donor_interval=row.diagnostic_intervals[0])
    with pytest.raises(ValueError, match="unregistered"):
        audit_group(m, row.x, row.diagnostic_intervals, strategy="arbitrary")
    x = row.x.clone(); lo, hi = row.donor_interval; x[..., lo:hi] *= 100
    with pytest.raises(ValueError, match="compatibility"):
        audit_group(m, x, row.diagnostic_intervals, strategy="background", donor_interval=row.donor_interval)


def test_generator_seed_identity_and_complementary_information(config):
    a = make_fixtures(5, 45, "complementary", "dual", config["generator"])
    b = make_fixtures(5, 45, "complementary", "dual", config["generator"])
    c = make_fixtures(5, 46, "complementary", "dual", config["generator"])
    assert all(torch.equal(x.x, y.x) for x, y in zip(a, b))
    assert not torch.equal(a[0].x, c[0].x)
    assert [row.label for row in a] == list(range(5))
    assert all(len(row.diagnostic_intervals) == 2 for row in a)


def test_experiment_reports_all_estimands_and_negative_control(config):
    # Exploratory software fixture, not the frozen confirmatory run.
    config.update(train_size=15, optimizer_steps=3, audit_examples_per_case=1)
    m, training = fit(config, 123, "substitutable", "q1")
    rows = make_fixtures(1, 39, "substitutable", "correlated", config["generator"])
    result = audit_case(m, rows, config, 123)
    assert result["max_reconciliation_error"] < 1e-5
    assert len(result["signed_stage_vector_means"]["full_effect"]) == 5
    assert "phase_preserving_negative_control_delta_mean" in result
    assert "individual_effect_abs_mean" in result and "group_effect_abs_mean" in result
    assert training["optimizer_steps"] == 3


def test_calibration_stability_versus_physical_gain_and_trace_roles(config):
    model = make_model("q1").eval(); row = make_fixtures(1, 35, "substitutable", "single", config["generator"])[0]
    e = model(row.x, return_evidence=True)
    identity = EpochIdentity("synthetic_person", "night_a", "1", "synthetic", 9, "a" * 64, e.physical_input_sha256[0], False)
    ledger = source_ledger(e, [identity])
    assert ledger and all(r["subject_id"] == "synthetic_person" and r["units"] == "uV" for r in ledger)
    assert all(r["absolute_start_sample"] == 27000 + r["start_sample"] and not r["scoring_mask"] for r in ledger)
    torch.testing.assert_close(model(row.x), model(transformed(row.x, "calibrated_gain", config["stability"], 3)), atol=1e-5, rtol=0)
    with patch.object(Path, "open", side_effect=AssertionError("data access")):
        with pytest.raises(ValueError, match="prohibited"):
            EpochIdentity("x", "y", "1", "test", 0, "a" * 64, "b" * 64, True)


def test_preregistration_and_config_validation_are_metadata_only(config):
    proof = verify_preregistration()
    assert proof["before_phase2_experiments"] and not proof["real_data_authorized"]
    with patch.object(Path, "open", side_effect=AssertionError("file access")):
        validate_config(config)
        config["real_data_access"] = True
        with pytest.raises(ValueError, match="prohibited"): validate_config(config)


def test_corrected_complementary_single_factor_is_symmetric_and_versions_match():
    from mist_cove.model import spectral_features
    c = json.loads((ROOT / "configs/mist_cove_phase2_synthetic_v2.json").read_text())
    validate_config(c); verify_preregistration(2)
    rows = make_fixtures(100, 51, "complementary", "single", c["generator"])
    retained_a, retained_b = 0, 0
    for row in rows:
        lo, hi = row.diagnostic_intervals[0]
        bands, _, _ = spectral_features(row.x[0, 0, lo:hi][None])
        band = int(bands.argmax())
        assert band in {row.label, (row.label + 1) % 5}
        retained_a += band == row.label
        retained_b += band == (row.label + 1) % 5
    assert retained_a > 20 and retained_b > 20
    c["generator"]["version"] = "known-components-phase2-v1"
    with pytest.raises(ValueError, match="versions disagree"): validate_config(c)


def test_local_report_exposes_redundancy_and_separates_all_four_claims():
    m = make_model("q1").eval()
    with torch.no_grad():
        m.background_head.weight.zero_(); m.background_head.bias.zero_()
        m.event_head.weight.zero_(); m.event_head.bias.zero_()
        m.event_head.weight[0, 2] = 1.; m.event_head.weight[1, 2] = -1.
    x = torch.zeros(1, 1, 3000)
    waveform = 30 * torch.hann_window(200, periodic=False) * torch.sin(2 * torch.pi * 10 * torch.arange(200) / 100)
    intervals = [(start, start + 200) for start in (200, 600, 1000, 1400, 1800, 2200)]
    for lo, hi in intervals: x[0, 0, lo:hi] = waveform
    report = local_evidence_report(m, x, {"known_redundant_group": intervals}, donor_specs=[{"strategy": "bridge"}],
                                   transformations={"identical_reordering": x.clone()})
    group = report["groups"]["known_redundant_group"]["donors"][0]
    assert len(group["A_additive_contribution"]) == 5
    assert len(group["B_individual_counterfactual_effects"]) == 6
    assert group["C_group_counterfactual_effect"][0] > max(v[0] for v in group["B_individual_counterfactual_effects"]) + 1
    assert group["flags"]["individual_necessity_not_established"]
    assert report["D_attribution_stability"]["identical_reordering"]["witness_set_jaccard_distance"] == 0

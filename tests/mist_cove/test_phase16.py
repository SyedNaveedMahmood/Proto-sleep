"""Generated-only Phase XVI contracts; no external EEG or annotation arrays."""

from decimal import Decimal as D
from pathlib import Path

import pytest

from mist_cove.phase16.annotations import (Coverage, parse_spindles,
    parse_kcomplexes, parse_rems, parse_artifacts)
from mist_cove.phase16.archive_security import Member, validate_members
from mist_cove.phase16.benchmark import (COVERED_NEGATIVE, POSITIVE, UNKNOWN,
    enforce_participant_split, joint_coverage, label_interval, match_events)
from mist_cove.phase16.source import ARCHIVES, RightsReceipt, canonical_url, download_authorized
from mist_cove.phase16.reference import reviewer_events, union_events, intersection_events
from mist_cove.phase16.spectral import atom_scores, top_atoms, selected_intervals
from mist_cove.phase16.edf_contract import ChannelHeader, validate_event_channel
from mist_cove.phase9.circuits import compile_robdd, evaluate_robdd
from mist_cove.phase9.clauses import verify_clause
import numpy as np


def coverage(kind="spindle", scorer="expert1", stop="1000", channel="C3-A1"):
    return Coverage("excerpt1", scorer, D(0), D(stop), channel, kind)


def events(kind="spindle", scorer="expert1", text=b"[spindles]\n2.0 1.0\n",
           stop="1000", channel="C3-A1"):
    parser = {"spindle": parse_spindles, "k_complex": parse_kcomplexes,
              "rapid_eye_movement": parse_rems, "artifact": parse_artifacts}[kind]
    return parser(text, recording="excerpt1", participant="person1", channel=channel,
                  scorer=scorer, sample_rate=100, duration_seconds=D(1800),
                  coverages=(coverage(kind, scorer, stop, channel),))


def test_owner_allowlist_and_rights_fail_closed(tmp_path):
    assert len(ARCHIVES) == 4
    assert canonical_url("DatabaseSpindles.rar").startswith("https://zenodo.org/records/2650142/files/")
    with pytest.raises(ValueError):
        canonical_url("DatabasePatients.rar")
    with pytest.raises(PermissionError):
        download_authorized("DatabaseSpindles.rar", tmp_path / "DatabaseSpindles.rar", RightsReceipt())
    with pytest.raises(PermissionError):
        RightsReceipt(True, True, True, True, True).require_acquisition()


@pytest.mark.parametrize("member", [
    Member("../out.edf", "file", 100, 100),
    Member("/out.edf", "file", 100, 100),
    Member("C:\\out.edf", "file", 100, 100),
    Member("x/link.edf", "symlink", 100, 100),
    Member("x/script.sh", "file", 100, 100),
    Member("x/bomb.edf", "file", 1_000_000, 1),
])
def test_archive_preflight_rejects_unsafe_member(member):
    with pytest.raises(ValueError):
        validate_members([member])


def test_archive_preflight_accepts_bounded_text_and_edf():
    receipt = validate_members([Member("excerpt1/signal.edf", "file", 1000, 500),
                                Member("excerpt1/Visual_scoring1.txt", "file", 50, 50)])
    assert receipt["members"] == 2
    assert "PREFLIGHT_ONLY" in receipt["status"]


def test_spindle_exact_coordinates_and_provenance():
    event = events(text=b"[spindles]\n2.005 0.505\n")[0]
    assert (event.start_seconds, event.end_seconds) == (D("2.005"), D("2.510"))
    assert (event.start_sample, event.end_sample) == (201, 251)
    assert event.source_sha256 and event.dataset == "DREAMS"


@pytest.mark.parametrize("kind,text", [
    ("k_complex", b"[kcomplex]\n2.0 0.8\n"),
    ("rapid_eye_movement", b"[rem]\n2.0 0.5\n"),
    ("artifact", b"[vis_Atf_mvtE/cz-A1]\n2.0 1.0\n"),
])
def test_four_documented_annotation_adapters_on_generated_text(kind, text):
    event = events(kind=kind, text=text, channel="cz-A1" if kind == "artifact" else "C3-A1")[0]
    assert event.event_type == kind
    assert event.original_category is None if kind != "artifact" else event.original_category.startswith("vis_")


@pytest.mark.parametrize("text", [
    b"[spindles]\n999.9 0.2\n", b"[spindles]\n2.0 -1\n",
    b"[spindles]\nNaN 1\n", b"[spindles]\n1799.9 1\n",
    b"[spindles]\n2.0\n", b"2.0 1.0\n",
])
def test_parser_rejects_outside_or_malformed(text):
    with pytest.raises(ValueError):
        events(text=text)


def test_unknown_is_not_negative_and_joint_coverage_is_exact():
    first, second = coverage(stop="1000"), coverage(scorer="expert2", stop="1800")
    joint = joint_coverage([first], [second])
    assert joint == [(D(0), D(1000))]
    e = events()[0]
    assert label_interval(D(2), D(3), [e], joint) == POSITIVE
    assert label_interval(D(5), D(6), [e], joint) == COVERED_NEGATIVE
    assert label_interval(D(1001), D(1002), [e], joint) == UNKNOWN
    assert label_interval(D(999), D(1001), [e], joint) == UNKNOWN


def test_max_cardinality_one_to_one_matching_and_empty():
    left = events(text=b"[spindles]\n2 1\n4 1\n")
    right = events(scorer="expert2", text=b"[spindles]\n2.1 1\n4.1 1\n")
    result = match_events(left, right)
    assert result["matched"] == 2 and result["f1"] == 1
    assert result["mean_iou"] < 1
    assert match_events([], [])["f1"] is None
    with pytest.raises(ValueError):
        match_events(left, left)


def test_participant_disjointness():
    enforce_participant_split({"one": "A", "two": "B"}, {"three": "C"})
    with pytest.raises(ValueError):
        enforce_participant_split({"one": "A"}, {"three": "A"})
    with pytest.raises(ValueError):
        enforce_participant_split({"one": ""}, {"three": "C"})


def test_reference_definitions_are_explicit_and_distinct():
    first = events(text=b"[spindles]\n2 1\n")
    second = events(scorer="expert2", text=b"[spindles]\n2.2 1\n")
    assert reviewer_events(first, "expert1")[0].definition == "reviewer:expert1"
    assert union_events(first, second)[0].end == D("3.2")
    overlap = intersection_events(first, second)
    assert len(overlap) == 1 and overlap[0].start == D("2.2")
    assert overlap[0].definition == "matched_intersection"
    with pytest.raises(ValueError):
        union_events(first, events(scorer="expert2", channel="Cz-A1"))


def test_generated_sigma_burst_ranks_by_fixed_duration():
    rate = 100
    rng = np.random.default_rng(123)
    wave = rng.normal(0, 0.01, 30 * rate)
    t = np.arange(2 * rate) / rate
    wave[4 * rate:6 * rate] += np.sin(2 * np.pi * 13 * t)
    scores = atom_scores(wave, rate)
    assert top_atoms(scores["sigma_power"], 1) == [2]
    assert selected_intervals([2]) == [(4, 6)]
    assert len(scores["relative_sigma"]) == 15


@pytest.mark.parametrize("formula", [
    lambda a, b, c: a or b,
    lambda a, b, c: a and b,
    lambda a, b, c: (a ^ b),
    lambda a, b, c: (a or b) and not c,
])
def test_existing_circuit_engine_on_generated_event_response_functions(formula):
    truth = np.array([formula(bool(i & 1), bool(i & 2), bool(i & 4))
                      for i in range(8)], dtype=bool)
    circuit = compile_robdd(truth)
    assert np.array_equal(evaluate_robdd(circuit), truth)
    if truth[7]:
        assert verify_clause(truth, 7, 0)


def test_generated_edf_header_channel_geometry_and_calibration():
    header = ChannelHeader("C3-A1", 100, "uV", -100, 100, -32768, 32767)
    assert validate_event_channel(header, event_type="spindle", expected_rate=100,
                                  duration_seconds=1800, sample_count=180000)["status"] == "HEADER_CONTRACT_ONLY"
    with pytest.raises(ValueError):
        validate_event_channel(ChannelHeader("Fpz-Cz", 100, "uV", -100, 100, -32768, 32767),
                               event_type="spindle", expected_rate=100,
                               duration_seconds=1800, sample_count=180000)
    with pytest.raises(ValueError):
        validate_event_channel(header, event_type="spindle", expected_rate=100,
                               duration_seconds=1800, sample_count=179999)

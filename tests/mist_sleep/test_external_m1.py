"""Generated-only M1 integrity, signal and statistical falsification contracts."""
import io
import json
from datetime import datetime

import numpy as np
import pyedflib
import pytest
import torch

from mist_sleep.core import MISTSleep
from mist_sleep.external import source_checkpoint
from mist_sleep.external.protocol import CHECKPOINTS, SOURCE_PROTOCOL_SHA256, specification, scientific_decision, file_sha256
from mist_sleep.external.hmc_manifest import parse_manifest, capacity_preflight, acquire
from mist_sleep.external.signal_contract import physical_epochs, calibrated_microvolts, channel_index
from mist_sleep.external.blind_inference import frozen_execution, predict, state_digest
from mist_sleep.external.prediction_seal import seal_predictions, begin_scoring
from mist_sleep.external.hmc_scoring import epoch_labels, read_scoring, csv_events
from mist_sleep.external.stage_metrics import stage_metrics
from mist_sleep.external.paired_statistics import paired_effect
from mist_sleep.external.representation_audit import summarize
from mist_sleep.external.publication import public_aggregate


def generated_manifest():
    ids = [f"SN{i:03}" for i in range(1, 155) if i not in (14, 64, 135)]
    records = "\n".join(f"recordings/{r}.edf" for r in ids)
    sums = "\n".join(f"{'a'*64} recordings/{r}{suffix}" for r in ids
                     for suffix in (".edf", "_sleepscoring.edf", "_sleepscoring.txt"))
    return records, sums


def test_exact_model_and_protocol_identities():
    assert len(CHECKPOINTS) == 4 and all(len(v) == 64 for v in CHECKPOINTS.values())
    protocol = specification()
    assert protocol["primary_channel"] == "C4/M1" and protocol["secondary_channel"] == "F4/M1"
    assert protocol["limits"]["optimizer_updates"] == 0
    assert protocol["stage_order"] == ["Wake", "N1", "N2", "N3", "REM"]


@pytest.mark.parametrize("defect", ["missing_record", "duplicate", "removed", "missing_score", "unsafe", "wrong_hash"])
def test_manifest_rejects_integrity_failures(defect):
    records, sums = generated_manifest()
    if defect == "missing_record": records = "\n".join(records.splitlines()[1:])
    if defect == "duplicate": records = records.replace("SN002", "SN001")
    if defect == "removed": records = records.replace("SN001", "SN014")
    if defect == "missing_score": sums = "\n".join(sums.splitlines()[1:])
    if defect == "unsafe": sums += f"\n{'a'*64} ../recordings/bad.edf"
    if defect == "wrong_hash": sums = sums.replace("a"*64, "z"*64)
    with pytest.raises(ValueError): parse_manifest(records, sums)


def test_manifest_complete_triplets():
    rows = parse_manifest(*generated_manifest())
    assert len(rows) == 151 and sum(len(r["files"]) for r in rows) == 453


def test_capacity_and_source_allowlist_fail_before_network(tmp_path, monkeypatch):
    from mist_sleep.external.protocol import digest
    manifest = parse_manifest(*generated_manifest())
    monkeypatch.setattr("mist_sleep.external.hmc_manifest.OFFICIAL_MANIFEST_SHA256", digest(manifest))
    monkeypatch.setattr("mist_sleep.external.hmc_manifest.shutil.disk_usage", lambda _: type("Disk", (), {"free": 7_100_000_000})())
    assert not capacity_preflight(tmp_path)["pass"]
    monkeypatch.setenv("HMC_DATA_ROOT", str(tmp_path))
    tmp_path.chmod(0o700)
    monkeypatch.setattr("mist_sleep.external.hmc_manifest.urlopen", lambda *a, **k: pytest.fail("no network on failed capacity"))
    with pytest.raises(OSError): acquire(tmp_path, manifest)
    with pytest.raises(ValueError): acquire(tmp_path, [], source_url="https://example.org/")
    with pytest.raises(ValueError): capacity_preflight(tmp_path, required_bytes=18_000_000_001)


@pytest.mark.parametrize("rate,unit", [(100, "uV"), (256, "unknown"), (255.9, "uV")])
def test_wrong_rate_and_units_rejected(rate, unit):
    with pytest.raises(ValueError): physical_epochs(np.zeros(7680), unit=unit, sample_rate=rate)


def test_signal_calibration_units_and_source_frozen_normalization():
    values = calibrated_microvolts(np.array([-10, 0, 10]), -1, 1, -10, 10, "mV")
    np.testing.assert_allclose(values, [-1000, 0, 1000])
    with pytest.raises(ValueError): calibrated_microvolts([0], 0, 0, -10, 10, "uV")
    from mist_sleep.external.protocol import SOURCE_MEAN_UV, SOURCE_STD_UV
    epochs, onsets = physical_epochs(np.ones(7680 * 2 + 19) * 15, unit="uV", sample_rate=256)
    assert epochs.shape == (2, 1, 3000) and onsets.tolist() == [0, 30]
    assert abs(float(epochs[0, 0, 100:-100].mean()) - (15-SOURCE_MEAN_UV)/SOURCE_STD_UV) < 1e-5
    assert abs(float(epochs.mean())) > .5  # Per-record normalization would erase this mean.
    other, _ = physical_epochs(np.ones(7680 * 2 + 19) * .015, unit="mV", sample_rate=256)
    np.testing.assert_allclose(epochs, other, atol=1e-6)


@pytest.mark.parametrize("frequency", [.75, 13.])
def test_polyphase_preserves_inband_signal(frequency):
    from mist_sleep.external.protocol import SOURCE_MEAN_UV, SOURCE_STD_UV
    t = np.arange(7680*2) / 256
    epochs, _ = physical_epochs(np.sin(2*np.pi*frequency*t), unit="uV", sample_rate=256)
    restored = epochs.reshape(-1) * SOURCE_STD_UV + SOURCE_MEAN_UV
    expected = np.sin(2*np.pi*frequency*np.arange(6000)/100)
    assert np.sqrt(np.mean((restored[100:-100]-expected[100:-100])**2)) < .002


def test_polyphase_antialias_and_timestamp_impulse():
    from mist_sleep.external.protocol import SOURCE_MEAN_UV, SOURCE_STD_UV
    x = np.sin(2*np.pi*80*np.arange(7680)/256)
    epochs, _ = physical_epochs(x, unit="uV", sample_rate=256)
    restored = epochs.ravel()*SOURCE_STD_UV + SOURCE_MEAN_UV
    assert np.sqrt(np.mean(restored[100:-100]**2)) < .002
    x[:] = 0; x[2560] = 100
    epochs, _ = physical_epochs(x, unit="uV", sample_rate=256)
    assert np.argmax(epochs.ravel()) == 1000


@pytest.mark.parametrize("labels,channel", [(["F4/M1"], "C4/M1"), (["C4/M1", "EEG C4/M1"], "C4/M1"), (["C4/M1"], "O2/M1")])
def test_channel_substitution_and_ambiguity_rejected(labels, channel):
    with pytest.raises(ValueError): channel_index(labels, channel)


def test_literal_edf_aliases_preserve_electrode_order_and_reference():
    assert channel_index(["EEG F4-M1", "EEG C4-M1"], "C4/M1") == 1
    assert channel_index(["EEG F4-M1", "EEG C4-M1"], "F4/M1") == 0
    for labels in (["EEG M1-C4"], ["EEG C4-M2"], ["EEG C3-M2"]):
        with pytest.raises(ValueError): channel_index(labels, "C4/M1")


def test_edf_header_parser_generated_geometry_and_discontinuity(tmp_path):
    from mist_sleep.external.edf_header import parse_header
    path = tmp_path / "generated.edf"
    with pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders([dict(label="EEG C4-M1", dimension="uV", sample_frequency=256,
                                     physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767,
                                     transducer="generated", prefilter="HP:0.2Hz LP:35Hz")])
        writer.writeSamples([np.zeros(7680)])
    data = path.read_bytes()
    size = int(data[184:192])
    header = parse_header(data[:size])
    assert header["duration_seconds"] == 30 and header["signals"][0]["sample_rate"] == 256
    assert "patient" not in header and "start_datetime" not in header
    with pytest.raises(ValueError): parse_header(data[:size-1])
    changed = bytearray(data[:size]); changed[192:197] = b"EDF+D"
    with pytest.raises(ValueError): parse_header(bytes(changed))


def test_generated_edf_blind_reader(tmp_path, monkeypatch):
    from mist_sleep.external.hmc_loader import load_waveform
    path = tmp_path / "recordings" / "SN001.edf"; path.parent.mkdir()
    with pyedflib.EdfWriter(str(path), 2, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        headers = [dict(label=label, dimension="uV", sample_frequency=256,
                        physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767,
                        transducer="generated", prefilter="generated") for label in ("EEG C4/M1", "EEG F4/M1")]
        writer.setSignalHeaders(headers); writer.setStartdatetime(datetime(2020, 1, 1))
        writer.writeSamples([np.sin(np.arange(7680)/256*2*np.pi*13)*20, np.zeros(7680)])
        writer.writeAnnotation(0, 30, "Sleep stage N2")
    original = pyedflib.EdfReader
    class SpyReader(original):
        def __init__(self, *args, **kwargs):
            assert kwargs["annotations_mode"] == pyedflib.DO_NOT_READ_ANNOTATIONS
            super().__init__(*args, **kwargs)
        def readAnnotations(self):
            pytest.fail("blind waveform phase accessed labels")
    monkeypatch.setattr("mist_sleep.external.hmc_loader.pyedflib.EdfReader", SpyReader)
    wave, onsets, receipt = load_waveform(tmp_path, "recordings/SN001.edf", file_sha256(path), "C4/M1")
    assert wave.shape == (1, 1, 3000) and onsets.tolist() == [0] and receipt["complete_epochs"] == 1
    with pytest.raises(ValueError): load_waveform(tmp_path, "recordings/SN001_sleepscoring.edf", "a"*64, "C4/M1")
    # Identical normalized input reaches the unchanged architecture at full 3,000 samples.
    model = MISTSleep().eval().requires_grad_(False)
    result = predict(model, wave)
    assert result["logits"].shape == (1, 5) and np.isfinite(result["embedding"]).all()


@pytest.mark.parametrize("defect", ["checksum", "step", "arm", "seed", "protocol", "shape"])
def test_source_checkpoint_rejects_substitution(tmp_path, monkeypatch, defect):
    from mist_cove.phase11b.security import IntegrityError
    model = MISTSleep()
    payload = {"model": model.state_dict(), "step": 1200, "arm": "M0", "seed": 123,
               "protocol_sha256": SOURCE_PROTOCOL_SHA256}
    if defect == "step": payload["step"] = 750
    if defect == "arm": payload["arm"] = "M2"
    if defect == "seed": payload["seed"] = 456
    if defect == "protocol": payload["protocol_sha256"] = "a"*64
    if defect == "shape": payload["model"]["bank.vectors"] = torch.zeros(32, 64)
    path = tmp_path / "M0_123" / "step_1200.pt"; path.parent.mkdir(); torch.save(payload, path)
    monkeypatch.setitem(CHECKPOINTS, "M0_123", file_sha256(path) if defect != "checksum" else "0"*64)
    # Generated fixture only; production restricted_load keeps its patched-runtime gate.
    monkeypatch.setattr(source_checkpoint, "restricted_load", lambda snapshot: torch.load(io.BytesIO(snapshot.data), weights_only=True))
    with pytest.raises((IntegrityError, RuntimeError)): source_checkpoint.load_source(tmp_path, "M0", 123)
    with pytest.raises(IntegrityError): source_checkpoint.load_source(tmp_path, "M1", 123)


def test_frozen_inference_state_and_prediction_order():
    torch.set_num_threads(2)
    torch.manual_seed(70)
    model = MISTSleep().eval().requires_grad_(False)
    x = np.random.default_rng(2).normal(size=(3, 1, 3000)).astype("float32")
    before = state_digest(model)
    result = predict(model, x, batch_size=2)
    with torch.inference_mode(): expected = model(torch.from_numpy(x))["logits"].numpy()
    np.testing.assert_allclose(result["logits"], expected, atol=1e-6)
    assert state_digest(model) == before
    with frozen_execution(model):
        with pytest.raises(RuntimeError): torch.optim.Adam(model.parameters())
        with pytest.raises(RuntimeError): torch.tensor(1.).backward()
        with pytest.raises(RuntimeError): torch.autograd.grad(torch.tensor(1.), [])
    with pytest.raises(RuntimeError):
        with frozen_execution(model): next(model.parameters()).add_(1)
    model.train()
    with pytest.raises(RuntimeError): predict(model, x)


def make_seal(tmp_path):
    path = tmp_path / "generated_predictions.bin"; path.write_bytes(b"generated logits")
    return seal_predictions(tmp_path, [path.name], protocol_sha256="a"*64, manifest_sha256="b"*64,
                            model_identities=CHECKPOINTS, preprocessing_sha256="c"*64)


def test_prediction_seal_once_and_tamper(tmp_path):
    public = make_seal(tmp_path)
    assert "files" not in public
    begin_scoring(tmp_path, public)
    with pytest.raises(FileExistsError): begin_scoring(tmp_path, public)
    (tmp_path/"generated_predictions.bin").write_bytes(b"changed")
    with pytest.raises(ValueError): begin_scoring(tmp_path, public)


def test_label_access_rejected_before_file_open(tmp_path, monkeypatch):
    monkeypatch.setattr("mist_sleep.external.hmc_scoring.verify_file", lambda *a: pytest.fail("annotation touched before claim"))
    with pytest.raises(RuntimeError):
        read_scoring(tmp_path, "SN001", {}, epoch_count=1, waveform_start="x", claim_root=tmp_path)


def test_generated_complete_blind_seal_unblind_pipeline(tmp_path, monkeypatch):
    """Real generated EDF/CSV I/O, four model slots, both channels and paired scoring."""
    from mist_sleep.external import runner
    raw, output = tmp_path / "raw", tmp_path / "predictions"
    (raw / "recordings").mkdir(parents=True)
    manifest = []
    for number in (1, 2):
        recording = f"SN{number:03}"
        waveform = raw / "recordings" / f"{recording}.edf"
        with pyedflib.EdfWriter(str(waveform), 2, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
            writer.setSignalHeaders([dict(label=label, dimension="uV", sample_frequency=256,
                physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767,
                transducer="generated", prefilter="generated") for label in ("EEG C4-M1", "EEG F4-M1")])
            writer.setStartdatetime(datetime(2020, 1, 1))
            writer.writeSamples([np.zeros(7680*2), np.zeros(7680*2)])
            writer.writeAnnotation(0, 30, "Sleep stage W")
            writer.writeAnnotation(30, 30, "Sleep stage N2")
        scoring = waveform.with_name(recording+"_sleepscoring.edf")
        scoring.write_bytes(waveform.read_bytes())
        text = waveform.with_name(recording+"_sleepscoring.txt")
        text.write_text("onset_seconds,duration_seconds,annotation\n0,30,Sleep stage W\n30,30,Sleep stage N2\n")
        manifest.append({"recording": recording, "files": {f"recordings/{p.name}": file_sha256(p) for p in (waveform, scoring, text)}})
    monkeypatch.setattr(runner, "private_root", lambda: raw)
    monkeypatch.setattr(runner, "verified_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "require_frozen_protocol", lambda _: specification())
    monkeypatch.setattr(runner, "require_inference_readiness", lambda _: None)
    model = MISTSleep().eval().requires_grad_(False)
    monkeypatch.setattr(runner, "load_source", lambda *args: (model, {}))
    original_reader = pyedflib.EdfReader
    class ScoringSpy(original_reader):
        def readAnnotations(self):
            assert (output/"UNBLIND_STARTED.json").is_file()
            return super().readAnnotations()
    monkeypatch.setattr(pyedflib, "EdfReader", ScoringSpy)
    public = runner.blind(tmp_path, tmp_path, output)
    assert public["file_count"] == 18 and not (output/"UNBLIND_STARTED.json").exists()
    monkeypatch.setattr(runner, "committed_identity", lambda *args: public)
    result = runner.score(tmp_path, tmp_path, output, "generated_seal.json")
    assert result["C4/M1"]["statistics"]["recordings"] == 2
    assert result["C4/M1"]["statistics"]["effect"] == 0
    assert result["F4/M1"]["analysis"] == "secondary sensitivity"
    with pytest.raises(FileExistsError): runner.score(tmp_path, tmp_path, output, "generated_seal.json")


@pytest.mark.parametrize("events", [[(1, 30, "Sleep stage N2")], [(0, 29, "Sleep stage N2")],
    [(0, 30, "N2")], [(0, 30, "Sleep stage N2"), (0, 30, "Sleep stage R")],
    [(0, 60, "Sleep stage N2")], [(-30, 30, "Sleep stage N2")]])
def test_annotation_geometry_falsification(events):
    with pytest.raises(ValueError): epoch_labels(events, 1)


def test_label_mapping_gaps_and_schema(tmp_path):
    assert epoch_labels([(0, 30, "Sleep stage W"), (60, 30, "Sleep stage R")], 3).tolist() == [0, -1, 4]
    path = tmp_path / "generated.csv"; path.write_text("onset_seconds,duration_seconds,annotation\n0,30,Sleep stage N1\n")
    assert csv_events(path) == [(0., 30., "Sleep stage N1")]
    path.write_text("time,score\n0,N1\n")
    with pytest.raises(ValueError): csv_events(path)


def test_fixed_five_absent_classes_calibration_and_stage_permutation():
    assert stage_metrics(np.array([0, 0]), np.array([[1000., 0, 0, 0, 0]]*2))["macro_f1"] == .2
    wrong = stage_metrics(np.array([1, 1]), np.array([[1000., 0, 0, 0, 0]]*2))
    assert wrong["ece_10"] == 1 and wrong["high_confidence_wrong_count"] == 2
    y = np.tile(np.arange(5), 100)
    logits = np.eye(5)[y]*10
    assert stage_metrics(y, logits)["macro_f1"] == 1
    assert stage_metrics(y, np.roll(logits, 1, axis=1))["macro_f1"] == 0
    shuffled = np.random.default_rng(34).permutation(y)
    assert .1 < stage_metrics(shuffled, logits)["macro_f1"] < .3


def test_paired_bootstrap_keeps_seeds_and_cluster_recording_weights():
    a = np.ones((3, 2)) * .4
    b = a + np.array([[.01, .05], [.01, .05], [.01, .05]])
    result = paired_effect(a, b, ["a", "b", "c"])
    assert result["known_participants"] is None and result["clusters"] == 3
    np.testing.assert_allclose(result["interval_95"], [.03, .03])
    assert result["effect"] == pytest.approx(.03)
    clustered = paired_effect(a, b, ["a", "b", "c"], participant_groups=["p1", "p1", "p2"])
    assert clustered["clusters"] == 2 and clustered["effect"] == pytest.approx(.03)


@pytest.mark.parametrize("defect", ["epoch_ids", "seed_rows", "one_cluster", "bad_draws", "bad_grouping"])
def test_invalid_bootstrap_units_rejected(defect):
    a, b, ids, kwargs = np.ones((3, 2))*.4, np.ones((3, 2))*.5, ["a", "b", "c"], {}
    if defect == "epoch_ids": ids = ["a", "a", "b"]
    if defect == "seed_rows": a = a.T; b = b.T
    if defect == "one_cluster": kwargs["participant_groups"] = ["p"]*3
    if defect == "bad_draws": kwargs["draws"] = 500
    if defect == "bad_grouping": kwargs["participant_groups"] = ["p"]
    with pytest.raises(ValueError): paired_effect(a, b, ids, **kwargs)


@pytest.mark.parametrize("effect,seeds,ci,integrity,expected", [
    (.03, [.02, .04], [.01, .05], True, "MPI_HMC_CROSS_MONTAGE_TRANSFER_POSITIVE"),
    (.01, [.01, .01], [.001, .02], True, "MPI_HMC_PRIMARY_GAIN_NOT_ESTABLISHED"),
    (.03, [-.01, .07], [.01, .05], True, "MPI_HMC_RESULT_INCONCLUSIVE"),
    (.03, [.02, .04], [-.01, .05], True, "MPI_HMC_RESULT_INCONCLUSIVE"),
    (None, [], None, False, "PRIMARY_EVALUATION_BLOCKED")])
def test_scientific_gate(effect, seeds, ci, integrity, expected):
    assert scientific_decision(effect, seeds, ci, integrity_passed=integrity)["decision"] == expected


def test_representation_and_publication_do_not_imply_transfer():
    result = summarize(np.zeros((3, 128)), np.ones((3, 64))/64)
    assert result["embedding_feature_variance"] == 0 and result["source_distance"] is None
    assert result["effective_prototype_usage"] == pytest.approx(64)
    with pytest.raises(ValueError): public_aggregate({"recording_id": "SN001", "predictions": [1]})
    with pytest.raises(ValueError): public_aggregate({"reason": "/private/source/path"})
    with pytest.raises(ValueError): public_aggregate({"effect": float("nan")})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires compatible CUDA; inference only")
def test_generated_cuda_inference_no_training():
    torch.manual_seed(77)
    model = MISTSleep().eval().requires_grad_(False)
    x = np.random.default_rng(55).normal(size=(2, 1, 3000)).astype("float32")
    expected = predict(model, x)["logits"]
    model.cuda()
    actual = predict(model, x)["logits"]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

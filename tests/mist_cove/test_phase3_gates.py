"""Synthetic checks of explicit source permission and independent launch gates."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mist_cove.phase3 import source as s
from mist_cove.phase2.synthetic_runtime import mock_recordings


@pytest.fixture
def config():
    return json.loads((s.ROOT / 'configs/mist_cove_phase3_source_v1.json').read_text())


@pytest.mark.parametrize('change', [{'variants': ['q1']}, {'seeds': [123]}, {'max_epochs': 1}, {'amp': True},
                                  {'maximum_gpu_hours': 25}, {'reserved_access': True}, {'lr': .01}, {'temporal_radius': 1}])
def test_recipe_mutations_rejected_without_source_access(config, change):
    with patch.object(Path, 'open', side_effect=AssertionError('data opened')):
        assert s.prepared(config)['max_epochs'] == 60
        config.update(change)
        with pytest.raises(ValueError):
            s.prepared(config)


def test_missing_authorization_before_any_file_open():
    with patch.object(Path, 'open', side_effect=AssertionError('data opened')):
        with pytest.raises(ValueError, match='authorization required'):
            s.preflight('unused', 'unused')
        with pytest.raises(ValueError, match='authorization required'):
            s.run_reviewed('unused', 'unused', 'unused', 'unused', gate='a')


def test_gpu_contention_rejects_without_signalling(monkeypatch):
    monkeypatch.setattr(s.subprocess, 'check_output', lambda *a, **kw: '999999\n')
    with patch('os.kill', side_effect=AssertionError('signalled other user')):
        with pytest.raises(RuntimeError, match='other compute processes'):
            s.require_idle_gpu()


def test_private_path_rejects_public_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'PRIVATE', tmp_path / 'private')
    (tmp_path / 'private').mkdir()
    (tmp_path / 'private/escape').symlink_to(tmp_path / 'public', target_is_directory=True)
    for path in [tmp_path / 'public/result', tmp_path / 'private/escape/result', tmp_path / 'private']:
        with pytest.raises(ValueError):
            s.private_path(path)


@pytest.mark.parametrize('review', [{}, {'reviewer': 'codex'}, {'reviewer': 'self'}, {'reviewer': 'agent'},
                                  {'decision': 'APPROVE_GATE_A', 'reviewer': 'human', 'proposal_sha256': 'wrong'}])
def test_missing_or_self_review_never_launches(tmp_path, review):
    p, r = tmp_path / 'proposal.json', tmp_path / 'review.json'
    p.write_text('{}'); r.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='independent user launch review'):
        s.validate_review(p, r)


def test_preflight_scans_unscored_and_all_positions_without_training(tmp_path, monkeypatch, config):
    p = tmp_path / 'config.json'; p.write_text(json.dumps(config))
    train, val = mock_recordings()
    # An array wrapper counts all physical reads and supplies loader telemetry.
    class View:
        def __init__(self, x):
            self.x = x; self.read_batches = 0; self.peak_bytes = 0
        def __getitem__(self, sl):
            x = self.x[sl]; self.read_batches += 1; self.peak_bytes = max(self.peak_bytes, x.nbytes)
            return x
    for row in [train, val]:
        row.x = View(row.x)
    monkeypatch.setattr(s, 'PRIVATE', tmp_path)
    monkeypatch.setattr(s, 'configure_precision', lambda *a: None)
    monkeypatch.setattr(s, 'require_idle_gpu', lambda: {'mock': True})
    monkeypatch.setattr(s, 'reconciled_environment', lambda: {'scope': 'MOCK_ONLY'})
    monkeypatch.setattr(s, 'load_sources', lambda c: ([train], [val], {}, {}, {'mock': True}, {}))
    trainer = Mock(side_effect=AssertionError('preflight trained'))
    monkeypatch.setattr(s, 'sealed_train', trainer)
    result = s.preflight(p, tmp_path / 'preflight', authorized=True)
    assert result['source_scan']['train']['physical_epochs'] == 10
    assert result['source_scan']['val']['physical_epochs'] == 10
    assert result['training_executed'] is False
    trainer.assert_not_called()
    with pytest.raises(FileExistsError):
        s.preflight(p, tmp_path / 'preflight', authorized=True)


def test_proof_tampering_rejected(tmp_path):
    p = tmp_path / 'receipt.json'; p.write_text('first')
    proof = s.file_proof(p); p.write_text('second')
    with pytest.raises(ValueError, match='digest changed'):
        s.check_proof(proof)


def test_environment_mismatch_stops_before_source_loading(monkeypatch):
    monkeypatch.setattr(s, 'environment', lambda device: {})
    monkeypatch.setattr(s.metadata, 'version', lambda package: '0.0.mismatch')
    with pytest.raises(ValueError, match='import/metadata mismatch'):
        s.reconciled_environment()


def test_failed_correctness_cannot_form_launch_receipt(tmp_path):
    p = tmp_path / 'cpu.json'; p.write_text(json.dumps({'status': 'FAIL'}))
    g = tmp_path / 'cuda.json'; g.write_text('{}')
    with pytest.raises(ValueError, match='CPU correctness'):
        s.validate_correctness({'cpu': s.file_proof(p), 'cuda': s.file_proof(g)}, {})


def test_recorded_original_path_not_silently_replaced(tmp_path):
    directory = tmp_path / 'SLeep_edf_fulldataset/sleep-cassette'; directory.mkdir(parents=True)
    wave, annotation = directory / 'generated-PSG.edf', directory / 'generated-Hypnogram.edf'
    wave.write_bytes(b'generated waveform fixture'); annotation.write_bytes(b'generated annotation fixture')
    record = {'split': 'train', 'path': str(wave), 'annotation_source': str(annotation),
              'source_sha256': s.file_sha(wave), 'annotation_sha256': s.file_sha(annotation)}
    proofs = s.verify_source_files(record)
    assert proofs[0]['path'] == str(wave)
    wave.write_bytes(b'changed waveform')
    with pytest.raises(ValueError, match='hash mismatch'):
        s.verify_source_files(record)
    record['split'] = 'test'
    with patch.object(Path, 'open', side_effect=AssertionError('reserved file opened')):
        with pytest.raises(ValueError, match='unauthorized role'):
            s.verify_source_files(record)

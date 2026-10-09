import json
import pytest
from mist_transfer.provenance import AuditRun, atomic_text


def test_interruption_resume_completion_and_changed_inputs(tmp_path):
    run = AuditRun(tmp_path, {'a': 1}, {'record': 'hash'})
    atomic_text('deterministic', tmp_path/'result.txt')
    assert not (tmp_path/'COMPLETE.json').exists()
    with pytest.raises(FileExistsError):
        AuditRun(tmp_path, {'a': 1}, {'record': 'hash'})
    resumed = AuditRun(tmp_path, {'a': 1}, {'record': 'hash'}, resume=True)
    assert resumed.signature == run.signature
    resumed.finish(['result.txt'], 'PASS')
    AuditRun(tmp_path, {'a': 1}, {'record': 'hash'}, resume=True)
    with pytest.raises(ValueError, match='signature'):
        AuditRun(tmp_path, {'a': 2}, {'record': 'hash'}, resume=True)
    atomic_text('corrupted', tmp_path/'result.txt')
    with pytest.raises(ValueError, match='digest'):
        AuditRun(tmp_path, {'a': 1}, {'record': 'hash'}, resume=True)

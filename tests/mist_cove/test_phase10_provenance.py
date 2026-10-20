import pytest
from mist_cove.phase10.provenance import REQUIRED,assess_metadata,committed_aggregate


def test_complete_metadata_never_authorizes_deserialization():
    fields={k:'synthetic_metadata' for k in REQUIRED};fields.update(variant='raw_context',seed=123)
    result=assess_metadata(fields)
    assert result['metadata_status']=='READY_FOR_SEPARATE_REVIEW'
    assert result['authentication_status']=='BLOCKED'
    fields['original_checkpoint_byte_digest']=None
    assert 'original_checkpoint_byte_digest' in assess_metadata(fields)['missing_independent_evidence']


def test_unknown_artifact_metadata_path_rejected_before_git_or_io(monkeypatch):
    import mist_cove.phase10.provenance as module
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:pytest.fail('must reject before read'))
    for name in ['best.pt','/absolute/source/normalization.json','../bank','reports/p3b/results.json/../best.pt']:
        with pytest.raises(PermissionError):committed_aggregate(name)

import hashlib
import json
import subprocess
import pytest
from mist_cove.phase11a import provenance as p


def test_strict_manifest_and_typed_digest_distinction():
    root={'status':'ALL_12_VALIDATED_COMPLETE','run_fingerprint':'a'*64,'outputs':{'seed_123/raw_context/best.pt':'b'*64}}
    assert p.parse_publication_manifest(p.canonical(root),'p3b-root-completion-v1')==root
    ref={'kind':'MODEL_STATE_DIGEST','experiment':'P3B','value':'b'*64}
    with pytest.raises(p.InvalidEvidence):p.typed_digest_equal(ref,'CHECKPOINT_BYTES_SHA256','P3B','b'*64)
    with pytest.raises(p.InvalidEvidence):p.typed_digest_equal(ref,'MODEL_STATE_DIGEST','MIST_COVE_PHASE_III','b'*64)
    assert p.typed_digest_equal(ref,'MODEL_STATE_DIGEST','P3B','b'*64)
    for data in [b'{"x":1,"x":2}',b'{"x":NaN}',b'[]',b'{}']:
        with pytest.raises(p.InvalidEvidence):p.parse_publication_manifest(data,'p3b-root-completion-v1')


@pytest.mark.parametrize('name',['/source/best.pt','../bank.pt','a/../bank','a//b','a\\b','C:best.pt'])
def test_logical_paths_reject_unsafe_names(name):
    with pytest.raises(p.InvalidEvidence):p.logical_name(name)


def test_wrong_marker_schema_run_seed_variant_rejected():
    marker={'status':'VALIDATED_COMPLETE','run_fingerprint':'a'*64,'seed':123,'variant':'raw_context','outputs':{'best.pt':'b'*64}}
    assert p.parse_publication_manifest(p.canonical(marker),'p3b-validated-completion-v1')==marker
    for field,value in [('seed',456),('seed',True),('variant','evidence'),('status','COMPLETE')]:
        wrong={**marker,field:value}
        with pytest.raises(p.InvalidEvidence):p.parse_publication_manifest(p.canonical(wrong),'p3b-validated-completion-v1')


def test_rejected_artifact_path_never_invokes_git_or_filesystem(monkeypatch):
    monkeypatch.setattr(p,'git',lambda *args:pytest.fail('must reject before access'))
    for path in ['best.pt','/absolute/real/checkpoint','mist_transfer_runs/p3b/v1/COMPLETE.json','../source']:
        with pytest.raises((PermissionError,p.InvalidEvidence)):p.historical_blob(p.P3B_PUBLICATION,path)
    with pytest.raises(PermissionError):p.real_artifact_access(object(),role='train',authorize=True)


def test_actual_generated_git_commit_blob_and_symlink_identity(tmp_path,monkeypatch):
    # This is a fresh generated Git repository, not a historical/private run.
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],stderr=subprocess.PIPE)
    git('init');git('config','user.name','Synthetic Fixture');git('config','user.email','fixture@example.invalid')
    path=tmp_path/'reports/p3b/results.json';path.parent.mkdir(parents=True);path.write_text('{"generated":true}')
    git('add','.');git('commit','-m','generated fixture');commit=git('rev-parse','HEAD').decode().strip()
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'COMMITS',{commit})
    data,record=p.historical_blob(commit,'reports/p3b/results.json')
    expected=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert record['blob_oid']==expected
    with pytest.raises(p.InvalidEvidence):p.historical_blob(commit,'reports/p3b/results.json',expected_blob='0'*40)
    path.unlink();path.symlink_to('outside_generated');git('add','.');git('commit','-m','generated symlink');second=git('rev-parse','HEAD').decode().strip();monkeypatch.setattr(p,'COMMITS',{second})
    with pytest.raises(p.InvalidEvidence,match='symlink'):p.historical_blob(second,'reports/p3b/results.json')

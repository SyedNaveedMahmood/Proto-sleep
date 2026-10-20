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


@pytest.mark.parametrize('attack',['split_value','split_kind','phase3_seal','phase3_as_p3b','invented_checkpoint_hash'])
def test_typed_inventory_references_cannot_drift_from_publication(monkeypatch,attack):
    # Generated publication bytes isolate reference binding from the Git transport,
    # which is exercised with real generated Git objects above.
    import copy
    a='a'*64;b='b'*64;c='c'*64;empty=p.sha256(p.canonical({}))
    summary={'run_fingerprint':a,'implementation_commit':p.P3B_IMPL,'split_fingerprint':b,'root_completion_sha256':c,
             'implementation_hash':empty,'reused_implementation_hash':empty}
    result={'run_fingerprint':a,'implementation_commit':p.P3B_IMPL,'split_fingerprint':b,
            'per_seed':[{'variant':'raw_context','seed':123,'best_state_digest':a}]}
    phase3={'split_fingerprint':b,'split_file_sha256':a,'rows':[{'variant':'raw_context','seed':s,'complete_seal_sha256':c} for s in (123,456,789)]}
    objects={(p.P3B_PUBLICATION,'reports/p3b/validation_summary.json'):summary,
             (p.P3B_PUBLICATION,'reports/p3b/results.json'):result,
             (p.P3B_IMPL,'configs/p3b_sc_matched_v1.json'):{'split_fingerprint':b},
             (p.PHASE3_PUBLICATION,'reports/mist_cove/phase3_source_v1/source_results.json'):phase3,
             (p.PHASE3_PUBLICATION,'reports/mist_cove/phase3_source_v1/execution_receipt.json'):{'split_fingerprint':b,'split_file_sha256':a}}
    raw={key:p.canonical(value) for key,value in objects.items()}
    monkeypatch.setattr(p,'historical_blob',lambda commit,path,**kw:(raw[(commit,path)],{}))
    monkeypatch.setattr(p,'git',lambda *args:b'')
    inventory={'base_commit':p.BASE,'historical_private_artifacts_opened':0,'real_checkpoint_bytes_hashed':0,
               'records':[{'commit':key[0],'path':key[1],'blob_oid':'0'*40,'blob_sha256':p.sha256(value)} for key,value in raw.items()],
               'candidate':{'experiment':'P3B','variant':'raw_context','seed':123,'implementation_commit':p.P3B_IMPL,'run_fingerprint':a,'model_state_digest':a,'original_serialized_checkpoint_sha256':None},
               'source_manifests':{k:{'files':{},'canonical_SHA256':empty} for k in ('implementation','reused')},
               'typed_references':[{'kind':kind,'experiment':'P3B','value':value} for kind,value in [('MODEL_STATE_DIGEST',a),('COMPLETION_MARKER_BYTES_SHA256',c),('SPLIT_FINGERPRINT',b),('SPLIT_FILE_SHA256',a)]],
               'phase3_seal_references':[{'experiment':'MIST_COVE_PHASE_III','variant':'raw_context','seed':s,'source_commit':p.PHASE3_PUBLICATION,'checkpoint_byte_sha256':None,'complete_seal_sha256':c} for s in (123,456,789)]}
    assert p.verify_inventory(inventory)['status']=='PASS'
    wrong=copy.deepcopy(inventory)
    if attack=='split_value':wrong['typed_references'][2]['value']=a
    elif attack=='split_kind':wrong['typed_references'][2]['kind']='MODEL_STATE_DIGEST'
    elif attack=='phase3_seal':wrong['phase3_seal_references'][0]['complete_seal_sha256']=a
    elif attack=='phase3_as_p3b':wrong['phase3_seal_references'][0]['experiment']='P3B'
    else:wrong['candidate']['original_serialized_checkpoint_sha256']=a
    with pytest.raises(p.InvalidEvidence):p.verify_inventory(wrong)

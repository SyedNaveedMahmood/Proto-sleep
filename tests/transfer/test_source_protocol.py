"""Synthetic P3A scientific invariants; never depend on private EEG outputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score
from test_preprocessing import synthetic
from mist_transfer.protocol import (descriptor_identity,split_policy,validate_role_records,
    load_development,read_frozen_split,freeze_source_split,SPLIT_VERSION,IDENTITY_KEYS)
from mist_transfer.provenance import atomic_json,fingerprint,sha256_file
from mist_transfer.preprocessing import build_record,SourceProofs,iter_recording
from mist_transfer.source_data import fit_normalization,SourceRecording,training_bank,validate_bank
from mist_transfer.metrics import classification_metrics
from mist_transfer.baselines import make_baseline
from mist_transfer.experiment import evaluate_source,verify_outputs
from mist_evidence.model import ModelConfig
from mist_evidence.data import Recording,build_anchors,blocks
from mist_evidence.runtime import train,TrainConfig,seed_all,load_trusted,tensor_digest


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


def identity_rows():
    people=[i for i in range(78) if i not in {6,23,36,74}];rows=[]
    for i in people:
        for night in range(1,3 if i not in {0,13} else 2):
            rid=f'SC4{i:02d}{night}E0'
            rows.append(dict(study='SC',subject_id=f'SC:{i:02d}',night=night,recording_id=rid,
                recording_variant='E',path='/unavailable/'+rid+'-PSG.edf',annotation_source=rid[:-1]+'C-Hypnogram.edf',
                **{key:fingerprint([key,rid]) for key in ['source_sha256','annotation_sha256','signal_sha256','shape_sha256']}))
    return rows,[f'SC:{i:02d}' for i in people if i>=20]


def test_split_is_deterministic_label_independent_and_grouped():
    rows,extension=identity_rows();split=split_policy(rows,extension)
    assert list(split['subject_roles'].values()).count('train')==15
    assert list(split['subject_roles'].values()).count('val')==4
    assert list(split['subject_roles'].values()).count('test')==55
    assert split==split_policy(copy.deepcopy(rows),extension)
    changed=copy.deepcopy(rows)
    for r in changed:r['forbidden_stage_counts']={'REM':999}
    assert split_policy(changed,extension)['subject_roles']==split['subject_roles']
    assert all(r['split']==split['subject_roles'][r['subject_id']] for r in split['records'])
    assert split_policy(rows,extension,456)['subject_roles']!=split['subject_roles']


@pytest.mark.parametrize('key',['subject_id','night','source_sha256','signal_sha256','shape_sha256','path','recording_id'])
def test_split_corruption_rejected_before_io(key):
    rows,extension=identity_rows();split=split_policy(rows,extension);bad=copy.deepcopy(split['records'])
    bad[2][key]=bad[0][key]
    with patch.object(Path,'open',side_effect=AssertionError('source opened')):
        with pytest.raises(ValueError):validate_role_records(bad)


def test_reserved_annotations_and_waveforms_have_no_development_opening_path(tmp_path):
    rows,extension=identity_rows();split=split_policy(rows,extension);expected=[]
    for r in split['records']:
        p=tmp_path/(r['recording_id']+'.json');r['descriptor']=str(p)
        if r['split']=='test':r['descriptor_sha256']='must never read';continue
        record={k:r[k] for k in IDENTITY_KEYS}
        atomic_json({'record':record,'record_sha256':fingerprint(record)},p)
        r['descriptor_sha256']=sha256_file(p)
        if r['split']=='train':expected.append(r['recording_id'])
    assert [r['recording_id'] for r in load_development(split,'train')]==expected
    with patch.object(Path,'open',side_effect=AssertionError('reserved file opened')):
        with pytest.raises(ValueError,match='reserved test'):load_development(split,'test')


def test_scalar_prefix_never_fetches_annotations(tmp_path):
    row=identity_rows()[0][0];prefix={k:row[k] for k in IDENTITY_KEYS}
    p=tmp_path/'descriptor.json'
    p.write_text(json.dumps({**prefix,'schema_version':'v1'},indent=2)[:-2]+',\n  "annotation_segments": THIS IS DELIBERATELY NOT JSON')
    assert descriptor_identity(p)==prefix


def test_frozen_modifications_rejected_even_when_semantic_fingerprint_recomputed(tmp_path):
    rows,ext=identity_rows();frozen=split_policy(rows,ext);frozen['split_fingerprint']=fingerprint(frozen)
    p=tmp_path/'frozen.json';atomic_json(frozen,p)
    atomic_json({'split_fingerprint':frozen['split_fingerprint'],'file_sha256':sha256_file(p)},p.with_suffix('.receipt.json'))
    assert read_frozen_split(p)==frozen
    frozen['seed']=456;frozen['split_fingerprint']=fingerprint({k:v for k,v in frozen.items() if k!='split_fingerprint'})
    atomic_json(frozen,p)
    with pytest.raises(ValueError,match='fingerprint/digest'):read_frozen_split(p)


def test_freeze_resume_rejects_seed_change_and_does_not_read_test_labels(tmp_path,monkeypatch):
    import mist_transfer.protocol as protocol
    monkeypatch.chdir(tmp_path);rows,extension=identity_rows();canonical=tmp_path/'canonical';canonical.mkdir()
    refs=[];hashes={}
    for row in rows:
        p=canonical/(row['recording_id']+'.json');atomic_json({**row,'schema_version':protocol.SCHEMA,'annotations':'must not read'},p)
        hashes[p.name]=sha256_file(p);refs.append({'recording_id':row['recording_id'],'subject_id':row['subject_id'],'descriptor':p.name,'descriptor_sha256':hashes[p.name]})
    atomic_json({'schema_version':protocol.SCHEMA,'mode':'complete','signature':'synthetic-p2','records':refs},canonical/'manifest.json')
    hashes['manifest.json']=sha256_file(canonical/'manifest.json');atomic_json({'outputs':hashes},canonical/'COMPLETE.json')
    gate=tmp_path/'reports/data_audit/p1_5/gate_decisions.json'
    atomic_json({'sc_within_cohort':{'eligible_recordings':[r['recording_id'] for r in rows],'eligible_extension_subjects':extension}},gate)
    monkeypatch.setattr(protocol,'GATE_SHA256',sha256_file(gate));monkeypatch.setattr(protocol,'P2_COMPLETION',sha256_file(canonical/'COMPLETE.json'))
    monkeypatch.setattr(protocol,'git_info',lambda:{'sha':'synthetic'})
    p=tmp_path/'split.json';first=freeze_source_split(canonical,p)
    assert freeze_source_split(canonical,p,resume=True)==first
    with pytest.raises(ValueError,match='modification refused'):freeze_source_split(canonical,p,seed=456,resume=True)
    assert read_frozen_split(p)==first


def test_fixed_five_metrics_subject_average_masks_and_absent_classes():
    y=np.array([0,0,1,2,3,4,-1,0,0]);p=np.array([0,1,1,2,4,4,4,0,0]);s=np.array(['A']*7+['B']*2)
    m=y>=0;r=classification_metrics(y,p,s,m)
    assert r['macro_f1']==pytest.approx(f1_score(y[m],p[m],labels=range(5),average='macro',zero_division=0))
    assert r['kappa']==pytest.approx(cohen_kappa_score(y[m],p[m]))
    assert r['balanced_accuracy']==pytest.approx(balanced_accuracy_score(y[m],p[m]))
    assert r['mean_subject_macro_f1']==pytest.approx(np.mean([f1_score(y[m&(s==sid)],p[m&(s==sid)],labels=range(5),average='macro',zero_division=0) for sid in ['A','B']]))
    assert r['mean_subject_macro_f1']!=pytest.approx(r['macro_f1'])
    assert r['subject_metrics']['B']['kappa'] is None and r['subject_metrics']['B']['per_stage_recall'][1] is None
    assert r['subject_metrics']['B']['macro_f1']==.2 and r['subject_metrics']['B']['balanced_accuracy']==1
    assert r['n_epochs']==8 and r['unscored_epochs']==1
    with pytest.raises(ValueError,match='no scored epochs'):classification_metrics([-1],[0],['A'])
    with pytest.raises(ValueError,match='do not silently drop'):classification_metrics([0,-1],[0,0],['A','B'])
    with pytest.raises(ValueError,match='invalid scored'):classification_metrics([-1],[0],['A'],[True])


def raw_fixture(tmp_path,seconds=270):
    row,_=synthetic(tmp_path,seconds=seconds,intervals=[(0,seconds,'Sleep stage W')] if seconds<270 else None)
    r=build_record(row,SourceProofs());r['split']='train';return r


@pytest.mark.parametrize('role',['val','test',None])
def test_statistics_and_anchors_reject_nontrain_before_io(role):
    r={'split':role}
    with patch('mist_transfer.source_data.iter_recording',side_effect=AssertionError('nontrain EEG read')):
        with pytest.raises(ValueError,match='development role'):fit_normalization([r])
        with pytest.raises(ValueError,match='development role'):training_bank([r],{},123)


def test_train_norm_includes_unscored_physical_grid_without_annotation_access(tmp_path):
    r=raw_fixture(tmp_path)
    with patch('mist_transfer.preprocessing.epoch_targets',side_effect=AssertionError('labels accessed for normalization')):
        norm=fit_normalization([r])
    x=np.concatenate([b['x'] for b in iter_recording(r)]).astype(np.float64)
    assert norm['sample_count']==27000 and norm['physical_epoch_count']==9 and norm['label_access'] is False
    assert norm['mean_uV']==pytest.approx(x.mean()) and norm['std_uV']==pytest.approx(x.std())
    source=SourceRecording(r,norm);np.testing.assert_allclose(source.x[:],((x-norm['mean_uV'])/norm['std_uV']).astype(np.float32),atol=3e-7)
    assert source.indices.tolist()==list(range(9)) and source.scoring_mask.tolist()==[True]*6+[False]*3
    assert source.segments()==[(0,6)]
    source.indices=np.array([0,1,3,4,8,9,10,11,12]);source.scoring_mask=np.array([1,1,1,0,1,1,0,1,1],dtype=bool)
    assert source.segments()==[(0,2),(2,3),(4,6),(7,9)]


def test_short_smoke_view_keeps_nonnegative_physical_indices(tmp_path):
    r=raw_fixture(tmp_path,60);norm=fit_normalization([r]);source=SourceRecording(r,norm,smoke=True)
    assert source.indices.tolist()==[0,1]


def test_real_training_waveform_anchor_provenance_and_tampering(tmp_path):
    r=raw_fixture(tmp_path);norm=fit_normalization([r]);bank=training_bank([r],norm,123)
    validate_bank(bank,[r],norm,verify_waveforms=True)
    bad=copy.deepcopy(bank);bad['metadata'][0][0]['split']='test'
    with pytest.raises(ValueError,match='non-TRAIN'):validate_bank(bad,[r],norm)
    bad=copy.deepcopy(bank);bad['metadata'][0][0]['absolute_start_sample']+=1
    with pytest.raises(ValueError,match='original epoch/sample'):validate_bank(bad,[r],norm)
    bad=copy.deepcopy(bank);bad['waveforms'][0][0,0]+=.01
    with pytest.raises(ValueError,match='real source waveform'):validate_bank(bad,[r],norm,True)


def array_data():
    rng=np.random.default_rng(42);records=[]
    for subject,role in [('TRAIN_A','train'),('VAL_B','val')]:
        x=rng.normal(size=(10,1,3000)).astype(np.float32);y=np.array([0,1,2,3,4,0,1,2,3,4])
        r=Recording(subject,subject,x,y,np.arange(10),'synthetic-'+subject,False,True,True)
        r.scoring_mask=np.ones(10,dtype=bool);r.scoring_mask[3]=False;r.y[3]=-1
        r.split=role;r.recording_id=subject;r.segments=SourceRecording.segments.__get__(r)
        records.append(r)
    cfg=ModelConfig(scales=(100,),prototypes_per_scale=2,embedding_dim=8,radius=0,use_crf=False)
    bank=build_anchors([records[0]],cfg,per_subject=8)
    return records,bank


@pytest.mark.parametrize('variant',['attnsleep','evidence','summary_only','raw_context','evidence_aug'])
def test_adapters_masked_finite_gradients_and_streamed_evaluation(variant):
    records,bank=array_data();seed_all(123);model=make_baseline(variant,bank,{'mean_uV':0.,'std_uV':1.})
    model.train();x=torch.from_numpy(records[0].x[:5]);y=torch.tensor([[0,1,2,-1,4]]);mask=y>=0
    out=model.emissions(model.encode(x)[None]);loss=model.loss(out,y,mask);assert torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metrics,rows=evaluate_source(model,[records[1]],torch.device('cpu'),3)
    assert metrics['n_epochs']==9 and metrics['unscored_epochs']==1 and len(rows)==10
    assert rows[3]['truth'] is None and rows[3]['original_epoch_index']==3
    with pytest.raises(ValueError,match='source validation only'):evaluate_source(model,[records[0]],torch.device('cpu'),3)
    if variant=='evidence_aug':
        model.eval();a=model.encode(x);b=model.encode(x);assert torch.equal(a,b)


def test_masked_streaming_checkpoint_resume_is_exact(tmp_path):
    records,bank=array_data();recipe=TrainConfig(epochs=2,patience=5,core_epochs=4,encode_batch=4)
    def fresh():seed_all(123);return make_baseline('evidence',bank,{'mean_uV':0.,'std_uV':1.})
    args=([records[0]],[records[1]],recipe)
    train(fresh(),*args,tmp_path/'full',bank,{'code':'synthetic-p3'},torch.device('cpu'),evaluate_fn=evaluate_source)
    with pytest.raises(InterruptedError):train(fresh(),*args,tmp_path/'resume',bank,{'code':'synthetic-p3'},torch.device('cpu'),evaluate_fn=evaluate_source,interrupt_after_epoch=1)
    assert not (tmp_path/'resume/COMPLETE.json').exists()
    train(fresh(),*args,tmp_path/'resume',bank,{'code':'synthetic-p3'},torch.device('cpu'),resume=True,evaluate_fn=evaluate_source)
    a=load_trusted(tmp_path/'full/last.pt');b=load_trusted(tmp_path/'resume/last.pt')
    assert tensor_digest(a['state_dict'])==tensor_digest(b['state_dict'])
    assert torch.equal(a['rng']['torch'],b['rng']['torch']) and a['epoch']==b['epoch']==2
    for k in a['optimizer']['state']:
        for name,value in a['optimizer']['state'][k].items():assert torch.equal(value,b['optimizer']['state'][k][name])
    train(fresh(),*args,tmp_path/'resume',bank,{'code':'synthetic-p3'},torch.device('cpu'),resume=True,evaluate_fn=evaluate_source)
    with pytest.raises(ValueError,match='signature'):train(fresh(),*args,tmp_path/'resume',bank,{'code':'changed'},torch.device('cpu'),resume=True,evaluate_fn=evaluate_source)
    covered=[i for _,_,_,lo,hi in blocks([records[0]],4,0) for i in range(lo,hi)]
    assert covered==[0,1,2,4,5,6,7,8,9]


def test_completed_output_corruption_is_rejected(tmp_path):
    (tmp_path/'artifact.txt').write_text('original')
    marker=tmp_path/'COMPLETE.json';atomic_json({'run_fingerprint':'frozen','outputs':{'artifact.txt':sha256_file(tmp_path/'artifact.txt')}},marker)
    verify_outputs(tmp_path,marker,'frozen');(tmp_path/'artifact.txt').write_text('changed')
    with pytest.raises(ValueError,match='digest changed'):verify_outputs(tmp_path,marker,'frozen')

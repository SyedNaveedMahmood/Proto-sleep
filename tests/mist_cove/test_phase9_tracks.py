import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import optimizer,seal,restore
from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_evidence.runtime import rng_state,restore_rng
from mist_cove.phase9 import experiments as ex
from mist_cove.phase9.objectives import paired_objective,js_divergence
from mist_cove.phase9.operators import sample_query_masks,atom_to_candidate_mask,response_table
from mist_cove.phase9.clauses import verify_clause


def test_namespace_streams_and_class_blind_masks():
    keys=[ex.seed_for(role,t,s) for role in ('train','development','calibration','circuit_discovery','circuit_verification/0','locked_evaluation') for t in ('single','multiple') for s in (123,456,789)]
    assert len(set(keys))==len(keys)
    a=sample_query_masks(128,torch.Generator().manual_seed(918));b=sample_query_masks(128,torch.Generator().manual_seed(918))
    assert torch.equal(a,b) and atom_to_candidate_mask(a).any(-1).all() and not atom_to_candidate_mask(a).all(-1).any()


def test_additive_training_views_preserve_original_diagnostic_and_label():
    d=generate(10,18391,'multiple');hf=ex.add_view(d,'multiple',123,'fixture/hf','high_frequency');art=ex.add_view(d,'multiple',123,'fixture/art','transient')
    assert torch.equal(d.events,hf.events) and torch.equal(d.events,art.events)
    assert torch.equal(d.y,hf.y) and d.intervals==art.intervals
    assert not torch.equal(hf.base,d.base) and not torch.equal(hf.base,art.base)


@pytest.mark.parametrize('arm',['C0','C1','C2'])
def test_paired_losses_no_masked_cross_entropy_and_gradients(arm,monkeypatch):
    m=make_model('M1');x=generate(10,91822).x[:5];f,r=m.encode_candidates(x);g,s=m.encode_candidates(x+.3)
    calls=[];original=torch.nn.functional.cross_entropy
    def ce(logits,y):calls.append(logits.shape);return original(logits,y)
    monkeypatch.setattr(torch.nn.functional,'cross_entropy',ce)
    loss,parts=paired_objective(m,f,r,g,s,torch.arange(5),sample_query_masks(5),arm,150,300)
    loss.backward();assert calls==[torch.Size([5,5]),torch.Size([5,5])]
    assert torch.isfinite(loss) and m.encoder.branches[0].weight.grad.abs().sum()>0
    assert parts['warmup']==1
    if arm!='C2':assert parts['query_js']==0
    assert abs(float(js_divergence(torch.ones(3,5),torch.ones(3,5))))<1e-7


@pytest.mark.parametrize('arm',['C0','C1','C2'])
def test_exact_paired_model_optimizer_global_and_batch_rng_resume(arm,tmp_path):
    from mist_cove.phase2.training import configure_precision
    configure_precision('cpu');torch.set_num_threads(2);torch.manual_seed(384)
    c={**ex.config(),'updates':6,'batch_size':5};m=make_model('M1');initial=copy.deepcopy(m.state_dict());start=rng_state()
    d=generate(10,9172);plus=ex.training_pair(d,'single',123,'fixture_pair')
    def fresh():
        m=make_model('M1');m.load_state_dict(initial);o=optimizer(m,c);g=torch.Generator().manual_seed(728);restore_rng(start);return m,o,g
    m,o,g=fresh();ex.train_paired_steps(m,o,g,d.x,plus,d.y,c,arm,0,6,Budget(60));expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g.get_state(),rng_state())
    a,b,z=fresh();ex.train_paired_steps(a,b,z,d.x,plus,d.y,c,arm,0,3,Budget(60));ident={'scope':'SYNTHETIC_FIXTURE','arm':arm}
    proof=seal(tmp_path/'new.pt',a,b,z,3,ident);u,v,w=fresh();restore(tmp_path/'new.pt',proof,ident,u,v,w)
    ex.train_paired_steps(u,v,w,d.x,plus,d.y,c,arm,3,6,Budget(60));assert_exact(expected,(u.state_dict(),v.state_dict(),w.get_state(),rng_state()))
    wrong={**ident,'arm':'changed'}
    with pytest.raises(ValueError,match='before checkpoint deserialization'):restore(tmp_path/'new.pt',proof,wrong,u,v,w)


def test_clean_heads_do_not_depend_on_failed_robustness(tmp_path,monkeypatch):
    c={**ex.config(),'tasks':['single'],'seeds':[123],'train_examples':10,'development_examples':5,'batch_size':5,'head_updates':2}
    (tmp_path/'core_v1').mkdir();monkeypatch.setattr(ex,'PRIVATE',tmp_path);monkeypatch.setattr(ex,'config',lambda:c)
    monkeypatch.setattr(ex,'frozen',lambda:{'fixture':'immutable'});monkeypatch.setattr(ex,'update',lambda *a,**kw:None)
    monkeypatch.setattr(ex,'trusted_reference',lambda *args:(make_model('M1'),{'fixture':True}))
    development_y=ex.draw_for('development','single',123,5).y
    monkeypatch.setattr(ex,'predict',lambda m,x,budget:torch.nn.functional.one_hot(development_y[:len(x)],5).float()*5)
    def forbidden(*args):pytest.fail('Track B cannot gate clean A2')
    monkeypatch.setattr(ex,'saved_receipt',forbidden)
    def evaluated(*args,**kwargs):return {'conditions':{'clean':{'raw':{'macro_f1':.9},'localization':{'precision':.5,'coverage':.8}}},'calibration':{'temperature':1.}}
    monkeypatch.setattr(ex,'model_evaluation',evaluated);monkeypatch.setattr(ex,'analyze_predictor',lambda *a,**kw:{'fixture':True})
    monkeypatch.setattr(ex,'finish_track',lambda wp,receipt,*args:receipt)
    result=ex.track_heads(c,Budget(60))
    assert {r['method'] for r in result['rows']}=={'E0','E1','E2','E3'}
    assert len({r['encoder_sha256'] for r in result['rows']})==1
    assert result['severe_robustness_prerequisite'] is False


def test_scope_and_tamper_reject_before_unsafe_open(tmp_path,monkeypatch):
    class Forbidden:
        def __fspath__(self):pytest.fail('unauthorized filesystem access')
    with pytest.raises(ValueError):ex.output(Forbidden(),scope='RESERVED')
    with pytest.raises(ValueError):ex.output(tmp_path,scope='SYNTHETIC_ONLY')
    m=make_model('M1');o=optimizer(m,ex.config());g=torch.Generator();p=tmp_path/'state.pt';proof=seal(p,m,o,g,0,{'new':'fixture'})
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='before checkpoint deserialization'):restore(p,proof,{'new':'fixture'},m,o,g)


def test_frozen_family_fixed_vs_substitution_and_wrong_target():
    # Quantifiers apply to this bounded displayed family, not all full-input clauses.
    ids=np.arange(4);a=(ids&1)>0;b=(ids&2)>0;clauses=[(1,0),(2,0)]
    matrix=[[verify_clause(t,p,n) for p,n in clauses] for t in (a,b)]
    assert all(any(row) for row in matrix) and not any(all(row[j] for row in matrix) for j in range(2))
    m=make_model('M1').eval();d=generate(5,8731);f,r=m.encode_candidates(d.x[:1]);target=int(m(d.x[:1]).argmax())
    first,_=response_table(m,f,r,target)
    # Permuting labels does not change queries or retarget an incorrect prediction.
    d.y=(d.y+1)%5;second,_=response_table(m,f,r,target)
    assert np.array_equal(first,second)


def test_operator_contract_identity_fails_erasure_oracle_can_change_spectrum():
    d=generate(5,7722);other=generate(5,7733);model=make_model('M1').eval()
    row=ex.operator_records(model,d,other,0,ex.config());records={v['operator']:v for v in row['records']}
    assert records['identity']['target_removal_fraction']==0
    assert records['exact_diagnostic_removal']['target_removal_fraction']==1
    assert records['exact_diagnostic_removal']['nontarget_max_residual']<1e-5
    if row['accepted_donors']<2:assert row['accepted_donor_margin_spread'] is None

def test_complete_circuit_audit_fixture_archive_and_target_schema(tmp_path,monkeypatch):
    c={**ex.config(),'audit_generation_examples':25,'audit_per_class':1}
    private=tmp_path/'private';(private/'core_v1').mkdir(parents=True)
    monkeypatch.setattr(ex,'PRIVATE',private);monkeypatch.setattr(ex,'ROOT',tmp_path)
    m=make_model('M1').eval();monkeypatch.setattr(ex,'trusted_reference',lambda *args:(m,{'checkpoint_sha256':'0'*64}))
    result=ex.analyze_predictor(m,'single',123,'P0',c,Budget(60),{'temperature':1.,'scope':'engineering fixture'})
    assert len(result['rows'])==5 and result['head_mask_queries']==5*4*32768
    assert (tmp_path/result['private_circuits_file']).exists()
    for row in result['rows']:
        assert row['original_prediction']==row['target_class']
        assert row['mask_search_size']==32768 and row['exact']
        assert all(circuit['exact_equality'] for circuit in row['circuits'])
        assert all(clause['minimum_margin']>1e-6 for clause in row['signed_summary']['clauses'])

def test_constant_confidence_correlation_null_and_json_safe():
    logits=torch.tensor([1.,2.,0.,-1.,.2]).repeat(1000,1)
    result=ex.reliability(logits,torch.arange(1000)%5)
    assert result['error_confidence_correlation'] is None
    json.dumps(result,allow_nan=False)

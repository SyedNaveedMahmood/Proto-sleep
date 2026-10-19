import copy
import json
import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate
from mist_cove.phase7.robustness import CONFIG,ClippedModel,head_stage,collective,qualifies,controlled_audit
from mist_evidence.runtime import tensor_digest

C=json.loads(CONFIG.read_text())


def test_same_encoder_and_trainable_capacity_with_fixed_clip_buffers():
    torch.manual_seed(9341);a=make_model('M1')
    torch.manual_seed(9341);b=ClippedModel(torch.zeros(18),torch.ones(18))
    assert tensor_digest(a.encoder.state_dict())==tensor_digest(b.encoder.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==3593
    assert b.clip_mean.numel()+b.clip_std.numel()==36
    x=torch.randn(5,1,3000)*100
    f,_=b.encode_candidates(x)
    assert f.shape==(5,14,18) and f.abs().max()<=3
    b(x).sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())


def rows():
    return [{'task':task,'seed':seed,'method':'R1','gate_clauses':{'pass':True,'clauses':{}}} for task in C['tasks'] for seed in C['seeds']]


@pytest.mark.parametrize('mutation',['one_failed','one_missing','duplicate','other_method_only'])
def test_collective_primary_gate_never_best_of_methods(mutation):
    values=rows()
    if mutation=='one_failed':values[0]['gate_clauses']['pass']=False
    elif mutation=='one_missing':values.pop()
    elif mutation=='duplicate':values[-1]=copy.deepcopy(values[0])
    else:
        for value in values:value['method']='R2'
    assert collective(values,C)['status']=='FAIL'
    class Forbidden:
        def __iter__(self):pytest.fail('head work started after failed collective gate')
    result=head_stage(collective(values,C),Forbidden(),{},Forbidden(),Forbidden())
    assert result['fits']==result['updates']==0


def test_strict_competitive_threshold_not_rounded_up():
    item={'evaluation':{'conditions':{k:{'macro_f1':.9,'collapsed':False} for k in ['clean','competitive','added_nuisance']},
                        'pairs':{k:{'appropriate_diagnostic_change':.9,'nuisance_agreement':.9} for k in ['clean','competitive']}}}
    item['evaluation']['conditions']['nuisance_only']={'accuracy_ci99':[.18,.22]}
    assert qualifies(item,C)['pass']
    item['evaluation']['conditions']['competitive']['macro_f1']=.695932
    assert not qualifies(item,C)['pass']


@pytest.mark.parametrize('method',['M1','M2','M3','M2_rms'])
def test_controlled_same_construction_component_audit(method):
    torch.manual_seed(9361);model=make_model(method).eval();draw=generate(10,9361,'multiple','competitive')
    result=controlled_audit(model,draw,0,9361)
    assert result['same_component_intervention_construction']
    for key in ['diagnostic','nondiagnostic']:
        r=result[key]
        assert len(r['full'])==5 and max(abs(v) for v in r['residual'])<1e-5
        assert not r['internal_paths_are_physiological']
    assert len(result['individual_diagnostic_effects'])==2
    assert all(b-a==200 for a,b in result['control_intervals'])
    assert result['donor_robustness']=='NOT TESTED'

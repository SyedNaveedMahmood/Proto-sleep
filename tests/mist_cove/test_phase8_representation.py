import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import encode,head,oracle_mask
from mist_cove.phase8.pooling import pool,TrimmedModel
from mist_cove.phase8.representation import compare
from mist_cove.phase8.generator import intervene

@pytest.mark.parametrize('method',['mean','max','meanmax','trimmed','bounded','oracle'])
def test_pooling_exact_geometry_finite_gradient(method):
    f=torch.randn(5,14,18,requires_grad=True);mask=torch.ones(5,14,dtype=torch.bool)
    p=pool(f,method,mask);assert p.shape==(5,36);p.square().sum().backward();assert torch.isfinite(f.grad).all()

def test_trimmed_ignores_two_extreme_candidates_without_oracle():
    f=torch.zeros(5,14,18);f[:,0]=100;f[:,1]=200
    assert not pool(f,'trimmed').any() and pool(f,'meanmax').max()==200

def test_parameter_identity_encoder_and_accounting():
    torch.manual_seed(923);a=make_model('M1');torch.manual_seed(923);b=TrimmedModel()
    assert sum(p.numel() for p in b.encoder.parameters())==3408
    assert sum(p.numel() for p in b.parameters())==3593
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=generate(5,922).x;ev=b(x,return_evidence=True)
    torch.testing.assert_close(ev.logits,ev.background_logits+ev.contributions.sum(1))
    f,r=encode(b,x);torch.testing.assert_close(b(x),head(b,pool(f,'trimmed')))

def test_representation_groups_overlap_and_null_safe():
    a=generate(5,922);b=intervene(a,'overlapping',1.);m=make_model('M1');f,_=encode(m,a.x);g,_=encode(m,b.x)
    report=compare(m,a,b,f,g)
    assert report['mixed']['candidate_count']>0 and report['diagnostic_drift']>0
    with pytest.raises(ValueError):pool(f,'oracle',torch.zeros(5,14,dtype=torch.bool))

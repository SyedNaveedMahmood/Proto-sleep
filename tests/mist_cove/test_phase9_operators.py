import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase9.operators import atom_to_candidate_mask,internal_logits,margin,sample_query_masks,component_contract
from mist_cove.phase6.generator import generate

@pytest.mark.parametrize('j',range(15))
def test_overlap_closure(j):
    atoms=torch.ones(1,15,dtype=torch.bool);atoms[:,j]=False
    expected=torch.ones(1,14,dtype=torch.bool)
    if j>0:expected[:,j-1]=False
    if j<14:expected[:,j]=False
    assert torch.equal(atom_to_candidate_mask(atoms),expected)

@pytest.mark.parametrize('kind',['M1','M2','M3','M2_rms'])
def test_native_full_identity_and_hidden_sample_noninterference(kind):
    torch.manual_seed(927);model=make_model(kind).eval();x=generate(10,33871).x
    atoms=torch.ones(10,15,dtype=torch.bool);atoms[:,3:7]=False
    changed=x.clone();changed[:,:,600:1400]=torch.randn(10,1,800)*10000
    f,r=model.encode_candidates(x);g,s=model.encode_candidates(changed)
    full=torch.ones(10,15,dtype=torch.bool)
    assert torch.equal(model(x),internal_logits(model,f,r,full))
    assert torch.equal(internal_logits(model,f,r,atoms),internal_logits(model,g,s,atoms))
    # Candidate RMS is withheld too; an unrestricted global RMS would change.
    assert not torch.equal(x.square().mean(-1),changed.square().mean(-1))
    assert torch.equal(internal_logits(model,f,r,torch.zeros_like(atoms)),model.background_head.bias.expand(10,-1))


def test_class_margin_shift_invariant_and_two_query_axis():
    m=make_model('M1');d=generate(10,18391);f,r=m.encode_candidates(d.x);a=sample_query_masks(10)
    assert a.shape==(10,2,15) and not atom_to_candidate_mask(a).all(-1).any()
    assert atom_to_candidate_mask(a).any(-1).all()
    l=internal_logits(m,f,r,a);assert l.shape==(10,2,5)
    torch.testing.assert_close(margin(l,2),margin(l+17,2),atol=2e-6,rtol=1e-5)

@pytest.mark.parametrize('target',['identity','diagnostic','nuisance'])
def test_component_intervention_same_semantics(target):
    d=generate(5,38191,'multiple','competitive');changed,record=component_contract(d,0,target)
    assert record['nontarget_max_residual']<1e-5
    if target=='identity':assert not record['target_present'] and torch.equal(changed,d.x[:1])
    if target=='diagnostic':assert torch.equal(changed,d.base[:1])

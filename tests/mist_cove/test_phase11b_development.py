import torch
from mist_cove.phase11b.development import ReliabilityLedger,local_hf,preservation_loss

def test_additive_ledger_identity_and_hidden_local_independence():
    torch.manual_seed(19);model=ReliabilityLedger().eval();x=torch.randn(3,1,3000);atoms=torch.tensor([1,4,7]);z=local_hf(x,atoms)
    a=model(x);b=model(z)
    torch.testing.assert_close(a['logits'],model.bias+a['global']+a['contributions'].sum(1),atol=0,rtol=0)
    untouched=torch.ones(3,15,dtype=torch.bool);untouched[torch.arange(3),atoms]=False
    torch.testing.assert_close(a['contributions'][untouched],b['contributions'][untouched],atol=0,rtol=0)
    assert preservation_loss(a,b,atoms)>=0
    assert not torch.equal(a['global'],b['global'])

def test_class_blind_scalar_routing_and_matched_base_initialization():
    torch.manual_seed(123);m0=ReliabilityLedger(False)
    torch.manual_seed(123);m1=ReliabilityLedger(True)
    for (ka,a),(kb,b) in zip(m0.named_parameters(),m1.named_parameters()):
        assert ka==kb and torch.equal(a,b)
    x=torch.randn(2,1,3000)
    assert m1(x)['reliability'].shape==(2,15,1)
    assert torch.equal(m0(x)['reliability'],torch.ones(2,15,1))

def test_preservation_is_local_target_not_full_response_consistency():
    clean={'contributions':torch.ones(1,15,5,requires_grad=True)}
    changed={'contributions':torch.ones(1,15,5,requires_grad=True)};atoms=torch.tensor([0])
    loss=preservation_loss(clean,changed,atoms);loss.backward()
    assert clean['contributions'].grad is None
    assert changed['contributions'].grad[0,0].abs().sum()>0
    assert changed['contributions'].grad[0,1:].abs().sum()==0

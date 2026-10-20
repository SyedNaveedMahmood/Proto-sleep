import pytest
import torch
from mist_cove.phase10.mock_models import make_mock, mock_input
from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase10.source_availability import aggregate_embeddings


def test_partial_pooling_fixed_denominator_negative_max_and_empty():
    embed=torch.tensor([[[-2.,1.],[-3.,4.],[8.,-9.]]])
    masks=torch.tensor([[True,True,False],[False,False,False],[True,True,True]])
    mean,max_=aggregate_embeddings(embed,masks)
    torch.testing.assert_close(mean[0,0],torch.tensor([-5/3,5/3]))
    assert torch.equal(max_[0,0],torch.tensor([-2.,4.]))
    assert torch.equal(mean[1],torch.zeros_like(mean[1]))
    assert torch.equal(max_[1],torch.zeros_like(max_[1]))
    assert torch.equal(mean[2],embed.mean(1)) and torch.equal(max_[2],embed.max(1).values)


def test_cached_interface_preserves_parameters_and_trace():
    model=make_mock().network;before={n:(id(p),p.detach().clone(),p.requires_grad) for n,p in model.named_parameters()}
    adapter=SourceAdapter(model);x=mock_input(n=2);cache=adapter.cache(x)
    masks=torch.zeros(4,15,dtype=torch.bool);masks[0]=True;masks[1,0]=True;masks[2,::2]=True
    trace=adapter.features(cache,masks)
    assert adapter.logits(cache,masks).shape==(4,2,5)
    assert trace.features.shape==(4,2,209) and trace.feature_availability.shape==(4,209)
    assert trace.valid_window_counts[0].tolist()==[59,57,53]
    assert not trace.feature_availability[3].any()
    assert torch.equal(trace.features[3],torch.zeros_like(trace.features[3]))
    assert torch.equal(trace.features[1:,:,-17:],torch.zeros_like(trace.features[1:,:,-17:]))
    assert cache.nbytes==2*(169*32+17)*4
    for n,p in model.named_parameters():
        i,value,grad=before[n];assert id(p)==i and torch.equal(p,value) and p.requires_grad==grad


@pytest.mark.parametrize('seed',[123,456,789])
@pytest.mark.parametrize('summary,amplitude',[(True,True),(True,False),(False,False)])
def test_full_visibility_native_identity(seed,summary,amplitude):
    model=make_mock(seed,use_summary=summary,use_amplitude=amplitude).network
    adapter=SourceAdapter(model);x=mock_input(n=4);x[0]=0;x[1]=3.25;x[2,0,199:202]=12.
    cache=adapter.cache(x);mask=torch.ones(1,15,dtype=torch.bool)
    assert torch.equal(adapter.features(cache,mask).features[0],model.encode(x))
    native=adapter.native(x);actual=adapter.logits(cache,mask)[0]
    torch.testing.assert_close(actual,native,atol=1e-6,rtol=1e-6)
    assert torch.equal(actual,native)
    assert torch.equal(actual,adapter.logits(cache,mask,neutral=.01)[0])


def test_unused_bank_values_and_trace_tamper():
    model=make_mock().network;adapter=SourceAdapter(model);x=mock_input(n=1);expected=adapter.native(x)
    with torch.no_grad():
        for bank in model.banks:bank.waveforms.add_(9.)
    with pytest.raises(ValueError,match='identity'):adapter.native(x)
    new=SourceAdapter(model)
    assert torch.equal(expected,new.native(x))
    cache=new.cache(x);cache.local_embeddings[0].add_(1.)
    with pytest.raises(ValueError,match='tampered'):new.logits(cache,torch.ones(1,15,dtype=torch.bool))


def test_unreviewed_model_geometry_and_modes_rejected():
    model=make_mock().network;model.train()
    with pytest.raises(ValueError):SourceAdapter(model)
    model.eval();model.banks[0].waveforms=torch.randn(16,200)
    with pytest.raises(ValueError,match='geometry'):SourceAdapter(model)

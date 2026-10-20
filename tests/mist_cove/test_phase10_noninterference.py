import pytest
import torch
from mist_cove.phase10.mock_models import make_mock,mock_input
from mist_cove.phase10.source_adapter import SourceAdapter


def hidden_variant(x,mask,kind):
    hidden=(~mask).repeat_interleave(200)[None,None]
    gen=torch.Generator().manual_seed(44);t=torch.arange(3000)/100
    noise=torch.randn(x.shape,generator=gen)
    if kind=='random':change=noise*8
    elif kind=='impulses':
        change=torch.zeros_like(x);change[:,:,::7]=1e4;change[:,:,199::200]=-1e4
    elif kind=='sinusoid':change=(300*torch.sin(2*torch.pi*37*t))[None,None].expand_as(x)
    elif kind=='rms':change=x*1000
    elif kind=='dc':change=torch.full_like(x,10000.)
    elif kind=='broadband':change=noise*10000
    elif kind=='boundary':
        change=torch.zeros_like(x);change[:,:,::200]=9000;change[:,:,199::200]=-7000
    else:raise ValueError(kind)
    return torch.where(hidden,x+change,x)


@pytest.mark.parametrize('kind',['random','impulses','sinusoid','rms','dc','broadband','boundary'])
@pytest.mark.parametrize('mask_id',[0,1,1<<7,0b010101010101010,32767 ^ (1<<7)])
def test_every_retained_path_noninterference(kind,mask_id):
    mask=((torch.tensor(mask_id)>>torch.arange(15))&1).bool();x=mock_input(n=2);y=hidden_variant(x,mask,kind)
    assert torch.equal(x[:,:,mask.repeat_interleave(200)],y[:,:,mask.repeat_interleave(200)])
    adapter=SourceAdapter(make_mock().network);a=adapter.cache(x);b=adapter.cache(y)
    for neutral in (0.,.01):
        fa=adapter.features(a,mask[None],neutral=neutral);fb=adapter.features(b,mask[None],neutral=neutral)
        assert torch.equal(fa.features,fb.features)
        assert torch.equal(fa.feature_availability,fb.feature_availability)
        assert torch.equal(adapter.logits(a,mask[None],neutral=neutral),adapter.logits(b,mask[None],neutral=neutral))
        for ea,eb,valid in zip(a.local_embeddings,b.local_embeddings,fa.window_visibility):
            assert torch.equal(ea[:,valid[0]],eb[:,valid[0]])


def test_broken_global_bypass_detected_by_path_checker():
    adapter=SourceAdapter(make_mock().network);x=mock_input(n=1);mask=torch.zeros(15,dtype=torch.bool);mask[2]=True
    a=adapter.cache(x);b=adapter.cache(hidden_variant(x,mask,'dc'))
    assert torch.equal(adapter.features(a,mask[None]).features,adapter.features(b,mask[None]).features)
    # Deliberately broken implementation retains native global summary/amplitude.
    assert not torch.equal(a.global_features,b.global_features)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(a.global_features,b.global_features,atol=1e-6,rtol=1e-6)


def test_invalid_masks_neutral_and_foreign_cache_rejected():
    a=SourceAdapter(make_mock().network);cache=a.cache(mock_input(n=1))
    for mask in [torch.ones(1,15),torch.ones(15,dtype=torch.bool),torch.zeros(0,15,dtype=torch.bool)]:
        with pytest.raises(ValueError):a.logits(cache,mask)
    with pytest.raises(ValueError):a.logits(cache,torch.ones(1,15,dtype=torch.bool),neutral=.5)
    b=SourceAdapter(make_mock(456).network)
    with pytest.raises(ValueError,match='identity'):b.logits(cache,torch.ones(1,15,dtype=torch.bool))

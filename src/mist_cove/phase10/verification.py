"""Measured mock correctness and explicit scientific dependency gates."""
import torch
from .mock_models import make_mock,mock_input
from .source_adapter import SourceAdapter


def errors(a,b):
    difference=(a-b).abs()
    return {'maximum_absolute':float(difference.max()),
            'maximum_relative_reference_floor_1e_12':float((difference/b.abs().clamp_min(1e-12)).max()),
            'bitwise_equal':bool(torch.equal(a,b))}


def require_gates(receipt, names):
    if any(receipt.get(name,{}).get('status')!='PASS' for name in names):
        raise ValueError('required scientific dependency gate failed or absent')


@torch.no_grad()
def identity_probe():
    rows=[]
    for seed in (123,456,789):
        for summary,amplitude in ((True,True),(True,False),(False,False)):
            model=make_mock(seed,use_summary=summary,use_amplitude=amplitude).network
            adapter=SourceAdapter(model);x=mock_input(n=4);x[0]=0;x[1]=3.25;x[2,0,199:202]=12.
            cache=adapter.cache(x);mask=torch.ones(1,15,dtype=torch.bool)
            f=adapter.features(cache,mask).features[0];nf=model.encode(x)
            actual=adapter.logits(cache,mask)[0];native=adapter.native(x)
            torch.testing.assert_close(f,nf,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(actual,native,atol=1e-6,rtol=1e-6)
            rows.append({'seed':seed,'summary':summary,'amplitude':amplitude,'examples':len(x),
                         'features':errors(f,nf),'logits':errors(actual,native),
                         'parameters':sum(p.numel() for p in model.parameters()),
                         'historical_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                         'new_trainable_parameters':0,'cache_bytes_per_example':cache.nbytes//len(x)})
    return {'status':'PASS','atol':1e-6,'rtol':1e-6,'rows':rows,'real_artifacts_opened':0}


def hidden_change(x,mask,kind):
    gen=torch.Generator().manual_seed(2048)
    noise=torch.randn(x.shape,generator=gen);t=torch.arange(3000)/100
    if kind=='random':change=noise*8
    elif kind=='impulses':
        change=torch.zeros_like(x);change[:,:,::7]=1e4;change[:,:,199::200]=-1e4
    elif kind=='sinusoid':change=(300*torch.sin(2*torch.pi*37*t))[None,None].expand_as(x)
    elif kind=='rms':change=x*1000
    elif kind=='dc':change=torch.full_like(x,10000.)
    elif kind=='broadband':change=noise*10000
    elif kind=='boundary':
        change=torch.zeros_like(x);change[:,:,::200]=9000;change[:,:,199::200]=-7000
    else:raise ValueError('unknown synthetic probe')
    hidden=(~mask).repeat_interleave(200)[None,None]
    return torch.where(hidden,x+change,x)


@torch.no_grad()
def noninterference_probe():
    rows=[];negative=[]
    for seed in (123,456,789):
        adapter=SourceAdapter(make_mock(seed).network);x=mock_input(n=2);a=adapter.cache(x)
        for mask_id in (0,1,1<<7,21845,32767^(1<<7),32767):
            mask=((torch.tensor(mask_id)>>torch.arange(15))&1).bool()
            for kind in ('random','impulses','sinusoid','rms','dc','broadband','boundary'):
                y=hidden_change(x,mask,kind);b=adapter.cache(y)
                for neutral in (0.,.01):
                    fa=adapter.features(a,mask[None],neutral=neutral);fb=adapter.features(b,mask[None],neutral=neutral)
                    la=adapter.logits(a,mask[None],neutral=neutral);lb=adapter.logits(b,mask[None],neutral=neutral)
                    torch.testing.assert_close(fa.features,fb.features,atol=1e-6,rtol=1e-6)
                    torch.testing.assert_close(la,lb,atol=1e-6,rtol=1e-6)
                    for ea,eb,v in zip(a.local_embeddings,b.local_embeddings,fa.window_visibility):
                        torch.testing.assert_close(ea[:,v[0]],eb[:,v[0]],atol=1e-6,rtol=1e-6)
                    rows.append({'seed':seed,'mask_id':mask_id,'perturbation':kind,'neutral':neutral,
                                 'features':errors(fa.features,fb.features),'logits':errors(la,lb)})
                if mask_id!=32767 and kind=='dc':
                    diff=float((a.global_features-b.global_features).abs().max())
                    if diff<=1e-6:raise ValueError('broken global-feature negative control did not expose leakage')
                    negative.append({'seed':seed,'mask_id':mask_id,'deliberately_broken_global_path_absolute_difference':diff,'detected':True})
    return {'status':'PASS','atol':1e-6,'rtol':1e-6,'comparisons':len(rows),'rows':rows,
            'broken_path_controls':negative,'real_artifacts_opened':0}

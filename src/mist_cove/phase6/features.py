"""Class C leakage features and explicitly allowed Class B diagnostic proxies."""
import torch


def signal_view(draw,name):
    if name=='full':return draw.x
    if name=='oracle':return draw.diagnostic
    if name=='removed':return draw.base
    if name=='background':return draw.background+draw.noise
    if name=='nuisance':return draw.nuisance+draw.noise
    if name=='artifact':return draw.artifact
    raise ValueError('unknown waveform control')


def summaries(x):
    return torch.stack((x.mean(-1),x.square().mean(-1).sqrt(),x.std(-1,unbiased=False),x.amin(-1),x.amax(-1)),-1)


def spectrum(x):return torch.log1p(torch.fft.rfft(x.double()).abs()/20).float()


def features(draw,name):
    if name in {'background','nuisance','artifact'}:
        x=signal_view(draw,name)[:,0]
        return torch.cat((spectrum(x),summaries(x)),1)
    if name=='position':
        return torch.tensor([[*r['diagnostic_onsets'],*r['nuisance_onsets'],r['artifact_onset'],float(r['mode']=='complementary'),r['generation_order']/len(draw.y)] for r in draw.latents],dtype=torch.float32)
    if name=='full_magnitude':return spectrum(draw.x[:,0])
    if name in {'diagnostic_amplitude','diagnostic_spectrum'}:
        rows=[];length=5 if name=='diagnostic_amplitude' else 101
        for i,truth in enumerate(draw.intervals):
            # Ordered by physical time. Oracle intervals are audit privileges,
            # never inputs to blind full-waveform M1.
            values=[(summaries if name=='diagnostic_amplitude' else spectrum)(draw.x[i,0,lo:hi]) for lo,hi in sorted(truth)]
            rows.append(torch.cat(values+[torch.zeros(length)]*(2-len(values))))
        return torch.stack(rows)
    raise ValueError('unknown feature control')

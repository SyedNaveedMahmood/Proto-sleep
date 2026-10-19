"""Restricted information maps and transform-consistency measurements."""
import numpy as np
import torch
from mist_cove.phase4.synthetic import templates


def patches(x):
    if x.ndim != 3 or x.shape[1:] != (1,3000) or not torch.isfinite(x).all():
        raise ValueError('finite original physical epoch geometry required')
    return x[:,0].unfold(-1,400,200)


def zero_phase(x):
    mean = x.mean(-1,keepdim=True)
    return torch.fft.irfft(torch.fft.rfft(x-mean).abs().to(torch.complex64),n=x.shape[-1])+mean


def transform_diagnostics(x):
    p = patches(x); z = zero_phase(p); coherent = patches(zero_phase(x))
    def overlap(a): return float((a[:,:-1,200:]-a[:,1:,:200]).square().mean().sqrt())
    magnitude = torch.fft.rfft(p-p.mean(-1,keepdim=True)).abs()
    error = (torch.fft.rfft(z-z.mean(-1,keepdim=True)).abs()-magnitude).abs()
    sym = (z-z.mean(-1,keepdim=True))
    return {'original_overlap_rms':overlap(p),'patch_zero_overlap_rms':overlap(z),
        'coherent_zero_overlap_rms':overlap(coherent),
        'patch_magnitude_relative_max_error':float(error.max()/magnitude.max().clamp_min(1e-12)),
        'patch_zero_circular_even_rms':float((sym-torch.roll(sym.flip(-1),1,-1)).square().mean().sqrt()),
        'original_wrap_jump_rms':float((x[...,0]-x[...,-1]).square().mean().sqrt()),
        'coherent_wrap_jump_rms':float((zero_phase(x)[...,0]-zero_phase(x)[...,-1]).square().mean().sqrt())}


def _mag(x):
    return torch.log1p(torch.fft.rfft(x.double()).abs()/20).float()


def _summaries(x):
    mean=x.mean(-1);var=x.var(-1,unbiased=False);rms=x.square().mean(-1).sqrt()
    return torch.stack((mean,var.sqrt(),rms,x.amin(-1),x.amax(-1),x.abs().amax(-1)),-1).flatten(1)


def _pad(v, length):
    if len(v)>length: raise ValueError('latent vector exceeds frozen capacity')
    return list(v)+[0.]*(length-len(v))


def restricted(draw, name):
    x=draw.x; p=patches(x)
    if name in {'patch_magnitude','permuted_patch'}: return _mag(p).flatten(1)
    if name=='epoch_magnitude': return _mag(x[:,0])
    if name=='diagnostic_epoch_magnitude': return _mag(draw.diagnostic[:,0])
    if name=='nuisance_magnitude': return _mag(draw.nuisance[:,0])
    if name in {'isolated_magnitude','scaled_magnitude'}:
        bank=templates();vectors=[]
        for row in draw.latent:
            codes=row['diagnostic_template_codes']
            # Mean over code factors, not class identifiers in the decoder.
            waves=torch.stack([bank[c]*(a if name=='scaled_magnitude' else 1.) for c,a in zip(codes,row['diagnostic_amplitudes'])])
            vectors.append(_mag(waves).mean(0))
        return torch.stack(vectors)
    if name=='amplitude': return torch.cat((_summaries(x),_summaries(p)),1)
    if name=='envelope_boundary':
        envelope=x[:,0].abs().reshape(len(x),30,100).mean(-1)
        boundary=torch.cat((p[...,0],p[...,-1],p.diff(dim=-1).abs().amax(-1)),1)
        return torch.cat((envelope,boundary,_summaries(p)),1)
    if name=='position':
        return torch.tensor([[*_pad(sum(row['intervals'],[]),12),*_pad(sum(row['nuisance_intervals'],[]),8),row['generation_order']/len(x)] for row in draw.latent],dtype=torch.float32)
    if name=='nuisance_latent':
        return torch.tensor([[row['background_amplitude'],row['background_phase'],*_pad(row['diagnostic_amplitudes'],6),*_pad(row['nuisance_amplitudes'],4),*_pad(row['nuisance_codes'],4)] for row in draw.latent],dtype=torch.float32)
    raise ValueError('unknown restricted feature set')


def class_contrasts(values,y):
    values=values.double();means=torch.stack([values[y==c].mean(0) for c in range(5)])
    return {'feature_dimensions':values.shape[1],
        'max_class_mean_range':float((means.amax(0)-means.amin(0)).max()),
        'rms_class_mean_range':float((means.amax(0)-means.amin(0)).square().mean().sqrt()),
        'global_mean':float(values.mean()),'global_std':float(values.std(unbiased=False))}


def signal_audit(draw):
    x,d,u=draw.x[:,0].double(),draw.diagnostic[:,0].double(),draw.nuisance[:,0].double()
    X,D,U=(torch.fft.rfft(v) for v in (x,d,u))
    residual=X.abs().square()-D.abs().square()-U.abs().square()-2*(D*U.conj()).real
    f=torch.fft.rfftfreq(3000,d=.01)
    bands=torch.stack([X[:,(f>=lo)&(f<hi)].abs().square().sum(1) for lo,hi in [(0,4),(4,8),(8,12),(12,16),(16,50.01)]],1)
    boundaries=[(lo,hi,j,j+400) for row in draw.latent for lo,hi in row['intervals'] for j in range(0,2601,200) if lo<j+400 and hi>j]
    return {'transform':transform_diagnostics(draw.x),'signal_reconstruction_max_abs':float((x-d-u).abs().max()),
        'event_patch_intersections':len(boundaries),
        'boundary_truncated_event_patch_intersections':sum(not (a<=lo and hi<=b) for lo,hi,a,b in boundaries),
        'power_identity_relative_max_residual':float(residual.abs().max()/X.abs().square().max().clamp_min(1e-12)),
        'interference_absolute_power_fraction':float((2*(D*U.conj()).real).abs().sum()/X.abs().square().sum().clamp_min(1e-12)),
        'band_power':class_contrasts(bands,draw.y),
        'class_conditioned':{name:class_contrasts(restricted(draw,name),draw.y) for name in ['epoch_magnitude','patch_magnitude','amplitude','envelope_boundary','position','nuisance_latent']}}


def template_diagnostics():
    waves=templates()[:5].double()
    def contrast(a,n):
        mag=torch.fft.rfft(a,n=n).abs()
        return float((mag.amax(0)-mag.amin(0)).max()/mag.max().clamp_min(1e-12))
    embedded=torch.zeros(5,3000,dtype=torch.float64);embedded[:,100:300]=waves
    shifted=torch.roll(embedded,57,-1)
    return {'isolated_200_magnitude_relative_class_range':contrast(waves,200),
        'zero_padded_400_magnitude_relative_class_range':contrast(waves,400),
        'zero_padded_3000_magnitude_relative_class_range':contrast(waves,3000),
        'cropped_100_padded_400_magnitude_relative_class_range':contrast(waves[:,:100],400),
        'complete_event_shift_relative_magnitude_error':float((torch.fft.rfft(embedded).abs()-torch.fft.rfft(shifted).abs()).abs().max()/torch.fft.rfft(embedded).abs().max()),
        'shared_rms_max_difference':float((waves.square().mean(1).sqrt()-waves[0].square().mean().sqrt()).abs().max())}

"""Instrumented historical replay and a separate continuous harmonic code."""
from dataclasses import dataclass
import numpy as np
import torch
from mist_cove.phase4.synthetic import templates


@dataclass
class Draw:
    x: torch.Tensor
    y: torch.Tensor
    diagnostic: torch.Tensor
    nuisance: torch.Tensor
    latent: list
    family: str
    seed: int


def _labels(n, rng, override):
    if n <= 0 or n % 5:
        raise ValueError('positive balanced-five draw required')
    labels = np.arange(n) % 5
    rng.shuffle(labels)
    if override is not None:
        other = np.asarray(override)
        if other.shape != labels.shape or not np.isin(other, range(5)).all():
            raise ValueError('invalid counterfactual labels')
        labels = other.astype(int)
    return labels


def historical(n, seed, task='substitutable', condition='clean', *, labels=None):
    """Replay frozen v2 in its original floating-point addition and RNG order.

    The separate nuisance array never subtracts a class-coded signal, avoiding
    label-dependent cancellation roundoff in a purported nuisance-only control.
    """
    if task not in {'substitutable','complementary'} or condition not in {'train','clean','competitive','overlap','erased','abrupt_artifact'}:
        raise ValueError('unsupported historical audit condition/task')
    rng = np.random.default_rng(seed)
    ys = _labels(n, rng, labels)
    bank = templates()
    x = torch.randn(n,1,3000,generator=torch.Generator().manual_seed(seed))*.5
    nuisance = x.clone(); diagnostic = torch.zeros_like(x)
    t = torch.arange(3000)/100
    latents = []
    for b, y in enumerate(ys):
        amp, phase = float(rng.uniform(0,2)), float(rng.uniform(0,6.28))
        slow = amp*torch.sin(2*torch.pi*.7*t+phase)
        x[b,0] += slow; nuisance[b,0] += slow
        factors = [int(y)] if task == 'substitutable' else [int(y),(int(y)+1)%5]
        if condition == 'train' and task == 'substitutable':
            factors *= int(rng.integers(1,3))
        elif task == 'substitutable':
            factors *= 2
        sites = rng.choice(np.arange(100,2800,200),size=12,replace=False).tolist()
        parts = []; scales = []
        for j, code in enumerate(factors):
            lo = int(sites.pop()+rng.integers(-30,31))
            if condition == 'overlap' and j == 0: lo = min(lo,2700)
            if condition == 'overlap' and j == 1: lo = parts[0][0]+100
            scale = float(rng.uniform(4.,8.)); wave = bank[code]*scale
            if condition != 'erased':
                x[b,0,lo:lo+200] += wave
                diagnostic[b,0,lo:lo+200] += wave
            parts.append([lo,lo+200]); scales.append(scale)
        nsites = []; nscales = []; codes = []
        for _ in range(4 if condition == 'competitive' else 2):
            lo = int(sites.pop()+rng.integers(-30,31)); code = int(rng.integers(5,9))
            scale = float(rng.uniform(*([28.,40.] if condition == 'competitive' else [12.,24.])))
            wave = bank[code]*scale
            x[b,0,lo:lo+200] += wave; nuisance[b,0,lo:lo+200] += wave
            nsites.append([lo,lo+200]); nscales.append(scale); codes.append(code)
        artifact = None
        if condition == 'abrupt_artifact':
            lo = int(rng.integers(100,2800)); a = float(rng.choice([-80.,80.]))
            x[b,0,lo:lo+20] += a; nuisance[b,0,lo:lo+20] += a
            artifact = [lo,lo+20,a]
        latents.append({'class_code':int(y),'diagnostic_template_codes':factors,
            'diagnostic_phase_reference':'frozen_templates_4177_rfft_phase_by_code',
            'diagnostic_amplitudes':scales,'intervals':parts,'nuisance_intervals':nsites,
            'nuisance_codes':codes,'nuisance_amplitudes':nscales,
            'background_amplitude':amp,'background_phase':phase,
            'noise_seed':seed,'noise_stream_sample_offset':b*3000,
            'generation_order':b,'artifact':artifact,'condition':condition,
            'diagnostic_count':len(factors),'overlap':any(max(a[0],c[0])<min(a[1],c[1]) for j,a in enumerate(parts) for c in parts[:j])})
    return Draw(x,torch.tensor(ys,dtype=torch.long),diagnostic,nuisance,latents,'historical_'+task,seed)


def harmonic_component(code, theta, amplitude):
    t = torch.arange(3000,dtype=torch.float64)/100
    return (amplitude*(torch.cos(2*torch.pi*2*t+theta)+.7*torch.cos(2*torch.pi*4*t+2*theta+2*np.pi*int(code)/5))).float()


def harmonic(n, seed, condition='clean', *, labels=None):
    if condition not in {'train','clean','competitive','erased','artifact'}:
        raise ValueError('unknown harmonic condition')
    rng = np.random.default_rng(seed)
    ys = _labels(n,np.random.default_rng(np.random.SeedSequence([seed,971])),labels)
    noise = torch.randn(n,1,3000,generator=torch.Generator().manual_seed(seed))*.5
    x = noise.clone(); nuisance = noise.clone(); diagnostic = torch.zeros_like(x)
    t = torch.arange(3000,dtype=torch.float64)/100
    latents = []
    for b,y in enumerate(ys):
        theta = float(rng.uniform(0,2*np.pi)); amp = float(rng.uniform(8.,12.))
        namp = float(rng.uniform(4.,8.)); nphase = float(rng.uniform(0,2*np.pi))
        # Draw artifact variables for every condition: branches consume no RNG.
        onset = int(rng.integers(100,2780)); polarity = float(rng.choice([-80.,80.]))
        if condition == 'competitive': namp = 40.
        d = harmonic_component(int(y),theta,amp)
        u = (namp*torch.cos(2*torch.pi*13*t+nphase)).float()
        nuisance[b,0] += u
        if condition == 'artifact': nuisance[b,0,onset:onset+20] += polarity
        if condition != 'erased': diagnostic[b,0] = d
        x[b] = nuisance[b]+diagnostic[b]
        latents.append({'class_code':int(y),'diagnostic_template_codes':[int(y)],
            'diagnostic_phase':2*np.pi*int(y)/5,'global_phase':theta,
            'diagnostic_amplitudes':[amp],'intervals':[[0,3000]],'diagnostic_count':1,
            'nuisance_codes':[13],'nuisance_amplitudes':[namp],'nuisance_intervals':[[0,3000]],
            'background_amplitude':namp,'background_phase':nphase,
            'noise_seed':seed,'noise_stream_sample_offset':b*3000,'generation_order':b,
            'artifact':[onset,onset+20,polarity] if condition == 'artifact' else None,
            'overlap':False,'condition':condition})
    return Draw(x,torch.tensor(ys,dtype=torch.long),diagnostic,nuisance,latents,'harmonic_relative_phase_v1',seed)


def relative_phase_oracle(x):
    z = torch.fft.rfft(x[:,0].double())
    phase = torch.angle(z[:,120]*z[:,60].conj().square())
    return torch.remainder(torch.round(phase*5/(2*torch.pi)).long(),5)


def diagnostic_swap(draw, labels):
    if draw.family.startswith('historical_'):
        return historical(len(labels),draw.seed,draw.family.removeprefix('historical_'),draw.latent[0]['condition'],labels=labels)
    return harmonic(len(labels),draw.seed,draw.latent[0]['condition'],labels=labels)


def reverse_pair(draw, other):
    """Keep exact diagnostic tensor; replace only label-free nuisance realization."""
    if draw.x.shape != other.x.shape: raise ValueError('pair geometry mismatch')
    return draw.diagnostic+other.nuisance

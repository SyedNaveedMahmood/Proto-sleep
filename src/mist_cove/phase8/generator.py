"""Versioned additive interventions on the unchanged Phase VI construction."""
from dataclasses import replace
import numpy as np
import torch
from mist_cove.phase6.generator import generate, compose, rng

TAXONOMY = ('high_frequency','low_frequency','transient','impulse','broadband',
            'sustained','multiple','correlated','overlapping','disjoint','boundary',
            'inside_candidate','crossing_candidate','weak_diagnostic','competing')
STRENGTHS = (0., .5, 1., 2.)


def intervene(draw, kind, strength=1.):
    if kind not in TAXONOMY or strength not in STRENGTHS:
        raise ValueError('unregistered corruption or strength')
    extra=torch.zeros_like(draw.x);sites=[]
    for i,record in enumerate(draw.latents):
        g=rng(record['seed'],8801,record['index'])
        diagnostic=draw.intervals[i]
        # Site selection depends on geometry, never class or diagnostic frequency.
        length=200
        lo=int(g.integers(0,2801))
        if kind=='overlapping':lo=diagnostic[0][0]
        if kind=='disjoint':
            choices=[j for j in range(0,2801,100) if all(j+200<=a or j>=b for a,b in diagnostic)]
            lo=int(g.choice(choices))
        if kind=='boundary':lo=0 if g.random()<.5 else 2800
        if kind=='inside_candidate':lo=400*int(g.integers(0,7))+100
        if kind=='crossing_candidate':lo=400*int(g.integers(1,7))-100
        if kind=='impulse':length=3
        if kind=='transient':length=20
        if kind=='sustained':lo=0;length=3000
        count=4 if kind in {'multiple','correlated','competing'} else 1
        phase=float(g.uniform(0,2*np.pi));row=[]
        for j in range(count):
            start=lo if j==0 else int(g.integers(0,3001-length))
            t=np.arange(length)/100
            if kind in {'transient','impulse'}:wave=np.ones(length)*float(g.choice([-1,1]))
            elif kind=='broadband':wave=g.standard_normal(length)
            else:
                hz=1. if kind=='low_frequency' else 30.
                angle=phase if kind=='correlated' else float(g.uniform(0,2*np.pi))
                wave=np.sin(2*np.pi*hz*t+angle)
            if length==200:wave=wave*np.sin(np.pi*np.arange(length)/(length-1))**2
            wave=wave/max(float(np.sqrt(np.mean(wave**2))),1e-8)
            amplitude=100. if kind in {'transient','impulse'} else (12. if kind=='sustained' else 40.)
            extra[i,0,start:start+length]+=torch.tensor(wave,dtype=torch.float32)*(amplitude*strength)
            if strength:row.append((start,start+length))
        sites.append(row)
    # A named weak-signal regime is an explicit diagnostic manipulation; labels,
    # frequency, placement and nuisance streams stay fixed across strengths.
    events=draw.events*.2 if kind=='weak_diagnostic' else draw.events
    base=draw.base+extra
    return replace(draw,x=compose(base,events),base=base,events=events,
        nuisance=draw.nuisance+extra,
        nuisance_intervals=[a+b for a,b in zip(draw.nuisance_intervals,sites)],
        condition='phase8_'+kind+'_'+str(strength))


def training_draw(n,seed,task,method):
    if method not in {'B1','A1','T1'}:raise ValueError('unregistered training candidate')
    clean=generate(n,seed,task,'train');legacy=generate(n,seed,task,'competitive')
    x=clean.x.clone()
    # Balanced deterministic strata; assignments independent of labels.
    strata=torch.arange(n)%4
    mild=intervene(clean,'high_frequency',.5)
    strong=legacy if method!='A1' else intervene(clean,'broadband',2.)
    x[strata==1]=mild.x[strata==1]
    x[strata==2]=strong.x[strata==2]
    # Boundary/overlap/disjoint populations all occur without label conditioning.
    artifact_kinds=('impulse','broadband','transient','crossing_candidate','overlapping','disjoint')
    for j,kind in enumerate(artifact_kinds):
        ix=(strata==3)&((torch.arange(n)//4)%len(artifact_kinds)==j)
        draw=intervene(clean,kind,2. if kind=='transient' else 1.)
        x[ix]=draw.x[ix]
    return x,clean.y


def information_loss(draw,condition):
    if condition=='intact':return draw.x
    if condition=='weak':return compose(draw.base,draw.events*.2)
    if condition=='partial':
        keep=torch.ones(len(draw.y),2)
        if draw.task=='multiple':keep[:,0]=0
        else:
            events=draw.events.clone()
            for i,iv in enumerate(draw.intervals):
                a,b=iv[0];events[i,0,0,(a+b)//2:b]=0
            return compose(draw.base,events)
        return compose(draw.base,draw.events,keep)
    if condition in {'removed','high_energy_remaining'}:return draw.base
    if condition=='unrelated':
        return draw.base+intervene(replace(draw,base=torch.zeros_like(draw.base)),'high_frequency',1.).nuisance-draw.nuisance
    if condition=='background':return draw.background+draw.noise
    raise ValueError('unknown information-loss intervention')

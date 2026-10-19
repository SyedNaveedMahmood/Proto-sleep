"""Controlled generated class codes; never clinical sleep-event annotations."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Generated:
    x: torch.Tensor
    y: torch.Tensor
    components: list
    task: str
    condition: str
    seed: int


def templates(seed=4177):
    g=torch.Generator().manual_seed(seed)
    f=torch.fft.rfftfreq(200,d=.01)
    amplitude=torch.exp(-((f-6)/4)**2)+.6*torch.exp(-((f-13)/5)**2)
    amplitude[0]=0;amplitude[-1]=0
    phases=torch.rand(9,len(f),generator=g)*2*torch.pi
    waves=torch.fft.irfft(torch.polar(amplitude.expand_as(phases),phases),n=200)
    # The shared Fourier amplitudes imply common RMS; one shared scale avoids
    # introducing class-specific spectral differences by separate normalization.
    return waves/waves[0].std(unbiased=False)


def generate(n,seed,task='substitutable',condition='clean',cfg=None):
    if n%5 or task not in {'substitutable','complementary'}:
        raise ValueError('balanced five-class generated task required')
    conditions={'train','clean','single','redundant','correlated','competitive','abrupt_artifact','overlap','partial','erased','eventless'}
    if condition not in conditions:raise ValueError('unknown generated condition')
    cfg=cfg or {};rng=np.random.default_rng(seed);g=torch.Generator().manual_seed(seed)
    bank=templates(cfg.get('template_seed',4177));labels=np.arange(n)%5;rng.shuffle(labels)
    x=torch.randn(n,1,3000,generator=g)*cfg.get('noise_uV',.5)
    t=torch.arange(3000)/100
    all_parts=[]
    for b,y in enumerate(labels):
        x[b,0]+=float(rng.uniform(0,2))*torch.sin(2*torch.pi*.7*t+float(rng.uniform(0,6.28)))
        factors=[int(y)] if task=='substitutable' else [int(y),(int(y)+1)%5]
        if condition in {'single','partial'}:factors=[factors[int(rng.integers(len(factors)))]]
        elif condition=='train' and task=='substitutable':factors=factors*int(rng.integers(1,3))
        elif task=='substitutable':factors=factors*2
        if condition=='redundant':factors=([int(y)]*2 if task=='substitutable' else factors*2)
        if condition=='correlated':factors=([int(y)]*6 if task=='substitutable' else factors*3)
        sites=rng.choice(np.arange(100,2800,200),size=12,replace=False).tolist()
        count=4 if condition=='competitive' else 2
        parts=[]
        for j,code in enumerate(factors):
            lo=int(sites.pop()+rng.integers(-30,31))
            if condition=='overlap' and j==1:lo=parts[0]['interval'][0]+100
            low,high=cfg.get('diagnostic_rms_uV',[4.,8.]);scale=float(rng.uniform(low,high))
            if condition=='partial':scale*=.3
            wave=bank[code]*scale
            if condition not in {'erased','eventless'}:x[b,0,lo:lo+200]+=wave
            parts.append({'interval':(lo,lo+200),'wave':wave,'diagnostic':True,'present':condition not in {'erased','eventless'},'factor':code})
        for j in range(count):
            lo=int(sites.pop()+rng.integers(-30,31));code=int(rng.integers(5,9))
            bounds=cfg.get('competitive_rms_uV',[28.,40.]) if condition=='competitive' else cfg.get('nuisance_rms_uV',[12.,24.])
            wave=bank[code]*float(rng.uniform(*bounds))
            if condition!='eventless':x[b,0,lo:lo+200]+=wave
            parts.append({'interval':(lo,lo+200),'wave':wave,'diagnostic':False,'present':condition!='eventless','factor':code})
        if condition=='abrupt_artifact':
            lo=int(rng.integers(100,2800));wave=torch.ones(20)*float(rng.choice([-80.,80.]))
            x[b,0,lo:lo+20]+=wave;parts.append({'interval':(lo,lo+20),'wave':wave,'diagnostic':False,'present':True,'factor':'artifact'})
        if condition=='eventless':x[b]=0
        all_parts.append(parts)
    return Generated(x,torch.tensor(labels,dtype=torch.long),all_parts,task,condition,seed)


def diagnostic_intervals(parts):
    return [p['interval'] for p in parts if p['diagnostic'] and p['present']]


def localization(intervals,truth):
    def iou(a,b):
        intersection=max(0,min(a[1],b[1])-max(a[0],b[0]))
        return intersection/(max(a[1],b[1])-min(a[0],b[0]))
    hits=[any(iou(v,t)>=.25 for t in truth) for v in intervals]
    covered=[any(iou(v,t)>=.25 for v in intervals) for t in truth]
    return {'top1_hit':float(hits[0]) if hits else 0.,'precision':float(np.mean(hits)) if hits else 0.,
            'diagnostic_recall':float(np.mean(covered)) if covered else None,
            'mean_best_iou':float(np.mean([max((iou(v,t) for t in truth),default=0.) for v in intervals])) if intervals else 0.,
            'selected_overlap_fraction':float(np.mean([max(0,min(a[1],b[1])-max(a[0],b[0]))>0 for j,a in enumerate(intervals) for b in intervals[:j]])) if len(intervals)>1 else 0.}

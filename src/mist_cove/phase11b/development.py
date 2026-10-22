"""Compact prospective reliability ledger; no source access in model code."""
import torch
from torch import nn
import torch.nn.functional as F

def local_hf(x,atoms,phase=None):
    """One known corrupted atom per example; no diagnostic-event labels."""
    if x.ndim!=3 or x.shape[1:]!=(1,3000) or atoms.shape!=(len(x),):raise ValueError('physical batch/atom identity')
    if not ((atoms>=0)&(atoms<15)).all():raise ValueError('atom out of range')
    z=x.clone();rms=x.std(-1,unbiased=False).clamp_min(1e-8)
    phase=x.new_zeros(len(x)) if phase is None else phase
    t=torch.arange(200,device=x.device,dtype=x.dtype)/100
    noise=3*rms[:,0,None]*torch.sin(2*torch.pi*35*t[None]+phase[:,None])
    z.reshape(-1,15,200)[torch.arange(len(x),device=x.device),atoms]+=noise
    return z

def features(x):
    atoms=x.reshape(-1,15,200);center=atoms-atoms.mean(-1,keepdim=True);rms=center.square().mean(-1).clamp_min(1e-8).sqrt()
    powers=torch.fft.rfft(center).abs().square();freq=torch.fft.rfftfreq(200,.01,device=x.device);total=powers.sum(-1).clamp_min(1e-8)
    quality=torch.stack((rms.log(),powers[...,freq>=30].sum(-1)/total,atoms.mean(-1).abs(),center.abs().amax(-1)/rms),-1)
    epoch=x[:,0];ec=epoch-epoch.mean(-1,keepdim=True);ep=torch.fft.rfft(ec).abs().square();ef=torch.fft.rfftfreq(3000,.01,device=x.device);et=ep.sum(-1).clamp_min(1e-8)
    global_=torch.stack([ec.square().mean(-1).clamp_min(1e-8).sqrt().log(),epoch.mean(-1)]+[ep[:,(ef>=lo)&(ef<hi)].sum(-1)/et for lo,hi in ((.5,4),(4,8),(8,12),(12,16),(16,30))],-1)
    return atoms,quality,global_

class ReliabilityLedger(nn.Module):
    def __init__(self,routing=True):
        super().__init__();self.routing=routing
        self.encoder=nn.Sequential(nn.Conv1d(1,8,9,stride=2),nn.ELU(),nn.Conv1d(8,16,5,stride=2),nn.ELU(),nn.AdaptiveAvgPool1d(1))
        self.local=nn.Linear(16,5,bias=False)
        self.global_head=nn.Sequential(nn.Linear(7,16),nn.ELU(),nn.Linear(16,5,bias=False));self.bias=nn.Parameter(torch.zeros(5))
        self.reliability=nn.Sequential(nn.Linear(4,8),nn.ELU(),nn.Linear(8,1),nn.Sigmoid())
        if not routing:self.reliability.requires_grad_(False)
    def forward(self,x):
        atoms,quality,global_features=features(x);embedding=self.encoder(atoms.reshape(-1,1,200)).flatten(1).reshape(-1,15,16)
        evidence=3*torch.tanh(self.local(embedding))
        reliability=self.reliability(quality) if self.routing else torch.ones_like(evidence[...,:1])
        contributions=reliability*evidence/15;global_logits=self.global_head(global_features)
        logits=self.bias+global_logits+contributions.sum(1)
        return {'logits':logits,'global':global_logits,'contributions':contributions,'reliability':reliability,'evidence':evidence}

def preservation_loss(clean,changed,atoms):
    # A SINGLE local contribution target: unchanged regions retain their clean
    # signed contribution; the generated nuisance region contributes zero.
    # No availability-mask response query or full-logit consistency appears.
    target=clean['contributions'].detach().clone()
    target[torch.arange(len(atoms),device=atoms.device),atoms]=0
    return F.mse_loss(changed['contributions']*15,target*15)

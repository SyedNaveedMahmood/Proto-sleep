"""Patch-local learned waveform encoder and capacity-matched decision heads."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from mist_cove.evidence import Evidence, wave_digest


@dataclass(frozen=True)
class MorphConfig:
    method: str = 'M2'
    window: int = 400
    stride: int = 200
    event_budget: int = 4
    event_scale: float = 8.
    radius: int = 0
    use_crf: bool = False

    def __post_init__(self):
        if self.method not in {'M1', 'M2', 'M3', 'M2_rms', 'M1_phase'}:
            raise ValueError('unknown Phase IV method')
        if (self.window, self.stride, self.event_budget, self.event_scale, self.radius, self.use_crf) != (400,200,4,8.,0,False):
            raise ValueError('frozen Phase IV geometry/budget changed')

    def dictionary(self):
        return {'version':'mist-cove-morphology-v1', **asdict(self)}


class LocalEncoder(nn.Module):
    dimension = 18

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(2,4,k,stride=4,padding=k//2) for k in [9,25,75]])
        self.middle = nn.Conv1d(12,12,7,stride=4,padding=3)
        self.final = nn.Conv1d(12,16,5,stride=2,padding=2)
        self.project = nn.Linear(32,16)

    def forward(self, physical_patches):
        mean = physical_patches.mean(-1,keepdim=True)
        rms = (physical_patches-mean).square().mean(-1,keepdim=True).sqrt()
        channels = torch.stack(((physical_patches-mean)/rms.clamp_min(.1),physical_patches/20),dim=-2)
        batch, candidates, _, length = channels.shape
        x = channels.reshape(-1,2,length)
        h = torch.cat([torch.nn.functional.gelu(branch(x)) for branch in self.branches],1)
        h = torch.nn.functional.gelu(self.middle(h)); h = torch.nn.functional.gelu(self.final(h))
        pooled = torch.cat((h.mean(-1),h.amax(-1)),1)
        learned = torch.nn.functional.gelu(self.project(pooled)).reshape(batch,candidates,16)
        return torch.cat((learned,torch.log1p(rms/10),mean/10),-1),rms[...,0]


@dataclass
class MorphEvidence(Evidence):
    local_representations: torch.Tensor
    candidate_scores: torch.Tensor
    rank_indices: torch.Tensor


class MorphModel(nn.Module):
    def __init__(self, cfg=MorphConfig(), *, mean_uV=0., std_uV=1.):
        super().__init__()
        if not math.isfinite(mean_uV) or not math.isfinite(std_uV) or std_uV<=0:
            raise ValueError('finite supplied physical calibration required')
        self.cfg = cfg
        self.encoder = LocalEncoder()
        self.background_head = nn.Linear(18,5)
        self.event_head = nn.Linear(18,5,bias=False)
        self.register_buffer('mean_uV',torch.tensor(float(mean_uV)))
        self.register_buffer('std_uV',torch.tensor(float(std_uV)))
        if cfg.method == 'M3':
            self.event_head.requires_grad_(False)

    def physical_uV(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,3000) or x.dtype!=torch.float32 or not torch.isfinite(x).all():
            raise ValueError('finite FP32 [B,1,3000] required')
        return x*self.std_uV+self.mean_uV

    def encode_candidates(self,x):
        patches = self.physical_uV(x)[:,0].unfold(-1,400,200)
        if self.cfg.method == 'M1_phase':
            mean = patches.mean(-1,keepdim=True)
            patches = torch.fft.irfft(torch.fft.rfft(patches-mean).abs().to(torch.complex64),n=400)+mean
        return self.encoder(patches)

    def decision(self,features,rms,*,witnesses=None):
        b,n,d = features.shape; k=self.cfg.event_budget
        if (n,d)!=(14,18) or rms.shape!=(b,n):
            raise ValueError('candidate feature geometry changed')
        mean = features.mean(1)
        method = self.cfg.method
        if method in {'M1','M1_phase'}:
            # Exactly one bias: M1 and M2 each have 185 head parameters.
            bg_parts = mean[:,:,None]*self.background_head.weight.T
            extra = features.amax(1)[:,:,None]*self.event_head.weight.T
            bg_parts = torch.cat((bg_parts,extra),1)
            bg_features = torch.cat((mean,features.amax(1)),1)
        else:
            bg_features = mean; bg_parts=mean[:,:,None]*self.background_head.weight.T
        background = bg_parts.sum(1)+self.background_head.bias
        raw = self.event_head(features)
        scores = rms if method=='M2_rms' else (raw-raw.mean(-1,keepdim=True)).square().mean(-1)
        ranking=scores.masked_fill(rms<=1e-6,-float('inf'))
        indices = torch.argsort(ranking,dim=1,descending=True,stable=True)[:,:k]
        valid = rms.gather(1,indices)>1e-6
        if witnesses is not None:
            iv,va=witnesses.intervals,witnesses.valid
            if iv.shape!=(b,k,2) or va.shape!=(b,k) or iv.dtype!=torch.long or va.dtype!=torch.bool or iv.device!=features.device or va.device!=features.device:
                raise ValueError('fixed witness shape/dtype/device invalid')
            lo,hi=iv.unbind(-1)
            if not (((lo>=0)&(hi<=3000)&(hi-lo==400)&(lo%200==0))[va]).all() or iv[~va].any():
                raise ValueError('fixed witness interval invalid')
            if any(len(set(v.tolist()))!=len(v) for v in lo[va].split(va.sum(1).tolist())):
                raise ValueError('duplicate fixed witness')
            indices=lo//200;valid=va
        selected=features.gather(1,indices[:,:,None].expand(-1,-1,d))
        signed = self.event_head(selected).tanh()*(8/k)
        if method in {'M1','M1_phase','M3'}:
            valid=torch.zeros_like(valid)
        contributions=signed*valid[:,:,None]
        return background+contributions.sum(1),background,bg_features,bg_parts,indices,valid,selected,contributions,scores

    def forward(self,x,*,return_evidence=False,witnesses=None):
        features,rms=self.encode_candidates(x)
        logits,bg,bf,bp,indices,valid,local,contributions,scores=self.decision(features,rms,witnesses=witnesses)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite morphology logits')
        if not return_evidence:return logits
        intervals=torch.stack((indices*200,indices*200+400),-1)
        intervals=torch.where(valid[:,:,None],intervals,torch.zeros_like(intervals))
        q=valid[:,:,None].expand_as(contributions).float()
        return MorphEvidence(logits,bg,bf,bp,self.background_head.bias,intervals,valid,
            scores.gather(1,indices)*valid,contributions,q,contributions,
            {'kind':'q_equals_one_no_observability_claim','fixed_witness_internal_intervention':witnesses is not None},
            self.cfg.method,tuple(wave_digest(v) for v in self.physical_uV(x)),local*valid[:,:,None],scores,indices)

    def conventional_ranking(self,features,rms):
        """Internal feature deletion, not waveform replacement or an event score."""
        before=self.decision(features,rms)[0]
        changes=[]
        for j in range(features.shape[1]):
            changed=features.clone();changed[:,j]=0
            changes.append((before-self.decision(changed,rms)[0]).abs().amax(-1))
        return torch.stack(changes,1)


def make_model(method, *, mean_uV=0., std_uV=1.):
    if method=='M0':
        from mist_cove.phase2.model import make_model as historical
        return historical('q1',mean_uV=mean_uV,std_uV=std_uV)
    return MorphModel(MorphConfig(method=method),mean_uV=mean_uV,std_uV=std_uV)

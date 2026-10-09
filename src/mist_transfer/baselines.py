"""Thin shared-interface adapters; existing baselines remain the model definitions."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from mist_evidence.model import ModelConfig, EvidenceModel
from mist_evidence.cli import variant_config


@dataclass(frozen=True)
class AdapterConfig:
    baseline:str
    radius:int=0
    def dictionary(self):return {'baseline':self.baseline,'radius':0,'use_crf':False,'input_samples':3000}


class AttnSleepAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        from protosleep.attnsleep import AttnSleepBaseline, init_attnsleep_weights
        self.network=AttnSleepBaseline();self.network.apply(init_attnsleep_weights)
        self.cfg=AdapterConfig('attnsleep')
    def encode(self,x):return self.network(x)
    def emissions(self,features):return features
    def loss(self,emissions,labels,valid):return F.cross_entropy(emissions[valid].float(),labels[valid])
    def predict(self,emissions):return emissions.argmax(-1),emissions.float().softmax(-1)


class GainEvidence(EvidenceModel):
    """Optional source-TRAIN gain only, +/-2%; no waveform warping or label change."""
    def __init__(self,cfg,anchors,amplitude_stats,normalization):
        super().__init__(cfg,anchors,amplitude_stats)
        self.register_buffer('raw_mean_over_std',torch.tensor(normalization['mean_uV']/normalization['std_uV']))
    def encode(self,x):
        if self.training:
            gain=.98+.04*torch.rand((len(x),1,1),device=x.device)
            x=gain*x+(gain-1)*self.raw_mean_over_std
        return super().encode(x)


def make_baseline(name,bank,normalization):
    if name=='attnsleep':return AttnSleepAdapter()
    cfg=ModelConfig(**{**bank['config'],'scales':tuple(bank['config']['scales'])})
    if cfg.radius!=0 or cfg.use_crf:raise ValueError('P3A matched epoch-only adapters require radius=0, no CRF')
    if name=='evidence_aug':return GainEvidence(cfg,bank['waveforms'],tuple(bank['amplitude_stats']),normalization)
    if name not in {'evidence','summary_only','raw_context'}:raise ValueError('unknown source baseline')
    return EvidenceModel(variant_config(name,cfg),bank['waveforms'],tuple(bank['amplitude_stats']))

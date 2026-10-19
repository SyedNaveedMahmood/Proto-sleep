"""Unchanged waveform encoder and explicitly restricted diagnostic probes."""
import torch
from torch import nn
from mist_cove.phase4.model import make_model
from .features import zero_phase


class WaveformControl(nn.Module):
    def __init__(self, control):
        super().__init__()
        if control not in {'original','patch_zero','coherent_zero'}:
            raise ValueError('unknown frozen transform')
        self.control=control
        self.core=make_model('M1_phase' if control=='patch_zero' else 'M1')

    def forward(self,x,**kwargs):
        if self.control=='coherent_zero': x=zero_phase(x)
        return self.core(x,**kwargs)


class RestrictedProbe(nn.Module):
    def __init__(self, train_features, hidden=32, std_floor=.01):
        super().__init__()
        if train_features.ndim!=2 or not torch.isfinite(train_features).all():
            raise ValueError('finite TRAIN features required')
        self.register_buffer('train_mean',train_features.mean(0))
        self.register_buffer('train_std',train_features.std(0,unbiased=False).clamp_min(std_floor))
        self.net=nn.Sequential(nn.Linear(train_features.shape[1],hidden),nn.GELU(),nn.Linear(hidden,5))

    def forward(self,x):
        return self.net((x-self.train_mean)/self.train_std)

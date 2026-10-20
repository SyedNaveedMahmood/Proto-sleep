"""Separate generated calibration role, fixed scalar grid and rejection policy."""
import torch
from .evaluation import reliability


def fit_temperature(logits,y,*,role):
    if role!='SYNTHETIC_CALIBRATION':raise ValueError('calibration fitting role rejected before tensor access')
    if logits.ndim!=2 or logits.shape[1]!=5 or not torch.isfinite(logits).all():raise ValueError('finite five logits required')
    candidates=torch.logspace(-.6,.8,61)
    losses=torch.stack([torch.nn.functional.cross_entropy(logits/t,y) for t in candidates])
    return float(candidates[losses.argmin()])


def rejection_threshold(logits,*,role):
    if role!='SYNTHETIC_CALIBRATION':raise ValueError('rejection fitting role rejected')
    return float(torch.quantile(logits.softmax(1).amax(1),.2))


def audit(logits,y,t,threshold):
    raw=reliability(logits,y);cal=reliability(logits,y,t)
    conf=(logits/t).softmax(1).amax(1);accept=conf>=threshold
    cal['fixed_rejection']={'threshold':threshold,'coverage':float(accept.float().mean()),
        'risk':float((logits.argmax(1)[accept]!=y[accept]).float().mean()) if accept.any() else None}
    return {'raw':raw,'temperature_scaled':cal,'temperature':t}

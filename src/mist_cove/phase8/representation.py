"""Frozen local representations, explicit overlapping diagnostic/nuisance sets."""
import torch
from .pooling import pool
from mist_cove.phase7.prospective import oracle_mask,head


def masks(draw):
    diagnostic=oracle_mask(draw)
    nuisance=torch.tensor([[any(j*200<b and j*200+400>a for a,b in iv) for j in range(14)] for iv in draw.nuisance_intervals])
    return {'diagnostic_only':diagnostic&~nuisance,'mixed':diagnostic&nuisance,
        'nuisance_only':nuisance&~diagnostic,'background':~(diagnostic|nuisance)}


def separation(features,y):
    centers=torch.stack([features[y==k].mean(0) for k in range(5)])
    within=(features-centers[y]).square().sum(-1).mean()
    between=torch.pdist(centers).square().mean()
    return {'within_class_squared':float(within),'between_class_squared':float(between),
        'ratio':float(between/within.clamp_min(1e-9))}


def compare(model,clean,changed,a,b):
    groups=masks(changed);diagnostic=oracle_mask(clean);result={}
    for name,mask in groups.items():
        result[name]={'candidate_count':int(mask.sum()),'clean_norm':float(a.norm(dim=-1)[mask].mean()) if mask.any() else None,
            'corrupt_norm':float(b.norm(dim=-1)[mask].mean()) if mask.any() else None,
            'drift':float((a-b).norm(dim=-1)[mask].mean()) if mask.any() else None,
            'fraction_absolute_feature_over_3':float((b[mask].abs()>3).float().mean()) if mask.any() else None}
    result['diagnostic_drift']=float((a-b).norm(dim=-1)[diagnostic].mean())
    result['pooled_drift']=float((pool(a)-pool(b)).norm(dim=-1).mean())
    result['nondiagnostic_max_winner_fraction']=float((~diagnostic).gather(1,b.argmax(1)).float().mean())
    result['pooled_separation']=separation(pool(b),clean.y)
    result['privileged_local_separation']=separation(pool(b,'oracle',oracle_mask(clean,contain=True))[:,:18],clean.y)
    return result

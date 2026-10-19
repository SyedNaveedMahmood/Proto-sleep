"""Donor-conditional waveform effects, with exact internal-path accounting."""
import numpy as np
import torch

from mist_cove.phase2.model import WitnessSet
from mist_cove.phase2.audit import replace_group, union_intervals
from mist_cove.model import spectral_features


@torch.no_grad()
def decompose(model,x,changed,intervals):
    if model.training:raise ValueError('audit requires eval mode')
    before=model(x,return_evidence=True);after=model(changed,return_evidence=True)
    fixed=model(changed,return_evidence=True,witnesses=WitnessSet.from_evidence(before))
    mask=torch.zeros_like(before.valid[0])
    for lo,hi in intervals:
        mask|=(before.event_intervals[0,:,0]<hi)&(before.event_intervals[0,:,1]>lo)&before.valid[0]
    full=before.logits[0]-after.logits[0]
    bg=before.background_logits[0]-after.background_logits[0]
    path=before.contributions[0].sum(0)-fixed.contributions[0].sum(0)
    reselection=fixed.contributions[0].sum(0)-after.contributions[0].sum(0)
    residual=full-bg-path-reselection
    if residual.abs().max()>1e-5:raise ValueError('counterfactual reconciliation fails')
    def vector(t):return t.detach().cpu().tolist()
    original_set={tuple(v) for v in before.event_intervals[0,before.valid[0]].tolist()}
    after_set={tuple(v) for v in after.event_intervals[0,after.valid[0]].tolist()}
    return {'additive':vector(before.contributions[0,mask].sum(0)),'full':vector(full),
            'background':vector(bg),'fixed_witness':vector(bg+path),'fixed_event_path':vector(path),
            'reselection':vector(reselection),'residual':vector(residual),
            'confidence_after':float(after.logits.softmax(-1).max()),
            'selection_membership_changed':original_set!=after_set,
            'internal_paths_are_physiological':False}


def changed_wave(model,x,intervals,parts,donor,seed):
    if donor=='oracle':
        physical=model.physical_uV(x).clone()
        for p in parts:
            if not p['diagnostic'] or not p['present']:continue
            lo,hi=p['interval']
            for a,b in union_intervals(intervals):
                start,end=max(a,lo),min(b,hi)
                if start<end:physical[0,0,start:end]-=p['wave'][start-lo:end-lo].to(x.device)
        changed=(physical-model.mean_uV)/model.std_uV
        return changed,[]
    # Independent generated background is not a real-data donor. The unchanged
    # replacement helper handles bridge/noise and explicitly labels physiology.
    return_values=replace_group(model,x,intervals,strategy='bridge' if donor=='bridge' else 'noise',seed=seed)
    return return_values[0],return_values[2]


@torch.no_grad()
def group_report(model,x,intervals,parts,*,seed=0):
    reports=[]
    for donor in ['oracle','bridge','background']:
        changed,diagnostics=changed_wave(model,x,intervals,parts,donor,seed)
        effect=decompose(model,x,changed,intervals)
        individuals=[]
        for iv in intervals:
            one,_=changed_wave(model,x,[iv],parts,donor,seed)
            individuals.append(decompose(model,x,one,[iv])['full'])
        # Oracle subtraction may leave nuisance/noise intact, by construction.
        physical=model.physical_uV(x)[0,0];modified=model.physical_uV(changed)[0,0]
        pre=spectral_features(physical[None])[0];post=spectral_features(modified[None])[0]
        effect.update(donor=donor,individuals=individuals,elementary_supported=all(d.get('elementary_guard_passed',False) for d in diagnostics),
                      energy_mismatch=float((modified.square().mean()-physical.square().mean()).abs()/physical.square().mean().clamp_min(1e-8)),
                      spectral_distortion=float((pre-post).abs().sum()),
                      boundary_jump_change_max=max((abs(d['boundary_jump_after_uV']-d['boundary_jump_before_uV']) for d in diagnostics),default=0.),
                      physiological_validity='NOT_ESTABLISHED')
        reports.append(effect)
    spread=np.ptp([r['full'] for r in reports],axis=0)
    return {'donors':reports,'donor_spread_max':float(spread.max()),'group_declared_before_effects':True,
            'joint_bridge_geometry_can_differ_from_individual':True}

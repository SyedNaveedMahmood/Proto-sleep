"""Keep generator interventions, fixed score paths and donor effects distinct."""
import numpy as np
import torch
from mist_cove.phase4.audit import decompose
from mist_cove.phase2.audit import replace_group,union_intervals
from mist_cove.phase5.runtime import predict,paired_summary
from .generator import generate,class_swap,compose
from .evaluation import selected


@torch.no_grad()
def pairs(model,data,c,seed,budget):
    swapped=class_swap(data)
    other=generate(len(data.y),seed+c['pair_offset'],data.task,'nuisance_swap',latents=data.latents)
    a=predict(model,data.x,budget);b=predict(model,swapped.x,budget);u=predict(model,other.x,budget)
    # Requiring the changed prediction to match the NEW class excludes arbitrary
    # instability masquerading as diagnostic sensitivity.
    appropriate=((a.argmax(1)!=b.argmax(1))&(b.argmax(1)==swapped.y)).numpy()
    return {'appropriate_diagnostic_change':paired_summary(appropriate,seed,c['bootstrap_replicates']),
        'nuisance_agreement':paired_summary((a.argmax(1)==u.argmax(1)).numpy(),seed,c['bootstrap_replicates']),
        'diagnostic_baseline_exact':torch.equal(data.base,swapped.base),
        'nuisance_diagnostic_exact':torch.equal(data.events,other.events)}


def donor_diagnostics(x,modified,intervals,guard):
    union=union_intervals(intervals)
    old=torch.cat([x[0,0,lo:hi] for lo,hi in union]);new=torch.cat([modified[0,0,lo:hi] for lo,hi in union])
    energy=float((new.square().mean()-old.square().mean()).abs()/old.square().mean().clamp_min(1e-8))
    a=torch.fft.rfft(old).abs();b=torch.fft.rfft(new).abs()
    spectral=float((a-b).abs().sum()/a.abs().sum().clamp_min(1e-8));boundary=0.
    for lo,hi in union_intervals(intervals):
        for j in [lo,hi]:
            if 0<j<3000:boundary=max(boundary,float(((modified[0,0,j]-modified[0,0,j-1])-(x[0,0,j]-x[0,0,j-1])).abs()))
    return {'relative_energy_change':energy,'relative_magnitude_l1':spectral,'boundary_jump_change':boundary,
        'accepted':bool(torch.isfinite(modified).all() and energy<=guard['relative_energy_max'] and spectral<=guard['relative_magnitude_l1_max'] and boundary<=guard['boundary_change_uV_max']),
        'measurement_support':'concatenated targeted union segments; not physiological continuity','physiological_validity':'NOT_ESTABLISHED'}


def control_intervals(x,truth):
    # Fixed rule, evaluated before effects: nearest original RMS, stable onset tie.
    chosen=[]
    for lo,hi in truth:
        rms=float(x[0,0,lo:hi].square().mean().sqrt());length=hi-lo
        candidates=[(abs(float(x[0,0,a:a+length].square().mean().sqrt())-rms),a) for a in range(0,3001-length,100)
            if not any(a<b and a+length>l for l,b in truth+chosen)]
        if not candidates:return []
        _,a=min(candidates);chosen.append((a,a+length))
    return chosen


@torch.no_grad()
def audit_example(model,data,other,i,c):
    model.eval();x=data.x[i:i+1];truth=data.intervals[i]
    if not truth:return {'status':'NO_DIAGNOSTIC_EVENTS','oracle_effect':None,'donor_robustness':'NOT_TESTED'}
    before=model(x,return_evidence=True);keep=torch.ones(1,2)
    group=decompose(model,x,data.base[i:i+1],truth)
    active=[j for j in range(2) if data.events[i,j].abs().sum()>0]
    individuals=[]
    for j in active:
        keep=torch.ones(1,2);keep[0,j]=0
        modified=compose(data.base[i:i+1],data.events[i:i+1],keep)
        individuals.append(decompose(model,x,modified,truth)['full'])
    control=control_intervals(x,truth)
    # Nondiagnostic control zeroing is distinct from exact diagnostic-component
    # subtraction. It removes background too; retain this estimand limitation.
    controlled=x.clone()
    for lo,hi in control:controlled[0,0,lo:hi]=0
    effect_control=(model(x)-model(controlled))[0]
    donors=[]
    for name in ['bridge','independent_baseline']:
        if name=='bridge':modified,_,_=replace_group(model,x,truth,strategy='bridge')
        else:
            modified=x.clone()
            for lo,hi in union_intervals(truth):modified[0,0,lo:hi]=other.base[i,0,lo:hi]
        diagnostics=donor_diagnostics(x,modified,truth,c['donor_guard'])
        donors.append({'strategy':name,'diagnostics':diagnostics,**decompose(model,x,modified,truth)})
    accepted=[r for r in donors if r['diagnostics']['accepted']]
    spread=float(np.ptp([r['full'] for r in accepted],axis=0).max()) if len(accepted)==2 else None
    current=selected(model,x)[0]
    g=torch.Generator().manual_seed(817+i)
    after=selected(model,x+torch.randn(x.shape,generator=g)*.01)[0]
    first=set(current);second=set(after)
    stage=int(data.y[i]);native=before.contributions[0].sum(0)
    return {'status':'AUDITED','oracle_effect':group,'individual_full_effects':individuals,
        'native_score_minus_group_full':(native-torch.tensor(group['full'])).tolist(),
        'internal_all_event_score_deletion':native.tolist(),'internal_deletion_is_physiological':False,
        'joint_abs_true_stage':abs(group['full'][stage]),'largest_individual_abs_true_stage':max(abs(v[stage]) for v in individuals),
        'oracle_minus_control_abs_true_stage':abs(group['full'][stage])-float(effect_control[stage].abs()),
        'control_rule':'same-length original-RMS-nearest outside all diagnostic support; earliest ties',
        'control_available':bool(control),'donors':donors,'accepted_donor_count':len(accepted),
        'control_vs_oracle_caveat':'Control waveform zeroing also removes background; oracle recomposition removes only the diagnostic component. Not identical interventions.',
        'accepted_donor_spread':spread,'donor_robustness':'NOT_TESTED' if spread is None else ('PASS' if spread<=c['donor_guard']['spread_logit_max'] else 'FAIL'),
        'tiny_noise_set_jaccard_distance':1-len(first&second)/max(1,len(first|second)),
        'intervals_predeclared':True,'bridge_union_caveat':True}


@torch.no_grad()
def audit(model,data,c,seed):
    other=generate(len(data.y),seed+1,data.task,'clean')
    chosen=[int((data.y==s).nonzero()[0]) for s in range(5)]
    rows=[audit_example(model,data,other,i,c) for i in chosen]
    valid=[r for r in rows if r['status']=='AUDITED']
    return {'n_examples':len(rows),'examples':rows,
        'oracle_minus_control_abs_true_stage_mean':float(np.mean([r['oracle_minus_control_abs_true_stage'] for r in valid])) if valid else None,
        'donor_robustness':'NOT_TESTED' if not any(r['accepted_donor_spread'] is not None for r in valid) else 'SEE_PER_EXAMPLE_ACCEPTED_ONLY',
        'max_reconciliation_residual':max((abs(v) for r in valid for v in r['oracle_effect']['residual']),default=0.),
        'synthetic_only':True}

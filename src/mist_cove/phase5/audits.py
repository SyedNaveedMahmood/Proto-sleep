"""Generated oracle pairs, donor dependence and internal evidence estimands."""
import numpy as np
import torch
from mist_cove.phase4.experiment import selected_intervals
from mist_cove.phase4.audit import decompose
from mist_cove.phase2.audit import replace_group,union_intervals
from .features import patches
from .signals import diagnostic_swap,reverse_pair
from .runtime import predict,paired_summary


@torch.no_grad()
def pairing(model,draw,other,seed,c,budget=None):
    swapped=diagnostic_swap(draw,(draw.y.numpy()+1)%5)
    if not torch.equal(draw.nuisance,swapped.nuisance): raise ValueError('nuisance changed in diagnostic intervention')
    before=predict(model,draw.x,budget);diag=predict(model,swapped.x,budget)
    changed=predict(model,reverse_pair(draw,other),budget);removed=predict(model,draw.nuisance,budget)
    return {'diagnostic_change':paired_summary((before.argmax(1)!=diag.argmax(1)).numpy(),seed,c['bootstrap_replicates']),
        'nuisance_agreement':paired_summary((before.argmax(1)==changed.argmax(1)).numpy(),seed,c['bootstrap_replicates']),
        'diagnostic_swap_probability_l1':paired_summary((before.softmax(1)-diag.softmax(1)).abs().sum(1).numpy(),seed,c['bootstrap_replicates']),
        'oracle_erasure_logit_l1':paired_summary((before-removed).abs().sum(1).numpy(),seed,c['bootstrap_replicates']),
        'oracle_erasure_confidence':float(removed.softmax(1).amax(1).mean()),
        'paired_nuisance_exact':True,'counterfactual_is_physiological':False}


@torch.no_grad()
def witness_audit(model,draw,other,seed):
    """Five deterministic first-per-class examples; aggregate vectors only.

    Ground truth for the new task spans the epoch, so focal localization cannot
    distinguish useful windows from equally informative unused windows.
    """
    rows=[];stability=[]
    for stage in range(5):
        i=int((draw.y==stage).nonzero()[0]);x=draw.x[i:i+1]
        intervals=selected_intervals(model,x)
        g=torch.Generator().manual_seed(seed+i+619)
        changed=x+torch.randn(x.shape,generator=g)*.01
        original_set=set(intervals);after_set=set(selected_intervals(model,changed))
        stability.append(1-len(original_set&after_set)/max(1,len(original_set|after_set)))
        donors=[]
        for donor in ['oracle','bridge','independent_nuisance']:
            modified=x.clone()
            if donor=='bridge':
                modified,_,_=replace_group(model,x,intervals,strategy='bridge',seed=seed+i)
            else:
                for lo,hi in union_intervals(intervals):
                    if donor=='oracle': modified[0,0,lo:hi]-=draw.diagnostic[i,0,lo:hi]
                    else: modified[0,0,lo:hi]=other.nuisance[i,0,lo:hi]
            report=decompose(model,x,modified,intervals)
            individual=[]
            for lo,hi in intervals:
                one=x.clone()
                if donor=='oracle': one[0,0,lo:hi]-=draw.diagnostic[i,0,lo:hi]
                elif donor=='independent_nuisance': one[0,0,lo:hi]=other.nuisance[i,0,lo:hi]
                else: one,_,_=replace_group(model,x,[(lo,hi)],strategy='bridge',seed=seed+i)
                individual.append(decompose(model,x,one,[(lo,hi)])['full'])
            before_mag=torch.fft.rfft(x).abs();after_mag=torch.fft.rfft(modified).abs()
            report.update({'donor':donor,'individuals':individual,
                'energy_relative_change':float((modified.square().mean()-x.square().mean()).abs()/x.square().mean().clamp_min(1e-8)),
                'magnitude_relative_l1_change':float((after_mag-before_mag).abs().sum()/before_mag.abs().sum().clamp_min(1e-8)),
                'max_first_difference_change':float((modified.diff(dim=-1)-x.diff(dim=-1)).abs().max()),
                'physiological_validity':'NOT_ESTABLISHED'})
            donors.append(report)
        rows.append(donors)
    vectors=['additive','full','background','fixed_witness','fixed_event_path','reselection']
    return {'n_examples':5,'localization':'NOT_DISTINGUISHING_GLOBAL_CODE_IN_EVERY_WINDOW' if draw.family.startswith('harmonic') else 'FULL_EPOCH_ERASURE_IS_NOT_FOCAL_GROUND_TRUTH',
        'tiny_noise_jaccard_mean':float(np.mean(stability)),
        'mean_donor_spread':float(np.mean([np.ptp([d['full'] for d in row],axis=0).max() for row in rows])),
        'max_reconciliation_residual':max(abs(v) for row in rows for d in row for v in d['residual']),
        'donors':{name:{**{key:np.mean([row[j][key] for row in rows],axis=0).tolist() for key in vectors},
            'individual_full_mean':np.mean([np.mean(row[j]['individuals'],axis=0) for row in rows],axis=0).tolist(),
            'energy_relative_change_mean':float(np.mean([row[j]['energy_relative_change'] for row in rows])),
            'magnitude_relative_l1_change_mean':float(np.mean([row[j]['magnitude_relative_l1_change'] for row in rows])),
            'max_first_difference_change':max(row[j]['max_first_difference_change'] for row in rows)} for j,name in enumerate(['oracle','bridge','independent_nuisance'])},
        'internal_fixed_path_is_physiological':False,'individual_additive_is_necessity':False}

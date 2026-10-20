"""Fixed-five metrics and draw-cluster uncertainty; no fit on locked outcomes."""
import numpy as np
import torch
from mist_cove.phase7.prospective import score
from mist_cove.phase5.runtime import paired_summary


@torch.no_grad()
def reliability(logits,y,temperature=1.):
    logits=logits/temperature;result=score(logits,y)
    p=logits.softmax(-1);confidence,pred=p.max(-1);correct=pred==y
    bins=[]
    for j in range(15):
        mask=(confidence>=j/15)&((confidence<(j+1)/15) if j<14 else (confidence<=1))
        bins.append({'count':int(mask.sum()),'confidence':float(confidence[mask].mean()) if mask.any() else None,
                     'accuracy':float(correct[mask].float().mean()) if mask.any() else None})
    result['reliability_bins']=bins
    result['per_class_confidence']=[float(confidence[y==k].mean()) if (y==k).any() else None for k in range(5)]
    result['error_confidence_correlation']=float(np.corrcoef(confidence.numpy(),(~correct).float().numpy())[0,1]) if confidence.std()>1e-8 and correct.any() and (~correct).any() else None
    result['risk_coverage']=[]
    order=confidence.argsort(descending=True,stable=True)
    for fraction in (.1,.25,.5,.75,1.):
        keep=order[:max(1,round(len(y)*fraction))]
        result['risk_coverage'].append({'coverage':len(keep)/len(y),'risk':float((~correct[keep]).float().mean())})
    return result


def paired_accuracy_interval(a,b,y,seed):
    # Resample original generated draws, preserving both interventions.
    delta=(b.argmax(1)==y).float()-(a.argmax(1)==y).float()
    return paired_summary(delta.tolist(),seed)


def gate(rows,c):
    primary=[r for r in rows if r['method']==c['primary']]
    expected={(t,s) for t in c['tasks'] for s in c['seeds']};checks=[]
    for r in primary:
        v=r['conditions'];p0=next(x for x in rows if x['method']=='P0' and (x['task'],x['seed'])==(r['task'],r['seed']))
        clauses={'clean_floor':v['clean']['macro_f1']>=.8,
            'clean_penalty':v['clean']['macro_f1']>=p0['conditions']['clean']['macro_f1']-.03,
            'competitive':v['competitive']['macro_f1']>=.7,'added':v['added']['macro_f1']>=.7,
            'diagnostic_pairs':all(vv['appropriate_diagnostic_change']>=.8 for vv in r['pairs'].values()),
            'nuisance_pairs':all(vv['nuisance_agreement']>=.8 for vv in r['pairs'].values()),
            'no_collapse':all(not v[k]['collapsed'] for k in ('clean','competitive','added'))}
        checks.append({'task':r['task'],'seed':r['seed'],'clauses':clauses,'pass':all(clauses.values())})
    complete=len(primary)==len(expected) and {(r['task'],r['seed']) for r in primary}==expected
    return {'primary':c['primary'],'status':'PASS' if complete and all(v['pass'] for v in checks) else 'FAIL',
        'complete':complete,'checks':checks}

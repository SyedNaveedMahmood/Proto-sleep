"""Five-output metrics and comparable four-window localization policies."""
import numpy as np
import torch
from mist_cove.experiment import metrics
from mist_cove.phase5.runtime import predict,wilson
from .features import signal_view


def score(logits,y,c):
    r=metrics(logits,y);p=logits.softmax(-1);pred=p.argmax(-1)
    r['accuracy_ci99']=wilson(int((pred==y).sum()),len(y),c['criteria']['wilson_z'])
    r['equivalent_chance']=bool(c['criteria']['restricted_equivalence_low']<=r['accuracy_ci99'][0] and r['accuracy_ci99'][1]<=c['criteria']['restricted_equivalence_high'])
    r['reliable_leakage']=r['accuracy_ci99'][0]>c['criteria']['restricted_equivalence_high']
    r['false_confidence_0_9']=float(((pred!=y)&(p.amax(1)>=.9)).float().mean())
    return r


def iou(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))/(max(a[1],b[1])-min(a[0],b[0]))


def location(selected,truth,nuisance,threshold=.25):
    if not truth:return {'top1_hit':None,'topk_hit':None,'precision':None,'coverage':None,'mean_best_iou':None,'false_nuisance_fraction':None,'empty_truth_selected_count':len(selected)}
    hits=[any(iou(s,t)>=threshold for t in truth) for s in selected]
    covered=[any(iou(s,t)>=threshold for s in selected) for t in truth]
    false=[not hit and any(min(s[1],u[1])>max(s[0],u[0]) for u in nuisance) for s,hit in zip(selected,hits)]
    return {'top1_hit':float(hits[0]) if hits else 0.,'topk_hit':float(any(hits)),
        'precision':float(np.mean(hits)) if hits else 0.,'coverage':float(np.mean(covered)),
        'mean_best_iou':float(np.mean([max(iou(s,t) for t in truth) for s in selected])) if selected else 0.,
        'false_nuisance_fraction':float(np.mean(false)) if false else 0.,'empty_truth_selected_count':0}


@torch.no_grad()
def selected(model,x):
    f,r=model.encode_candidates(x);out=model.decision(f,r)
    scores=model.conventional_ranking(f,r) if model.cfg.method in {'M1','M3'} else out[-1]
    ix=scores.masked_fill(r<=1e-6,-float('inf')).argsort(dim=1,descending=True,stable=True)[:,:4]
    return [[(int(j)*200,int(j)*200+400) for j in row if r[b,j]>1e-6] for b,row in enumerate(ix)]


@torch.no_grad()
def evaluate(model,data,c,budget,*,view='full',localize=False,seed=0):
    model.eval();x=signal_view(data,view)
    if not localize:return score(predict(model,x,budget),data.y,c)
    logits=[];records={k:[] for k in ['model','random','rms']};residual=0.
    g=torch.Generator().manual_seed(seed+5151)
    for start in range(0,len(x),64):
        budget.check();xx=x[start:start+64];f,r=model.encode_candidates(xx);out=model.decision(f,r)
        logits.append(out[0]);residual=max(residual,float((out[0]-out[1]-out[7].sum(1)).abs().max()))
        policy=model.conventional_ranking(f,r) if model.cfg.method in {'M1','M3'} else out[-1]
        for name,rank in [('model',policy),('random',torch.rand(r.shape,generator=g)),('rms',r)]:
            ix=rank.masked_fill(r<=1e-6,-float('inf')).argsort(dim=1,descending=True,stable=True)[:,:4]
            for b,row in enumerate(ix):
                intervals=[(int(j)*200,int(j)*200+400) for j in row if r[b,j]>1e-6]
                i=start+b;records[name].append(location(intervals,data.intervals[i],data.nuisance_intervals[i],c['criteria']['iou_hit']))
    if residual>c['criteria']['reconciliation']:raise ValueError('additive reconciliation failed')
    result=score(torch.cat(logits),data.y,c)
    result['localization']={name:{key:float(np.mean([v[key] for v in values if v[key] is not None])) if any(v[key] is not None for v in values) else None for key in values[0]} for name,values in records.items()}
    result['localization_policy']='internal_feature_deletion' if model.cfg.method in {'M1','M3'} else 'native_additive_selection'
    result['max_additive_residual']=residual
    return result

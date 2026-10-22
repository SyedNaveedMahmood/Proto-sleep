"""Interval and nominal agreement; missing observations remain missing."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score


def interval_mask(intervals, tolerance_samples=0, n=3000):
    mask=np.zeros(n,bool)
    for a,b in intervals:
        if not 0<=a<b<=n:raise ValueError('interval out of bounds')
        mask[max(0,a-tolerance_samples):min(n,b+tolerance_samples)]=True
    return mask


def overlap_metrics(support, events, tolerance_seconds=0):
    support=interval_mask(support);events=interval_mask(events,round(tolerance_seconds*100))
    overlap=int((support&events).sum());s=int(support.sum());e=int(events.sum())
    return {'overlap_seconds':overlap/100,'support_seconds':s/100,'event_seconds':e/100,
            'precision':overlap/s if s else None,'event_coverage':overlap/e if e else None,
            'support_outside_event_seconds':int((support&~events).sum())/100}


def event_agreement(first,second,minimum_iou=.2):
    if not first or not second:
        return {'first':len(first),'second':len(second),'matched':0,'F1':0. if first or second else None,
                'onset_abs_seconds':[],'offset_abs_seconds':[],'IoU':[]}
    matrix=np.zeros((len(first),len(second)))
    for i,(a,b) in enumerate(first):
        for j,(c,d) in enumerate(second):matrix[i,j]=max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c))
    # Maximize eligible matches first, then overlap; do not pair by the model.
    eligible=matrix>=minimum_iou
    rows,cols=linear_sum_assignment(-(eligible.astype(float)*100+matrix))
    pairs=[(i,j) for i,j in zip(rows,cols) if eligible[i,j]]
    return {'first':len(first),'second':len(second),'matched':len(pairs),'F1':2*len(pairs)/(len(first)+len(second)),
            'onset_abs_seconds':[abs(first[i][0]-second[j][0])/100 for i,j in pairs],
            'offset_abs_seconds':[abs(first[i][1]-second[j][1])/100 for i,j in pairs],
            'IoU':[float(matrix[i,j]) for i,j in pairs]}


def categorical_agreement(first,second):
    pairs=[(a,b) for a,b in zip(first,second) if a is not None and b is not None]
    if not pairs:return {'pairs':0,'agreement':None,'kappa':None}
    a,b=map(list,zip(*pairs));kappa=cohen_kappa_score(a,b) if len(set(a+b))>1 else None
    return {'pairs':len(a),'agreement':float(np.mean(np.array(a)==np.array(b))),
            'kappa':float(kappa) if kappa is not None and np.isfinite(kappa) else None}


def require_locked_expert_labels(exports):
    if len(exports)<2 or any(not e.get('locked') or not e.get('qualified_approved') or e.get('domain')!='REAL_TRAIN' for e in exports):
        raise ValueError('independently approved qualified locked human exports required')
    if len({e.get('reviewer') for e in exports})!=len(exports) or any(not e.get('reviewer') for e in exports):
        raise ValueError('distinct independent reviewer identities required')
    if len({e.get('packet_sha256') for e in exports})!=1 or not exports[0].get('packet_sha256'):
        raise ValueError('same frozen blinded packet required')


def controls(mask,seed=123):
    mask=int(mask);k=mask.bit_count();rng=np.random.default_rng(seed)
    random=sum(1<<int(i) for i in rng.choice(15,k,replace=False))
    positional=0
    for start in (0,5,10):
        count=sum(bool(mask&(1<<j)) for j in range(start,start+5))
        positional|=sum(1<<int(j) for j in rng.choice(np.arange(start,start+5),count,replace=False))
    return {'length_matched_mask':random,'position_matched_mask':positional,
            'degenerate_whole_epoch':k==15,'position_identical':positional==mask}


def mask_intervals(mask):
    return [[j*200,(j+1)*200] for j in range(15) if mask&(1<<j)]


def review_quality(first,second,expected_cases,duplicate_pairs=()):
    from .annotation_schema import CATEGORIES
    a=first.get('assessments',{});b=second.get('assessments',{})
    result={'first_completion':len(set(a)&set(expected_cases))/len(expected_cases),
            'second_completion':len(set(b)&set(expected_cases))/len(expected_cases),'categories':{},'duplicates':[]}
    for category in CATEGORIES:
        states_a=[];states_b=[];events=[]
        for case in expected_cases:
            aa=[v for v in a.get(case,{}).get('assessments',[]) if v['category']==category]
            bb=[v for v in b.get(case,{}).get('assessments',[]) if v['category']==category]
            states_a.append(aa[0]['state'] if aa else None);states_b.append(bb[0]['state'] if bb else None)
            if aa and bb and aa[0]['state']=='present' and bb[0]['state']=='present':
                events.append(event_agreement(aa[0]['intervals'],bb[0]['intervals']))
        result['categories'][category]={'categorical':categorical_agreement(states_a,states_b),'matched_positive_cases':events,
            'first_missing':sum(v is None for v in states_a),'second_missing':sum(v is None for v in states_b),
            'first_uncertain_or_unassessable':sum(v in ('uncertain','not_assessable','insufficient_signal_quality') for v in states_a)}
    for original,duplicate in duplicate_pairs:
        for label,export in [('first',a),('second',b)]:
            if original in export and duplicate in export:
                result['duplicates'].append({'reviewer':label,'identical_assessments':export[original]['assessments']==export[duplicate]['assessments']})
    return result

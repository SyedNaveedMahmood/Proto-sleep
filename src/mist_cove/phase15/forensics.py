"""Exploratory ST diagnostics from authenticated Phase XIV private caches.

No model loading, training, EEG opening, or public per-record output occurs here.
"""
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .governance import OLD_PRIVATE, OLD_PUBLIC, PRIVATE, PUBLIC, LABEL, read, new, verify_plan_commit, sha, canonical


def _key(seed, row):
    token = f"{seed}|ST_EVAL|{row['recording']}|{row['epoch']}"
    return hashlib.sha256(token.encode()).hexdigest()


def _prob(logits):
    a = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(axis=1, keepdims=True)


def _safe(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {'n': len(a), 'mean': _safe(a.mean()) if len(a) else None,
            'median': _safe(np.median(a)) if len(a) else None,
            'q25': _safe(np.quantile(a, .25)) if len(a) else None,
            'q75': _safe(np.quantile(a, .75)) if len(a) else None}


def _stratum(r):
    if r['truth'] == 4:
        return 'correct_REM' if r['pred'] == 4 else 'REM_to_'+('W','N1','N2','N3')[r['pred']]
    if r['truth'] == 2 and r['pred'] == 2:
        return 'correct_N2'
    return 'non_REM_error' if r['pred'] != r['truth'] else 'other_correct'


def _round_robin(rows, limit, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['person'], r['night'])].append(r)
    for group in groups.values():
        group.sort(key=lambda r: _key(seed, r))
    keys = sorted(groups, key=lambda k: hashlib.sha256(f'{seed}|{k}'.encode()).hexdigest())
    selected = []
    depth = 0
    while len(selected) < limit:
        found = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth]); found = True
                if len(selected) == limit: break
        if not found: break
        depth += 1
    return selected


def _distances(y):
    out = np.full(len(y), len(y), dtype=int)
    # Search only within contiguous scored physical runs. A missing hypnogram
    # position is an impenetrable boundary, not a removable gap.
    valid=y>=0
    starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]])
    ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])+1
    for lo,hi in zip(starts,ends):
        changes=np.flatnonzero(y[lo+1:hi]!=y[lo:hi-1])+lo+1
        if not len(changes):continue
        indices=np.arange(lo,hi)
        j=np.searchsorted(changes,indices)
        out[lo:hi]=np.minimum(np.abs(indices-changes[np.minimum(j,len(changes)-1)]),
                             np.abs(indices-changes[np.maximum(j-1,0)]))
    return out


def _event_present(events, index, key):
    lo, hi = index*30, (index+1)*30
    return any(e['start'] < hi and e['end'] > lo for e in events.get(key, []))


def collect():
    """Return private epoch rows after exact published-cache identity checks."""
    verify_plan_commit()
    expected = read(OLD_PUBLIC/'publication_manifest.json')['file_sha256']
    # Phase XIV receipts identify the public data and the private source geometry.
    # Recheck geometry before using its role and label ordering.
    protocol = read(OLD_PUBLIC/'frozen_st_protocol.json')
    geometry = read(OLD_PRIVATE/'geometry.json')
    if sha(canonical(geometry)) != protocol['geometry_sha256']:
        raise ValueError('private Phase XIV geometry identity changed')
    geoms = {x['record']['recording']: x for x in geometry['rows'] if x['record']['role']=='ST_EVAL'}
    if len(geoms) != 44: raise ValueError('ST role count differs')
    rows=[]; confusion=np.zeros((5,5),dtype=np.int64); per_record={}
    for rec,g in sorted(geoms.items()):
        path=OLD_PRIVATE/'ST_EVAL'/(rec+'_epochs.npz')
        if path.is_symlink() or not path.is_file():raise ValueError('unsafe ST cache')
        with np.load(path,allow_pickle=False) as z:
            logits=z['logits'].copy(); truth=z['truth'].copy(); features=z['features'].copy()
        if len(truth)!=len(g['labels']) or not np.array_equal(truth,np.asarray(g['labels'])):
            raise ValueError('ST physical epoch or label alignment differs')
        if len(logits)!=len(truth) or features.shape!=(len(truth),15):
            raise ValueError('ST cache geometry differs')
        p=_prob(logits);pred=logits.argmax(1);d=_distances(truth)
        night=g['record']['night'];person=g['record']['subject']
        for i in np.flatnonzero(truth>=0):
            y=int(truth[i]);q=int(pred[i]);confusion[y,q]+=1
            order=np.partition(logits[i],-2)
            rows.append({'recording':rec,'person':person,'night':night,'epoch':int(i),'truth':y,'pred':q,
                         'confidence':float(p[i,q]),'truth_probability':float(p[i,y]),
                         'entropy':float(-(p[i]*np.log(p[i].clip(1e-30))).sum()),
                         'margin':float(order[-1]-order[-2]),'distance':int(d[i]),
                         'features':[float(x) for x in features[i]]})
        per_record[rec]={'person':person,'night':night,'scored':int((truth>=0).sum()),'physical':len(truth)}
    published=read(OLD_PUBLIC/'st_classification_metrics.json')['metrics']
    if confusion.tolist()!=published['confusion_matrix'] or confusion[4].tolist()!=[2322,1570,2648,4,1805]:
        raise ValueError('published ST confusion cannot be reproduced')
    if int(confusion.sum())!=42691 or len(rows)!=42691:raise ValueError('ST scored denominator differs')
    return rows,confusion,per_record


def select_panels(rows):
    config=read(Path(__file__).resolve().parents[3]/'configs/mist_cove_phase15_v1.json')
    seed=config['panels']['seed']
    grouped=defaultdict(list)
    for r in rows: grouped[(r['person'],r['night'],r['truth'])].append(r)
    e1=[min(group,key=lambda r:_key(seed,r)) for group in grouped.values()]
    e1=sorted(e1,key=lambda r:_key(seed,r))[:220]
    # E3 is intentionally model-outcome conditioned and cannot estimate prevalence.
    strata=('REM_to_W','REM_to_N1','REM_to_N2','correct_REM','correct_N2','non_REM_error')
    chosen=set();e3=[];shortfalls={}
    for s in strata:
        pool=[r for r in rows if _stratum(r)==s and (r['recording'],r['epoch']) not in chosen]
        selected=_round_robin(pool,40,seed)
        e3.extend(selected);chosen.update((r['recording'],r['epoch']) for r in selected)
        shortfalls[s]=40-len(selected)
    # Detector-eligible E2 is computed using the exact existing stage-blind
    # candidate events, with no explanation result as an input.
    events={};e2=[];statuses=Counter()
    for r in e1:
        if r['recording'] not in events:
            event_path=OLD_PRIVATE/'ST_EVAL'/(r['recording']+'_events.json')
            if event_path.is_symlink() or not event_path.is_file():raise ValueError('unsafe detector cache')
            events[r['recording']]=read(event_path)
            if events[r['recording']]['n_epochs']<r['epoch']+1 or not events[r['recording']]['stage_blind_detection']:
                raise ValueError('detector geometry or stage blindness differs')
        ev=events[r['recording']]
        key='EEG Pz-Oz|spindle'
        present=_event_present(ev['events'],r['epoch'],key)
        statuses['candidate_present' if present else 'no_candidate']+=1
        if present:e2.append(r)
    def coord(r):return {k:r[k] for k in ('recording','person','night','epoch','truth','pred')}
    panel={'status':LABEL,'selection_seed':seed,'E1':[coord(r) for r in e1],
           'E2':[coord(r) for r in e2], 'E3':[coord(r) for r in e3],
           'E3_shortfalls':shortfalls,'E2_status_counts':dict(statuses),
           'E1_E3_overlap':len({(r['recording'],r['epoch']) for r in e1}&chosen)}
    return panel


def analyze(rows,confusion,per_record,panel):
    groups=defaultdict(list);persons=defaultdict(list);conditions=defaultdict(list)
    cond=read(OLD_PRIVATE/'conditions.json')['subjects']
    for r in rows:
        s=_stratum(r);groups[s].append(r);persons[r['person']].append(r)
        c=cond[r['person']][str(r['night'])]
        conditions[c].append(r)
    fields={'confidence':None,'truth_probability':None,'entropy':None,'margin':None,'distance':None,
            'EEG_RMS_uV':1,'EEG_delta_fraction':3,'EEG_theta_fraction':4,'EEG_alpha_fraction':5,
            'EEG_sigma_fraction':6,'EEG_HF_fraction':7,'EOG_RMS_uV':8,
            'EMG_native_RMS_uV':9,'EMG_20_45Hz_RMS_uV':10,'flat_fraction':11,'step_fraction':12,
            'EEG_crosschannel_correlation':13}
    out={}
    for s,rr in groups.items():
        out[s]={'n':len(rr),'people':len({r['person'] for r in rr}),
                'night_records':len({r['recording'] for r in rr}),
                'within_1_epoch_of_transition':sum(r['distance']<=1 for r in rr),
                'within_2_epochs_of_transition':sum(r['distance']<=2 for r in rr),
                'within_4_epochs_of_transition':sum(r['distance']<=4 for r in rr),
                'features':{name:_summary([r[name] if idx is None else r['features'][idx] for r in rr]) for name,idx in fields.items()},
                'small_margin_count':sum(r['margin']<.5 for r in rr),
                'high_entropy_count':sum(r['entropy']>1.1 for r in rr),
                'high_confidence_count':sum(r['confidence']>=.9 for r in rr)}
    person_summary=[]
    for person,rr in sorted(persons.items()):
        rem=[r for r in rr if r['truth']==4]
        night={str(n):{'REM_scored':sum(r['truth']==4 and r['night']==n for r in rr),
                       'REM_correct':sum(r['truth']==4 and r['pred']==4 and r['night']==n for r in rr)} for n in (1,2)}
        person_summary.append({'person_digest':sha(person.encode())[:12],
                               'REM_scored':len(rem),'REM_correct':sum(r['pred']==4 for r in rem),
                               'paired_nights':night})
    night_rates=[]
    for person in sorted(persons):
        for night in (1,2):
            rr=[r for r in persons[person] if r['truth']==4 and r['night']==night]
            if rr:night_rates.append(sum(r['pred']==4 for r in rr)/len(rr))
    concentrations={}
    for s in ('REM_to_W','REM_to_N1','REM_to_N2'):
        counts=sorted((sum(_stratum(r)==s for r in persons[p]) for p in persons),reverse=True)
        concentrations[s]={'people_with_error':sum(v>0 for v in counts),
                           'top_5_people_fraction':sum(counts[:5])/sum(counts),
                           'largest_person_error_count':counts[0]}
    status={'status':LABEL,'exact_published_confusion_reproduced':True,'confusion_matrix':confusion.tolist(),
            'scored_epochs':int(confusion.sum()),'REM_scored':int(confusion[4].sum()),
            'REM_recall':int(confusion[4,4])/int(confusion[4].sum()),
            'physical_epochs':sum(v['physical'] for v in per_record.values()),
            'recordings':len(per_record),'people':len(persons),
            'taxonomy':out,'condition_summary':{
                c:{'scored':len(rr),'REM_scored':sum(r['truth']==4 for r in rr),
                   'REM_correct':sum(r['truth']==4 and r['pred']==4 for r in rr)} for c,rr in conditions.items()},
            'participant_REM_distribution':{
                'people_with_scored_REM':sum(p['REM_scored']>0 for p in person_summary),
                'per_person_recall':_summary([p['REM_correct']/p['REM_scored'] for p in person_summary if p['REM_scored']]),
                'per_night_recall':_summary(night_rates),
                'per_night_min':min(night_rates),'per_night_max':max(night_rates),
                'error_concentration':concentrations,
                'min_REM_scored':min(p['REM_scored'] for p in person_summary),
                'max_REM_scored':max(p['REM_scored'] for p in person_summary)},
            'panels':{'E1':len(panel['E1']),'E2':len(panel['E2']),'E3':len(panel['E3']),
                      'E3_shortfalls':panel['E3_shortfalls'],'E2_status_counts':panel['E2_status_counts'],
                      'E1_E3_overlap':panel['E1_E3_overlap']},
            'privacy':'Only aggregates and hashed participant summaries; no per-epoch public rows',
            'human_event_validation':'NOT_TESTED'}
    return status,person_summary


def run():
    started=time.perf_counter();rows,confusion,per_record=collect()
    panel=select_panels(rows)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    new(PRIVATE/'frozen_panels.json',panel)
    # Panel SHA is public, coordinates are private and never in Git.
    new(PUBLIC/'panel_seal.json',{'status':LABEL,'private_panel_canonical_sha256':sha(canonical(panel)),
                                  'E1':len(panel['E1']),'E2':len(panel['E2']),'E3':len(panel['E3']),
                                  'E3_shortfalls':panel['E3_shortfalls'],'E1_E3_overlap':panel['E1_E3_overlap'],
                                  'selection_before_new_explanation_queries':True})
    result,people=analyze(rows,confusion,per_record,panel)
    new(PUBLIC/'rem_forensics.json',result)
    new(PRIVATE/'participant_night_forensics.json',people)
    elapsed=time.perf_counter()-started
    print(json.dumps({'ST_scored':len(rows),'REM_row':confusion[4].tolist(),
                      'panels':result['panels'],'wall_seconds':elapsed}))


if __name__=='__main__':run()

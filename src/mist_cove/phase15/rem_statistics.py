"""Participant-aware exploratory REM, calibration and warning summaries."""
import json
from collections import defaultdict

import numpy as np

from .forensics import collect,_stratum,_summary
from .governance import OLD_PRIVATE,OLD_PUBLIC,PRIVATE,PUBLIC,LABEL,read,new,verify_plan_commit


def _rates(flag,outcome):
    flag=np.asarray(flag,bool);outcome=np.asarray(outcome,bool)
    return {'n':len(flag),'positives':int(outcome.sum()),'warnings':int(flag.sum()),
            'sensitivity':float(flag[outcome].mean()) if outcome.any() else None,
            'specificity':float((~flag[~outcome]).mean()) if (~outcome).any() else None,
            'false_warning_rate':float(flag[~outcome].mean()) if (~outcome).any() else None,
            'precision':float(outcome[flag].mean()) if flag.any() else None}


def _ece(rr,bins=15):
    confidence=np.asarray([r['confidence'] for r in rr]);correct=np.asarray([r['pred']==r['truth'] for r in rr]);
    b=np.minimum((confidence*bins).astype(int),bins-1)
    rows=[]
    for i in range(bins):
        pick=b==i
        rows.append({'bin':i,'n':int(pick.sum()),'confidence':float(confidence[pick].mean()) if pick.any() else None,
                     'accuracy':float(correct[pick].mean()) if pick.any() else None})
    return {'n':len(rr),'ECE15':float(sum(x['n']*abs(x['confidence']-x['accuracy']) for x in rows if x['n'])/len(rr)),
            'bins':rows}


def _person_bootstrap(rows,wave,seed=15015,draws=1000):
    # Person is the resampling unit; epoch/night remain nested within person.
    by_person=defaultdict(list)
    for r in rows:
        if r['truth']==4:by_person[r['person']].append(r)
    keys=sorted(by_person);rng=np.random.default_rng(seed)
    metrics={'EOG_excursion_count':lambda w,r:w['EOG_excursion_count'],
             'Fpz_sigma_fraction':lambda w,r:r['features'][6],
             'Pz_sigma_fraction':lambda w,r:w['features']['EEG Pz-Oz']['sigma_fraction'],
             'ST_EMG_within_night_rank':lambda w,r:w['EMG_within_night_RMS_rank'],
             'Fpz_spectral_entropy':lambda w,r:w['features']['EEG Fpz-Cz']['spectral_entropy'],
             'within_2_epochs_transition':lambda w,r:float(r['distance']<=2)}
    contrasts={}
    for s in ('REM_to_W','REM_to_N1','REM_to_N2'):
        contrasts[s]={}
        for name,fn in metrics.items():
            pair=[]
            for person in keys:
                correct=[fn(wave[(r['recording'],r['epoch'])],r) for r in by_person[person] if _stratum(r)=='correct_REM']
                error=[fn(wave[(r['recording'],r['epoch'])],r) for r in by_person[person] if _stratum(r)==s]
                if correct and error:pair.append((float(np.mean(error))-float(np.mean(correct))))
            a=np.asarray(pair)
            sampled=np.mean(a[rng.integers(0,len(a),(draws,len(a)))],axis=1)
            contrasts[s][name]={'people_with_both':len(a),'person_mean_difference':float(a.mean()),
                                  'cluster_bootstrap_95CI':[float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))],
                                  'unit':'person; error minus correct REM'}
    return contrasts


def _transition_types(rows):
    geometry=read(OLD_PRIVATE/'geometry.json')['rows']
    labels={r['record']['recording']:np.asarray(r['labels']) for r in geometry if r['record']['role']=='ST_EVAL'}
    result=defaultdict(lambda:defaultdict(int))
    for r in rows:
        if r['truth']!=4:continue
        y=labels[r['recording']];i=r['epoch'];s=_stratum(r)
        lo=i
        while lo>0 and y[lo-1]>=0:lo-=1
        hi=i+1
        while hi<len(y) and y[hi]>=0:hi+=1
        for k in (1,2,4):
            found=[]
            for j in range(max(lo+1,i-k),min(hi,i+k+1)):
                if y[j]>=0 and y[j-1]>=0 and y[j]!=y[j-1]:found.append((abs(j-i),int(y[j-1]),int(y[j])))
            result[s][f'within_{k}_any']+=bool(found)
            if k==4 and found:
                _,a,b=min(found)
                result[s][f'nearest_{a}_to_{b}']+=1
        result[s]['recording_first_or_last_4']+=i<4 or i>=len(y)-4
        result[s]['long_stable_REM_5plus']+=all(0<=j<len(y) and y[j]==4 for j in range(i-4,i+5))
    return {s:dict(v) for s,v in result.items()}


def run():
    verify_plan_commit();rows,confusion,_=collect()
    wave_rows=read(PRIVATE/'ST_REM_wave_features.json')['rows']
    wave={(x['recording'],x['epoch']):x for x in wave_rows}
    if len(wave)!=8349:raise ValueError('REM waveform diagnostic denominator differs')
    full=_ece(rows);published=read(OLD_PUBLIC/'st_classification_metrics.json')['metrics']
    # Phase XIV exposed the full-grid ECE. Reproduction is an integrity test.
    oldece=published['calibration']['ECE15'] if 'calibration' in published else published.get('ECE15')
    if oldece is not None and abs(full['ECE15']-oldece)>1e-8:raise ValueError('published ECE differs')
    stage={str(j):_ece([r for r in rows if r['truth']==j]) for j in range(5)}
    rem=[r for r in rows if r['truth']==4];wrong=np.array([r['pred']!=4 for r in rem])
    warnings={
        'small_margin_lt_0_5':_rates([r['margin']<.5 for r in rem],wrong),
        'high_entropy_gt_1_1':_rates([r['entropy']>1.1 for r in rem],wrong),
        'high_confidence_ge_0_9':_rates([r['confidence']>=.9 for r in rem],wrong),
        'within_2_epochs_transition':_rates([r['distance']<=2 for r in rem],wrong)}
    highwrong=sum(r['pred']!=r['truth'] and r['confidence']>=.9 for r in rows)
    if highwrong!=3209:raise ValueError('published high-confidence wrong count differs')
    condition=read(OLD_PRIVATE/'conditions.json')['subjects'];paired=[]
    for person in sorted({r['person'] for r in rem}):
        by_night={}
        for night in (1,2):
            rr=[r for r in rem if r['person']==person and r['night']==night]
            if not rr:raise ValueError('missing paired REM night')
            by_night[condition[person][str(night)]]=sum(r['pred']==4 for r in rr)/len(rr)
        if set(by_night)!={'placebo','temazepam'}:raise ValueError('condition pairing differs')
        paired.append(by_night['temazepam']-by_night['placebo'])
    rng=np.random.default_rng(15015);a=np.asarray(paired)
    draws=a[rng.integers(0,len(a),(1000,len(a)))].mean(1)
    result={'status':LABEL,'full_grid_ECE15':full['ECE15'],'published_ECE15':oldece,
            'full_grid_calibration_bins':full['bins'],'stage_calibration':stage,
            'high_confidence_wrong_scored':highwrong,'REM_warning_indicators':warnings,
            'participant_aware_within_REM_contrasts':_person_bootstrap(rows,wave),
            'REM_transition_types':_transition_types(rows),
            'paired_condition_REM_recall_difference':{'people':len(a),'mean_temazepam_minus_placebo':float(a.mean()),
              'person_bootstrap_95CI':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]},
            'no_ST_threshold_fitting':True,'no_ST_neural_training':True,
            'interpretation':'Exploratory descriptive associations; 1000 participant-level bootstrap draws; no causal or clinical inference'}
    new(PUBLIC/'rem_statistics.json',result)
    print(json.dumps({'ECE15':full['ECE15'],'high_conf_wrong':highwrong,
                      'paired_condition_difference':result['paired_condition_REM_recall_difference']}))


if __name__=='__main__':run()

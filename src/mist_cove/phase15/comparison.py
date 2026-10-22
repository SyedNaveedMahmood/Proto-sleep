"""Aggregate sealed exact circuits versus fixed physiological proxy selectors."""
import json
from collections import defaultdict

import numpy as np

from mist_cove.phase13.annotation_metrics import mask_intervals,overlap_metrics
from mist_cove.phase14.event_schema import event_intervals,shifted_intervals
from .governance import PRIVATE,PUBLIC,OLD_PRIVATE,OLD_PUBLIC,LABEL,read,new,sha,canonical,verify_plan_commit


METHODS=('exact_positive','sigma_matched','RMS_matched','delta_matched',
         'gradient_matched','absolute_gradient_matched','random_matched','position_matched')
PROXY='EEG Pz-Oz|spindle'


def _mean(values):
    return float(np.mean(values)) if values else None


def _ci_person(pairs,seed=15015):
    groups=defaultdict(list)
    for person,delta in pairs:groups[person].append(float(delta))
    people=sorted(groups)
    if not people:return {'people':0,'cases':0,'person_mean_delta':None,'bootstrap95':None}
    a=np.array([np.mean(groups[p]) for p in people]);rng=np.random.default_rng(seed)
    draws=a[rng.integers(0,len(a),(1000,len(a)))].mean(1)
    return {'people':len(people),'cases':len(pairs),'person_mean_delta':float(a.mean()),
            'bootstrap95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
            'unit':'participant; nights and cases nested','status':LABEL}


def _case_key(c):return c['recording'],c['epoch']


def _eligible(r):
    positive=r['summary']['positive_minimum_mask']
    if positive is None:return False,'NO_POSITIVE_SUPPORT'
    k=int(positive).bit_count()
    if k==0:return False,'EMPTY_SUPPORT'
    if k==15:return False,'WHOLE_EPOCH_SUPPORT'
    if r['proxy_alignment'] is None:return False,'NO_ALIGNMENT'
    if r['proxy_alignment'][PROXY]['event_count']==0:return False,'NO_PZ_SPINDLE_CANDIDATE'
    return True,'ELIGIBLE'


def _case_overlap(r,method,tolerance='0'):
    return r['proxy_alignment'][PROXY]['methods'][method][tolerance]['precision']


def _panel(rows,coords,name):
    rr=[rows[_case_key(c)] for c in coords]
    eligible=[];reasons=defaultdict(int)
    for r in rr:
        ok,why=_eligible(r);reasons[why]+=1
        if ok:eligible.append(r)
    statistics={'panel':name,'status':LABEL,'sampling':('outcome_independent' if name=='E1' else
                'proxy_conditioned' if name=='E2' else 'model_error_enriched'),
                'cases':len(rr),'people':len({c['person'] for c in coords}),
                'eligibility':dict(reasons),'eligible_cases':len(eligible),
                'positive_minimum_exists':sum(r['summary']['positive_minimum_mask'] is not None for r in rr),
                'prior_response':sum(r['summary']['prior_response'] for r in rr),
                'whole_epoch_positive':sum(r['summary']['positive_minimum_mask']==32767 for r in rr),
                'nonmonotone_cases':sum(r['summary']['nonmonotone_edges']['response_lost_when_adding_atom']>0 for r in rr),
                'neutral_changed_tables':sum(r['neutral']['table_disagreement_fraction']>0 for r in rr),
                'neutral_changed_positive_minimum':sum(r['neutral']['positive_minimum_mask_changed'] for r in rr),
                'BDD_nodes':{'mean':_mean([r['summary']['BDD_nonterminal_nodes'] for r in rr]),
                             'median':float(np.median([r['summary']['BDD_nonterminal_nodes'] for r in rr])) if rr else None},
                'active_compact_signed':sum(any(c['negative_bits']==0 and 0<c['retained_seconds']<=8 and c['signed_literals']<=8
                     for c in r['summary']['signed']['clauses']) for r in rr),
                'signed_interference_cases':sum(any(c['negative_bits']!=0 for c in r['summary']['signed']['clauses']) for r in rr),
                'model_wrong':sum(r['truth']!=r['pred'] for r in rr),
                'methods':{},'paired_differences':{}}
    for tolerance in ('0','0.1','0.25'):
        method_values={m:[] for m in METHODS}
        for r in eligible:
            if any(r['candidates'][m].bit_count()!=r['candidates']['exact_positive'].bit_count() for m in METHODS):
                raise ValueError('unfair physical atom budget')
            for m in METHODS:method_values[m].append(_case_overlap(r,m,tolerance))
        statistics['methods'][tolerance]={m:{'n':len(v),'mean_precision':_mean(v),
                                             'median_precision':float(np.median(v)) if v else None}
                                          for m,v in method_values.items()}
    for comparator in ('sigma_matched','RMS_matched','delta_matched','random_matched',
                       'gradient_matched','absolute_gradient_matched'):
        pairs=[]
        for r in eligible:
            c=next(c for c in coords if _case_key(c)==_case_key(r['coordinate']))
            pairs.append((c['person'],_case_overlap(r,'exact_positive')-_case_overlap(r,comparator)))
        statistics['paired_differences']['exact_minus_'+comparator]=_ci_person(pairs)
    return statistics


def _negative_controls(rows,coords):
    eligible=[];event_cache={}
    for c in coords:
        r=rows[_case_key(c)]
        if _eligible(r)[0]:eligible.append((c,r))
    shifted=[];next_epoch=[]
    for c,r in eligible:
        rec=c['recording'];epoch=c['epoch']
        if rec not in event_cache:event_cache[rec]=read(OLD_PRIVATE/'ST_EVAL'/(rec+'_events.json'))
        ev=event_cache[rec];events=ev['events'].get(PROXY,[])
        here=event_intervals(events,epoch)
        other=event_intervals(events,(epoch+1)%ev['n_epochs'])
        mask=r['candidates']['exact_positive'];support=mask_intervals(mask)
        shifted.append(overlap_metrics(support,shifted_intervals(here,10),0)['precision'])
        next_epoch.append(overlap_metrics(support,other,0)['precision'])
    return {'eligible_cases':len(eligible),'proxy_shift_10_seconds_mean_precision':_mean(shifted),
            'next_Pz_epoch_within_recording_mean_precision':_mean(next_epoch),
            'not_proxy_ground_truth':True}


def _inherited_operator():
    previous=read(OLD_PUBLIC/'st_explanation_results.json')
    primary=previous['strata']['primary']
    return {'phase14_ST_cases':primary['cases'], 'phase14_exact_tables':read(OLD_PUBLIC/'ST_EVAL_witness_execution.json')['resources']['exact_tables'],
            'phase14_views':primary['views'],'phase14_physical':primary['physical'],
            'phase14_neutral_changed_tables':primary['neutral_changed_tables'],
            'phase14_neutral_changed_minimum':primary['neutral_minimum_changed'],
            'status':'INHERITED_EXPLORATORY_REUSED_ST','interpretation':'same frozen model, distinct internal and physical operators'}


def run():
    verify_plan_commit();execution=read(PUBLIC/'explanation_execution.json')
    path=PRIVATE/'explanation_rows.jsonl'
    if sha(path.read_bytes())!=execution['private_rows_sha256']:raise ValueError('private explanation rows changed')
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if len(rows)!=360 or not all(r['summary']['exact_all_rows'] for r in rows):
        raise ValueError('incomplete exact table coverage')
    lookup={_case_key(r['coordinate']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('duplicate explanation coordinate')
    panel=read(PRIVATE/'frozen_panels.json')
    if sha(canonical(panel))!=read(PUBLIC/'panel_seal.json')['private_panel_canonical_sha256']:
        raise ValueError('panel changed')
    results={name:_panel(lookup,panel[name],name) for name in ('E1','E2','E3')}
    result={'status':LABEL,'panels':results,'negative_controls_E2':_negative_controls(lookup,panel['E2']),
            'method_access':{
                'exact_positive':'model cached head/full 32768 masks, own predicted class; no human labels or proxy locations',
                'signed_certificate':'same exact table; positive and negative literals; only active positive literals are waveform support',
                'greedy_own_budget':'same exact table; own support length, not falsely equal duration',
                'sigma_RMS_delta':'original Fpz-Cz waveform only, matched physical atoms, no model or proxy or labels',
                'gradient_absolute_gradient':'original model gradient and cached features; matched atoms, no proxy or labels',
                'random_position':'seeded fixed controls, matched atoms, no model or proxy or labels'},
            'formal_model_faithfulness':'Exact minimum and signed clauses verified against all 32768 masks for each of 360 new cases; comparator masks are interval selections, not certified sufficient reasons',
            'event_proxy_target':'Unvalidated stage-blind Pz-Oz spindle-like candidates, not expert-confirmed events',
            'registered_phase14_sigma_increment_gate':'FAIL_UNCHANGED',
            'human_event_validation':'NOT_TESTED','new_success_gate':None}
    new(PUBLIC/'explanation_comparison.json',result)
    new(PUBLIC/'operator_inheritance.json',_inherited_operator())
    print(json.dumps({'E1':results['E1']['eligible_cases'],'E2':results['E2']['eligible_cases'],
                      'E3':results['E3']['eligible_cases'],
                      'E2_exact_minus_sigma':results['E2']['paired_differences']['exact_minus_sigma_matched']}))


if __name__=='__main__':run()

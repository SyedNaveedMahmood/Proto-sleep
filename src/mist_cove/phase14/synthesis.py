"""Cross-study unpaired and condition-paired scientific synthesis."""
import json
from collections import defaultdict
import numpy as np
from mist_cove.phase11b.evaluation import clustered_mean
from .common import PRIVATE,PUBLIC,read_json,write_new,measured,ledger
from .statistics import load,stacked,FEATURES


def unpaired_difference(sc,st):
    a=np.asarray(sc,dtype=float);b=np.asarray(st,dtype=float)
    if not len(a) or not len(b):return {'SC_people':len(a),'ST_people':len(b),'mean':None,'bootstrap95':None}
    rng=np.random.default_rng(123);draws=b[rng.integers(0,len(b),(1000,len(b)))].mean(1)-a[rng.integers(0,len(a),(1000,len(a)))].mean(1)
    return {'SC_people':len(a),'ST_people':len(b),'mean':float(b.mean()-a.mean()),'bootstrap95':np.quantile(draws,[.025,.975]).tolist(),'method':'unpaired person bootstrap1000/seed123; withinpersonnightsretained'}


def condition_witnesses():
    rows=[json.loads(line) for line in (PRIVATE/'ST_EVAL_witness_rows.jsonl').read_text().splitlines()]
    lookup={r['record']['recording']:r['record'] for r in read_json(PRIVATE/'geometry.json')['rows'] if r['record']['role']=='ST_EVAL'}
    conditions=read_json(PRIVATE/'conditions.json')['subjects'];pairs=defaultdict(dict)
    for row in rows:
        c=row['coordinate'];r=lookup[c['recording_id']];condition=conditions[c['subject']][str(r['night'])];s=row['summary']
        compact=any(x['negative_bits']==0 and 0<x['retained_seconds']<=8 and x['signed_literals']<=8 for x in s['signed']['clauses'])
        a=row['proxy_alignment']['proxies'].get('EEG Pz-Oz|spindle',{})
        precision=a.get('methods',{}).get('exact_positive',{}).get('0',{}).get('precision') if a.get('event_eligible') else None
        pairs[c['subject'],c['stage']][condition]={'BDD_nodes':s['BDD_nonterminal_nodes'],'compact_active':float(compact),
             'global_stable_local_changed':float(row['branch_predictions']['global_only']==row['prediction'] and row['branch_predictions']['local_only']!=row['prediction']),
             'spindle_proxy_overlap':precision,'correct':float(row['correct'])}
    result={}
    for key in ('BDD_nodes','compact_active','global_stable_local_changed','spindle_proxy_overlap','correct'):
        grouped=defaultdict(list);stage_pairs=0
        for (person,stage),pair in pairs.items():
            if set(pair)=={'temazepam','placebo'} and all(pair[k][key] is not None for k in pair):grouped[person].append(pair['temazepam'][key]-pair['placebo'][key]);stage_pairs+=1
        people=sorted(grouped);values=[np.mean(grouped[p]) for p in people]
        result[key]={'paired_person_stage_strata':stage_pairs,'person_mean_delta':clustered_mean(values,people)}
    write_new(PUBLIC/'st_condition_witness_aggregates.json',{'sampling':'first perstage/person/night, up to220; missingstrata not replaced','contrast':'temazepam-placebo within same person/stage, averaged withinperson','results':result,'causal_effect':'NOT_ESTABLISHED'})


def run():
    with measured('cross_study_synthesis'):
        data={role:load(role) for role in ('SC_TRAIN','ST_EVAL')};features={};persons={}
        for role in data:
            y,z,p,f,t=stacked(data[role]);persons[role]=sorted(set(p));features[role]={}
            for j,key in enumerate(FEATURES):
                features[role][key]=[np.mean([f[(p==person)&(y==stage),j].mean() for stage in range(5) if ((p==person)&(y==stage)).any()]) for person in persons[role]]
        contrast={}
        for key in FEATURES:
            if key.startswith('EMG_') or key=='marker_change_fraction':contrast[key]={'status':'NOT_COMPARABLE_ACQUISITION'}
            else:contrast[key]=unpaired_difference(features['SC_TRAIN'][key],features['ST_EVAL'][key])
        a=read_json(PUBLIC/'sc_classification_metrics.json');b=read_json(PUBLIC/'st_classification_metrics.json');x=read_json(PUBLIC/'sc_alignment_results.json');z=read_json(PUBLIC/'st_explanation_results.json')
        old_val=read_json(PUBLIC.parent/'phase11b_v1'/'val_descriptive_metrics.json')
        result={'signal_feature_contrasts_ST_minus_SC_stage_balanced':contrast,'classification':{'SC_current_in_sample':a['metrics'],'ST_frozen':b['metrics'],'historical_reused_VAL_aggregate':old_val},
            'explanations':{'SC_stagebalanced':x['strata']['primary'],'SC_prevalence':x['strata']['prevalence'],'ST_stagebalanced':z['strata']['primary']},
            'cohort_caveats':['SC historically trained/developed, ST frozen cross-study','different people, acquisition, medication and daytimeWakeprevalence','ST/SC physical-person linkage not independently established','sample rank/night policies differ','rawEMG/envelope and marker rates not comparable','multiple correlated outcomes, no population causal claims'],
            'expert_event_validation':'NOT_TESTED'}
        write_new(PUBLIC/'cross_study_metrics.json',result)
        condition_witnesses()
        gates={'SC_proxy':x['primary_gates'],'ST_proxy':z['primary_gates'],'ST_transfer':b['transfer_gate'],'SC_context':read_json(PUBLIC/'context_results.json')['primary_gate'],
               'exact_operator_correctness':'PASS','human_event_validation':'NOT_TESTED','clinical_event_deletion':'NOT_TESTED','external_clinical_validation':'NOT_TESTED'}
        write_new(PUBLIC/'scientific_gates.json',gates)
        write_new(PUBLIC/'robustness_metrics.json',{'SC':x['strata']['primary']['views'],'ST':z['strata']['primary']['views'],
            'neutral':{role:{k:r['strata']['primary'][k] for k in ('cases','neutral_cases','neutral_changed_tables','neutral_minimum_changed')} for role,r in [('SC',x),('ST',z)]},
            'physical':{'SC':x['strata']['primary']['physical'],'ST':z['strata']['primary']['physical']},
            'interpretation':'fixed-target original clauses, active substitution distinct; rejected replacements preserved; no clinical label-preservation claim'})
        ledger([44,45,46,47],'COMPLETE','Structural, physical and automated-proxy views synthesized separately; numerical gates preserve all negatives; person/night clustering and confounding explicit',
            ['reports/mist_cove/phase14_v1/scientific_gates.json','reports/mist_cove/phase14_v1/cross_study_metrics.json'],science='MIXED')

"""Participant-aware proxy, classification and paired-condition summaries."""
import json
from collections import defaultdict
import numpy as np
from mist_cove.phase11b.evaluation import public_metrics,distribution,clustered_mean
from mist_transfer.metrics import classification_metrics
from .common import ROOT,PRIVATE,PUBLIC,read_json,write_new,measured,ledger
from .signal_preprocessing import FEATURES
from .event_alignment import interval_concordance
from .event_schema import event_intervals


def records(role):
    return [r['record'] for r in read_json(PRIVATE/'geometry.json')['rows'] if r['record']['role']==role]


def load(role):
    result=[]
    for r in records(role):
        path=PRIVATE/role/(r['recording']+'_epochs.npz')
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        result.append((r,d,read_json(PRIVATE/role/(r['recording']+'_events.json'))))
    return result


def stacked(data,model=None):
    y=[];z=[];people=[];features=[];transition=[]
    for r,d,e in data:
        y.extend(d['truth']);z.extend(d['logits'] if model is None else np.load(PRIVATE/r['role']/(r['recording']+'_'+model+'_logits.npy'),allow_pickle=False))
        people.extend([r['subject']]*len(d['truth']));features.extend(d['features']);transition.extend(d['transition'])
    return np.array(y),np.array(z),np.array(people),np.array(features),np.array(transition)


def rate_by_person(data,key):
    counts=defaultdict(int);seconds=defaultdict(float)
    for r,d,e in data:counts[r['subject']]+=len(e['events'].get(key,[]));seconds[r['subject']]+=e['eligible_seconds']
    people=sorted(seconds);return np.array([3600*counts[p]/seconds[p] for p in people]),people


def proxy_summary(role,data):
    keys=sorted(set().union(*(set(e['events']) for r,d,e in data)));summary={}
    for key in keys:
        rates,people=rate_by_person(data,key);ev=[v for r,d,e in data for v in e['events'].get(key,[])]
        durations=[v['end']-v['start'] for v in ev];frequency=[v['Frequency'] for v in ev if v.get('Frequency') is not None];amplitude=[v.get('Amplitude',v.get('PTP')) for v in ev if v.get('Amplitude',v.get('PTP')) is not None]
        stage_counts=np.zeros(6,int);stage_seconds=np.zeros(6,float)
        for r,d,e in data:
            truth=d['truth'];stage_seconds+=np.bincount(np.where(truth>=0,truth,5),minlength=6)*30
            for event in e['events'].get(key,[]):
                index=int(((event['start']+event['end'])/2)//30)
                if index<len(truth):stage_counts[int(truth[index]) if truth[index]>=0 else 5]+=1
        summary[key]={'event_count':len(ev),'person_rate_per_hour_distribution':distribution(rates),'person_mean_rate':clustered_mean(rates,people),
            'duration_seconds':distribution(durations),'frequency_Hz':distribution(frequency),'amplitude_uV':distribution(amplitude),
            'stage_and_unscored_event_counts':stage_counts.tolist(),'stage_and_unscored_hours':(stage_seconds/3600).tolist(),
            'stage_conditioning':'post-detection descriptive, primary detector stage-blind','clinical_event_accuracy':'NOT_TESTED'}
    concordance={}
    for kind in ('spindle','spindle_strict','slow_wave','slow_wave_relaxed','sigma','delta'):
        cc=[interval_concordance(e['events'].get('EEG Fpz-Cz|'+kind,[]),e['events'].get('EEG Pz-Oz|'+kind,[])) for r,d,e in data]
        values=[v['F1'] for v in cc if v['F1'] is not None]
        concordance[kind]={'first_events':sum(v['first_events'] for v in cc),'second_events':sum(v['second_events'] for v in cc),'matched':sum(v['matched'] for v in cc),
            'recording_F1_distribution':distribution(values),'recordings_estimated':len(values),'recordings_unestimable':len(cc)-len(values),
            'pooled_F1':2*sum(v['matched'] for v in cc)/max(1,sum(v['first_events']+v['second_events'] for v in cc)),'human_truth':False}
    y,z,people,f,t=stacked(data);signal={}
    for j,key in enumerate(FEATURES):
        unit_values=np.array([f[(people==p)&(y>=0),j].mean() for p in sorted(set(people))]);unit_people=sorted(set(people))
        # Stage-balanced descriptive values reduce, but do not remove, study confounding.
        staged=[]
        for p in unit_people:
            vals=[f[(people==p)&(y==stage),j].mean() for stage in range(5) if ((people==p)&(y==stage)).any()];staged.append(np.mean(vals))
        err=z.argmax(1)!=y;differences=[];subjects=[]
        for p in unit_people:
            delta=[]
            for stage in range(5):
                a=(people==p)&(y==stage)&err;b=(people==p)&(y==stage)&~err
                if a.any() and b.any():delta.append(f[a,j].mean()-f[b,j].mean())
            if delta:differences.append(np.mean(delta));subjects.append(p)
        signal[key]={'scored_person_mean':clustered_mean(unit_values,unit_people),'stage_balanced_person_mean':clustered_mean(staged,unit_people),
                     'person_stage_matched_wrong_minus_correct':clustered_mean(differences,subjects)}
    coh=np.array([[e['coherence'][j] if e['coherence'][j] is not None else np.nan for j in range(2)] for r,d,e in data])
    result={'role':role,'recordings':len(data),'participants':len(set(people)),'physical_epochs':len(y),'scored_epochs':int((y>=0).sum()),'unscored_epochs':int((y<0).sum()),
            'primary_detection_stage_blind':True,'proxies':summary,'EEG_cross_channel_concordance':concordance,'signal_distributions':signal,
            'coherence_delta':distribution(coh[np.isfinite(coh[:,0]),0]),'coherence_sigma':distribution(coh[np.isfinite(coh[:,1]),1]),
            'detector_failures':sum(len(e['failures']) for r,d,e in data),'eligible_detector_seconds':sum(e['eligible_seconds'] for r,d,e in data),
            'limitations':['automated proxies, not expert event labels','YASA amplitude/montage criteria unvalidated in this bipolar configuration','correlated channels are not independent experts','SCenvelope andSTrawEMG are not interchangeable'],
            'stage_label_source':'originalR&K epoch stages, not event locations'}
    write_new(PUBLIC/('sc_proxy_aggregates.json' if role=='SC_TRAIN' else 'st_proxy_results.json'),result)
    return result


def stage_classification(role,data):
    y,z,people,f,transition=stacked(data);metrics=public_metrics(y,z,people);pred=z.argmax(1);counts=np.bincount(pred,minlength=5)
    night_f1=[classification_metrics(d['truth'],d['logits'].argmax(1),np.full(len(d['truth']),r['subject']))['macro_f1'] for r,d,e in data]
    result={'role':role,'metrics':metrics,'physical_epochs':len(y),'unscored_epochs':int((y<0).sum()),'people':len(set(people)),
            'predicted_counts_all_physical':counts.tolist(),'largest_predicted_fraction':float(counts.max()/counts.sum()),'predicted_classes':int((counts>0).sum()),
            'night_macro_F1_distribution':distribution(night_f1),'worst_subject_macro_F1':metrics['subject_macro_f1_distribution']['min'],
            'transition':public_metrics(y[transition],z[transition],people[transition]),'human_event_validation':'NOT_TESTED','original_model_adapted':False}
    if role=='ST_EVAL':
        m=metrics;result['transfer_gate']='PASS' if m['mean_subject_macro_f1']>=.5 and m['per_stage_recall'][1]>=.2 and min(m['per_stage_f1'])>=.1 and (counts>0).sum()==5 and counts.max()/counts.sum()<.98 else 'FAIL'
        context=[]
        for arm in ('C0','C1'):
            for seed in (123,456,789):
                yy,zz,ss,ff,tt=stacked(data,f'{arm}_{seed}');context.append({'arm':arm,'seed':seed,'metrics':public_metrics(yy,zz,ss),'transition':public_metrics(yy[tt],zz[tt],ss[tt]),'promotion_from_ST':False})
        result['context_frozen_comparators']=context
    write_new(PUBLIC/('sc_classification_metrics.json' if role=='SC_TRAIN' else 'st_classification_metrics.json'),result)
    return result


def condition_summary(data):
    conditions=read_json(PRIVATE/'conditions.json')['subjects'];paired={};night_summary=[]
    for r,d,e in data:
        label=conditions[r['subject']][str(r['night'])];y=d['truth'];z=d['logits'];mask=y>=0
        metrics=public_metrics(y,z,np.full(len(y),r['subject']));p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        values={'accuracy':metrics['accuracy'],'macro_F1':metrics['macro_f1'],'confidence':float(p[mask].max(1).mean()),'N1_recall':metrics['per_stage_recall'][1],
            'spindle_rate_per_hour':3600*len(e['events']['EEG Fpz-Cz|spindle'])/e['eligible_seconds'],
            'slow_wave_rate_per_hour':3600*len(e['events']['EEG Fpz-Cz|slow_wave'])/e['eligible_seconds']}
        for j,name in enumerate(FEATURES):values[name]=float(d['features'][mask,j].mean())
        for stage in range(5):values['prevalence_'+str(stage)]=float((y[mask]==stage).mean());values['stage_F1_'+str(stage)]=metrics['per_stage_f1'][stage]
        paired.setdefault(r['subject'],{})[label]=values
    contrasts={}
    for key in next(iter(next(iter(paired.values())).values())):
        values=[];people=[]
        for person,pair in paired.items():
            if set(pair)=={'placebo','temazepam'} and pair['temazepam'][key] is not None and pair['placebo'][key] is not None:
                values.append(pair['temazepam'][key]-pair['placebo'][key]);people.append(person)
        contrasts[key]=clustered_mean(values,people)
    write_new(PRIVATE/'ST_condition_night_summaries.json',paired)
    result={'paired_subjects':sum(set(v)=={'placebo','temazepam'} for v in paired.values()),'condition_assignment':'authenticated officialSTspreadsheet nightcolumns, no filenamecondition guessing',
        'contrast':'temazepam minusplacebo withinperson','metrics':contrasts,'causal_drug_effect':'NOT_ESTABLISHED','multiple_comparisons':'descriptive exploratory associations; no multiplicity-adjusted causal claims'}
    write_new(PUBLIC/'st_condition_aggregates.json',result);return result


def proxy_alignment_summary(role):
    rows=[json.loads(line) for line in (PRIVATE/(role+'_witness_rows.jsonl')).read_text().splitlines()]
    fit=set(read_json(PRIVATE/'selection.json')['allocation']['fit']);groups={'primary':[r for r in rows if r['primary']], 'correct':[r for r in rows if r['primary'] and r['correct']],
        'wrong':[r for r in rows if r['primary'] and not r['correct']]}
    if role=='SC_TRAIN':groups.update({'fit_primary':[r for r in rows if r['primary'] and r['coordinate']['subject'] in fit],
        'development_primary':[r for r in rows if r['primary'] and r['coordinate']['subject'] not in fit],'prevalence':[r for r in rows if not r['primary']]})
    result={}
    for label,rr in groups.items():
        proxykeys=sorted(set().union(*(set(r['proxy_alignment']['proxies']) for r in rr))) if rr else []
        summaries={}
        for key in proxykeys:
            values=defaultdict(list);units=defaultdict(list);eligible=0;eligible_people=set();failures=0
            for r in rr:
                a=r['proxy_alignment']['proxies'].get(key)
                if a is None:continue
                statuses=r['detector_record_statuses'].get(key,{})
                if 'DETECTOR_FAILED' in statuses or 'UNASSESSABLE_FLAT' in statuses:failures+=1;continue
                if not a['event_eligible'] or a['support_atoms']==0:continue
                eligible+=1;eligible_people.add(r['coordinate']['subject'])
                for method,metrics in a['methods'].items():
                    m=metrics['0'];p=m['precision'];coverage=m['event_coverage']
                    if p is not None:values[method+'_precision_like'].append(p);units[method+'_precision_like'].append(r['coordinate']['subject'])
                    if coverage is not None:values[method+'_event_coverage'].append(coverage);units[method+'_event_coverage'].append(r['coordinate']['subject'])
                    if method in ('exact_positive','greedy_own_budget','proxy_shift10s','cross_channel_permuted_same_recording'):continue
                    delta=a['methods']['exact_positive']['0']['precision']-p
                    values['exact_minus_'+method].append(delta);units['exact_minus_'+method].append(r['coordinate']['subject'])
            summaries[key]={'event_eligible_nonempty_cases':eligible,'people_with_eligible_events':len(eligible_people),'detector_failed_or_flat_cases':failures,
                            'methods':{k:clustered_mean(v,units[k]) for k,v in values.items()}}
        compact=lambda r:any(c['negative_bits']==0 and 0<c['retained_seconds']<=8 and c['signed_literals']<=8 for c in r['summary']['signed']['clauses'])
        support=[r['summary']['positive_minimum_mask'] for r in rr];views={}
        if rr and rr[0]['views']:
            for view in rr[0]['views']:
                vv=[r['views'][view] for r in rr];views[view]={'cases':len(vv),'prediction_stable':sum(v['prediction_stable'] for v in vv),
                   'fixed_clause_eligible':sum(v['fixed_clause_valid'] is not None for v in vv),'fixed_clause_valid':sum(v['fixed_clause_valid'] is True for v in vv),
                   'active_substitution':sum(v['active_substitution'] for v in vv),'any_original_active_valid':sum(v['any_original_active_clause_valid'] for v in vv)}
        physical={}
        for kind in ('bridge','background'):
            attempts=[p for r in rr for p in r['replacements'] if p['kind']==kind];compatible=[p for p in attempts if p['status']=='ATTEMPTED' and p['historical_compatibility']]
            edge=[e for r in rr if 'triad' in r for e in r['triad']['edges'] if e['type']=='OPERATOR_COMPARISON' and e['operator']==kind]
            physical[kind]={'requests':len(attempts),'attempted':sum(v['status']=='ATTEMPTED' for v in attempts),'compatible':len(compatible),
                'nontarget_exact':sum(v['status']=='ATTEMPTED' and v['nontarget_exact'] for v in attempts),'target_response_lost':sum(not v['target_retained'] for v in compatible),
                'structural_physical_disagreements':sum(e['disagrees'] for e in edge),'not_estimable':not bool(compatible),'clinical_event_deletion':'NOT_TESTED'}
        result[label]={'cases':len(rr),'people':len({r['coordinate']['subject'] for r in rr}),'correct':sum(r['correct'] for r in rr),'exact_tables':sum(r['summary']['exact_all_rows'] for r in rr),
            'exact_nonempty_compact_positive':sum(v is not None and 0<v.bit_count()<=4 for v in support),'displayed_compact_active':sum(compact(r) for r in rr),
            'whole_epoch_positive':sum(v==32767 for v in support),'empty_universal_positive':sum(v==0 for v in support),'missing_positive':sum(v is None for v in support),
            'prior_response':sum(r['summary']['prior_response'] for r in rr),'nonmonotone':sum(r['summary']['nonmonotone_edges']['response_lost_when_adding_atom']>0 for r in rr),
            'BDD_nodes':distribution([r['summary']['BDD_nonterminal_nodes'] for r in rr]),'displayed_clause_state_coverage':distribution([r['summary']['signed']['coverage'] for r in rr]),
            'global_stable_local_changed':sum(r['branch_predictions']['global_only']==r['prediction'] and r['branch_predictions']['local_only']!=r['prediction'] for r in rr),
            'partial_confidence':{k:distribution([r['partial_confidence'][k] for r in rr if k in r['partial_confidence']]) for k in ('empty','exact_positive','greedy')},
            'neutral_cases':sum('neutral' in r for r in rr),'neutral_changed_tables':sum(r.get('neutral',{}).get('table_disagreement_fraction',0)>0 for r in rr),
            'neutral_minimum_changed':sum(r.get('neutral',{}).get('positive_minimum_mask_changed',False) for r in rr),'views':views,'physical':physical,'proxy_concordance':summaries}
    primary=result['development_primary'] if role=='SC_TRAIN' else result['primary'];endpoint=primary['proxy_concordance'].get('EEG Pz-Oz|spindle',{})
    gates={}
    for name,control in [('proxy_primary','random_matched'),('spectral_increment','sigma_matched')]:
        contrast=endpoint.get('methods',{}).get('exact_minus_'+control,{})
        enough=endpoint.get('event_eligible_nonempty_cases',0)>=20 and endpoint.get('people_with_eligible_events',0)>=5
        gates[name]={'status':'NOT_ESTIMABLE' if not enough else 'PASS' if contrast['mean']>=.05 and contrast['bootstrap95'][0]>0 else 'FAIL',
                     'contrast':contrast,'eligible_cases':endpoint.get('event_eligible_nonempty_cases',0),'eligible_people':endpoint.get('people_with_eligible_events',0)}
    output={'role':role,'strata':result,'primary_gates':gates,'precision_like':'fraction ofselectedseconds overlappingautomatedproxy, notclinicalprecision','human_validation':'NOT_TESTED'}
    write_new(PUBLIC/('sc_alignment_results.json' if role=='SC_TRAIN' else 'st_explanation_results.json'),output);return output


def summarize(role):
    with measured(role+'_statistical_analysis'):
        data=load(role);proxy=proxy_summary(role,data);classification=stage_classification(role,data)
        if (PRIVATE/(role+'_witness_rows.jsonl')).exists():alignment=proxy_alignment_summary(role)
        if role=='ST_EVAL':condition_summary(data)
        ledger([19,20,21] if role=='SC_TRAIN' else [33,34,35,36,37,38,39,40],'COMPLETE','Full authorized recordings, detector concordance, signal/quality/error and classification/condition summaries retained with person-level uncertainty',
            ['reports/mist_cove/phase14_v1/'+('sc_proxy_aggregates.json' if role=='SC_TRAIN' else 'st_proxy_results.json')],science='PENDING_SYNTHESIS')

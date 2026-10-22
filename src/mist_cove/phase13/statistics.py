"""Participant-clustered developmental contrasts; no epoch-as-person inference."""
import json
import numpy as np
from .provenance import PUBLIC,PRIVATE,write_new
from .gates import absolute_adequacy,width_gain,exposure_gain,routing_gain
from .data import load_cache
from mist_transfer.metrics import classification_metrics


def clustered_interval(values,seed=123):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('finite seeds by matched people required')
    rng=np.random.default_rng(seed);draw=rng.integers(0,a.shape[1],size=(1000,a.shape[1]))
    means=a[:,draw].mean(axis=(0,2))
    return {'mean':float(a.mean()),'seed_means':a.mean(1).tolist(),'seed_SD':float(a.mean(1).std(ddof=1)) if len(a)>1 else None,
            'participant_cluster_bootstrap_95_percent':np.quantile(means,[.025,.975]).tolist(),
            'people':a.shape[1],'seeds':a.shape[0],'bootstrap_draws':1000,'bootstrap_seed':seed,
            'interpretation':'descriptive previously studied TRAIN development people; seeds retained inside each person'}


def rows():
    return {(s,r,k,t):json.loads((PUBLIC/f'{s}_{r}_{k}_{t}.json').read_text())
            for s in ('S','L') for r in ('R0','R1') for k in (123,456,789) for t in (300,1000,3000)}


def summarize():
    records=rows();data=load_cache();n=int(data['n_fit']);y=data['truth'][n:];subjects=data['subjects'][n:]
    seeds=(123,456,789);people=sorted(set(subjects));private_metrics={};group=[]
    for (size,route,seed,step),row in records.items():
        path=PRIVATE/'model_states'/f'{size}_{route}_{seed}'/f'predictions_{step}.npz'
        with np.load(path,allow_pickle=False) as archive:
            pred={v:archive[v+'_logits'].argmax(1) for v in row['development']}
        private_metrics[size,route,seed,step]={v:np.array([classification_metrics(y[subjects==p],a[subjects==p],subjects[subjects==p])['macro_f1'] for p in people]) for v,a in pred.items()}
        if step!=3000:continue
        for view,a in pred.items():
            for factor,mask in [('transition',data['transition'][n:]),('nontransition',~data['transition'][n:]),
                                ('high_quality_proxy',data['quality'][n:,4]<=np.median(data['quality'][n:,4])),
                                ('low_quality_proxy',data['quality'][n:,4]>np.median(data['quality'][n:,4]))]:
                if not mask.any():continue
                m=classification_metrics(y[mask],a[mask],subjects[mask])
                group.append({'capacity':size,'routing':route,'seed':seed,'view':view,'stratum':factor,'epochs':int(mask.sum()),
                              'people':len(set(subjects[mask])),'macro_F1':m['macro_f1'],'per_stage_recall':m['per_stage_recall']})
    contrasts={}
    definitions={'width':(('L','R0',3000),('S','R0',3000)),
                 'exposure':(('S','R0',3000),('S','R0',300)),
                 'routing_large':(('L','R1',3000),('L','R0',3000)),
                 'routing_small':(('S','R1',3000),('S','R0',3000))}
    for label,(a,b) in definitions.items():
        contrasts[label]={}
        for view in records[a[0],a[1],123,a[2]]['development']:
            values=np.stack([private_metrics[a[0],a[1],k,a[2]][view]-private_metrics[b[0],b[1],k,b[2]][view] for k in seeds])
            contrasts[label][view]=clustered_interval(values)
    gates={'absolute_adequacy':{},'width':{},'exposure':{},'routing':{}}
    for k in seeds:
        gates['absolute_adequacy'][str(k)]=absolute_adequacy(records['L','R0',k,3000])
        gates['width'][str(k)]=width_gain(records['L','R0',k,3000],records['S','R0',k,3000])
        gates['exposure'][str(k)]=exposure_gain(records['S','R0',k,3000],records['S','R0',k,300])
        gates['routing'][str(k)]=routing_gain(records['L','R1',k,3000],records['L','R0',k,3000])
    write_new(PUBLIC/'scientific_gates.json',{k:{'by_seed':v,'status':'PASS' if all(v.values()) else 'FAIL'} for k,v in gates.items()})
    write_new(PUBLIC/'uncertainty_summary.json',{'contrasts':contrasts,'independent_development_people':5,
               'limitations':['TRAIN people previously fitted by historical P3B','3seeds are not3new cohorts','multiple exploratory views, no multiplicity-adjusted discovery claims']})
    summary=[]
    for s in ('S','L'):
        for r in ('R0','R1'):
            for t in (300,1000,3000):
                rr=[records[s,r,k,t] for k in seeds];m=[v['development']['clean']['metrics'] for v in rr]
                summary.append({'capacity':s,'routing':r,'step':t,'stored_parameters':rr[0]['counts']['stored'],
                    'participant_macro_F1':[v['mean_subject_macro_f1'] for v in m],
                    'N1_recall':[v['per_stage_recall'][1] for v in m],
                    'fit_monitor_macro_F1':[v['fit_monitor']['metrics']['macro_f1'] for v in rr],
                    'clean_NLL':[v['calibration']['NLL'] for v in m],
                    'worst_participant_F1':float(min(private_metrics[s,r,k,t]['clean'].min() for k in seeds))})
    write_new(PUBLIC/'adequacy_summary.json',{'seeds':list(seeds),'rows':summary,'all_milestones':36,'fit_people':10,'development_people':5,
               'no_VAL_evaluation':True,'selection':'fixed final3000','clinical_ambiguity':'NOT_TESTED','context':'transition proxy uses preceding physical epoch, no future labels as inputs'})
    write_new(PUBLIC/'per_seed_results.json',{'rows':[{k:records[key][k] for k in ('capacity','routing','seed','step','training_loss_last100','gradient_norm_last100','parameter_drift_L2','checkpoint_byte_sha256','state_digest','stream_sha256')} for key in sorted(records)],'detailed_metrics':'each named capacity_routing_seed_step JSON retains all7view metrics'})
    write_new(PUBLIC/'stage_context_analysis.json',{'rows':group,'quality':'unvalidated signal-quality proxy, median TRAIN development stratification, not medical artifact labels',
               'context_causality':'NOT_TESTED; descriptive stage/transition association only','transition':'previous physical scored stage differs within recording'})
    print(json.dumps({k:all(v.values()) for k,v in gates.items()}))

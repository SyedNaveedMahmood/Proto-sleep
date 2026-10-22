"""One source pass per recording, physical-unit proxies and private predictions."""
import json
import resource
import time
import numpy as np
import torch
from .common import ROOT,PRIVATE,PUBLIC,read_json,write_new,measured,ledger,check_budget,sha256,canonical
from .edf_reader import MultiReader
from .proxies import Detector
from .signal_preprocessing import segment_bounds,epoch_features,FEATURES
from mist_cove.phase11b.historical import authenticate_model,named_state_digest,STATE_SHA


@torch.no_grad()
def native(model,wave,norm):
    logits=[]
    for start in range(0,len(wave),16):
        x=torch.from_numpy(((wave[start:start+16]-norm['mean_uV'])/norm['std_uV']).astype('float32'))
        logits.append(model.emissions(model.encode(x)[None])[0].numpy())
    return np.concatenate(logits)


def cached_selection():
    value=read_json(PRIVATE/'selection.json');receipt=read_json(PUBLIC/'protocol_preregistration.json')
    if sha256(canonical(value))!=receipt['selection_sha256']:raise ValueError('selection identity differs')
    return value


def run(role):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if role not in ('SC_TRAIN','ST_EVAL'):raise ValueError('role denied')
    if role=='ST_EVAL':
        from .protocol import verify_st_seal
        verify_st_seal()
    if (PRIVATE/(role+'_started.json')).exists():raise ValueError('source campaign already attempted')
    config=read_json(ROOT/'configs/mist_cove_phase14_v1.json')
    if read_json(PUBLIC/'detector_manifest.json')['status']!='PASS':raise ValueError('detector pilot gate')
    write_new(PRIVATE/(role+'_started.json'),{'config_sha256':sha256(canonical(config))})
    started=time.perf_counter();selection=cached_selection();source=MultiReader();geometry=read_json(PRIVATE/'geometry.json')['rows'];geom={r['record']['recording']:r for r in geometry}
    model,payload,root,child,auth=authenticate_model();norm=source.manifest['normalization'];worker=Detector();execution=[]
    training=selection['training']['fit']+selection['training']['internal_dev'];fitn=len(selection['training']['fit'])
    if role=='SC_TRAIN':
        training_x=np.zeros((len(training),3,1,3000),np.float32);training_valid=np.zeros((len(training),3),bool)
        samples=selection['primary']+selection['prevalence'];sample_waves={k:np.empty((len(samples),3000),np.float32) for k in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal')}
    else:
        samples=[]
        # Final prospective STseal refines the initial pooled-night proposal to
        # one per stage/night for identifiable paired-condition summaries.
        for person in sorted({r['record']['subject'] for r in geometry if r['record']['role']==role}):
            for stage in range(5):
                for night in (1,2):
                    pool=[{'recording_id':r['record']['recording'],'subject':person,'stage':stage,'epoch':i}
                          for r in sorted(geometry,key=lambda a:a['record']['recording'])
                          if r['record']['subject']==person and r['record']['night']==night
                          for i,y in enumerate(r['labels']) if y==stage]
                    samples.extend(pool[:1])
        sample_waves={k:np.empty((len(samples),3000),np.float32) for k in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal')}
        write_new(PRIVATE/'ST_selection.json',{'primary':samples,'missing':220-len(samples),'policy':'first1perstagepersonnight,maximum220,missingstratakept; finalprospectiveSTrefinement,notpredictions'})
    outdir=PRIVATE/role;outdir.mkdir(exist_ok=True)
    with measured(role+'_source_campaign') as counts:
        for ordinal,record in enumerate(source.records(role)):
            check_budget(started);g=geom.get(record['recording'])
            if g is None:
                execution.append({'status':'QUARANTINED','role':role});continue
            record_started=time.perf_counter();duration=g['duration'];n=len(g['labels']);channels={};bounds={};fs={}
            with source.open(record,values=True) as (reader,header):
                for channel in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental'):
                    values,identity=source.read(record,reader,channel,0,duration);channels[channel]=values.astype('float32');fs[channel]=identity['native_fs']
                    h=next(h for h in g['headers'] if h['label']==channel);bounds[channel]=np.array([h['physical_min'],h['physical_max']])
                marker='Event marker' if role=='SC_TRAIN' else 'Marker'
                values,identity=source.read(record,reader,marker,0,duration);channels['marker']=values.astype('float32');fs['marker']=identity['native_fs']
            loading_seconds=time.perf_counter()-record_started
            t=time.perf_counter();features=epoch_features(channels,fs['EMG submental'],fs['marker'],n);feature_seconds=time.perf_counter()-t
            wave=channels['EEG Fpz-Cz'][:n*3000].reshape(n,1,3000)
            t=time.perf_counter();z=native(model,wave,norm);inference_seconds=time.perf_counter()-t
            y=np.array(g['labels']);transition=np.r_[False,(y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0)]
            np.savez_compressed(outdir/(record['recording']+'_epochs.npz'),logits=z,truth=y,features=features,transition=transition)
            if role=='SC_TRAIN':
                for i,c in enumerate(training):
                    if c['recording_id']!=record['recording']:continue
                    for j,off in enumerate((-1,0,1)):
                        e=c['epoch']+off
                        if 0<=e<n:training_x[i,j,0]=(wave[e,0]-selection['normalization']['mean_uV'])/selection['normalization']['std_uV'];training_valid[i,j]=True
            else:
                from .context_model import st_predict
                st_predict(record,wave,selection['normalization'],outdir,n)
            for i,c in enumerate(samples):
                if c['recording_id']==record['recording']:
                    for channel in sample_waves:sample_waves[channel][i]=channels[channel][c['epoch']*3000:(c['epoch']+1)*3000]
            events={};statuses={};failures=[];detector_seconds=0.;coherences=[];eligible_seconds=0.
            for lo,hi,corelo,corehi in segment_bounds(duration):
                check_budget(started)
                arrays={channel:values[int(lo*fs[channel]):int(hi*fs[channel])] for channel,values in channels.items() if channel!='marker'}
                arrays.update({'EMG submental__fs':np.array(fs['EMG submental']),**{k+'__bounds':v for k,v in bounds.items() if k.startswith('EEG')}})
                t=time.perf_counter();result=worker.run(arrays,config,lo,corelo,corehi,duration);detector_seconds+=time.perf_counter()-t
                for key,values in result['events'].items():events.setdefault(key,[]).extend(values)
                for key,value in result['statuses'].items():statuses.setdefault(key,{}).setdefault(value,0);statuses[key][value]+=1
                failures.extend(result['failures']);eligible_seconds+=result['eligible_seconds'];coherences.append([result['coherence_delta'],result['coherence_sigma']])
            write_new(outdir/(record['recording']+'_events.json'),{'record':{k:record[k] for k in ('subject','recording','night','role')},
                'source_sha256':source.manifest['allowed'][record['psg']]['sha256'],'duration':duration,'n_epochs':n,'events':events,
                'statuses':statuses,'failures':failures,'eligible_seconds':eligible_seconds,'coherence':[float(np.mean([v[j] for v in coherences if v[j] is not None])) if any(v[j] is not None for v in coherences) else None for j in range(2)],
                'feature_names':FEATURES,'EMG_native_fs':fs['EMG submental'],'loading_seconds':loading_seconds,'feature_seconds':feature_seconds,
                'native_inference_seconds':inference_seconds,'detector_seconds':detector_seconds,'stage_blind_detection':True})
            execution.append({'status':'COMPLETE','physical_epochs':n,'scored_epochs':int((y>=0).sum()),'detector_failures':len(failures),
                'loading_seconds':loading_seconds,'feature_seconds':feature_seconds,'native_inference_seconds':inference_seconds,'detector_seconds':detector_seconds})
            print(json.dumps({'role':role,'recordings_complete':ordinal+1,'elapsed':time.perf_counter()-started,'physical_epochs':sum(r.get('physical_epochs',0) for r in execution)}),flush=True)
        worker.close()
        if role=='SC_TRAIN':
            np.savez_compressed(PRIVATE/'context_data.npz',x=training_x,valid=training_valid,truth=np.array([c['stage'] for c in training]),subjects=np.array([c['subject'] for c in training]),n_fit=fitn)
        np.savez_compressed(PRIVATE/(role+'_audit_inputs.npz'),**sample_waves)
        write_new(PRIVATE/(role+'_audit_coordinates.json'),{'samples':samples})
        if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('historical model mutated')
        receipt={'status':'COMPLETE','role':role,'records':len(execution),'complete_records':sum(r['status']=='COMPLETE' for r in execution),'physical_epochs':sum(r.get('physical_epochs',0) for r in execution),
            'scored_epochs':sum(r.get('scored_epochs',0) for r in execution),'detector_failures':sum(r.get('detector_failures',0) for r in execution),'sample_cases':len(samples),
            'source_access':source.access.counts,'native_full_epoch_forwards':sum(r.get('physical_epochs',0) for r in execution),'GPU_seconds':0,
            'historical_state_unchanged':True,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'stage_cost_seconds':{key:sum(r.get(key,0) for r in execution) for key in ('loading_seconds','feature_seconds','native_inference_seconds','detector_seconds')},
            'audit_input_sha256':sha256((PRIVATE/(role+'_audit_inputs.npz')).read_bytes()),'coordinates_sha256':sha256(canonical({'samples':samples}))}
        if role=='SC_TRAIN':receipt['context_data_sha256']=sha256((PRIVATE/'context_data.npz').read_bytes())
        write_new(PUBLIC/(role+'_execution.json'),receipt);counts.update(**receipt);source.access.close()

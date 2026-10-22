"""Twelve fixed trajectories; sealed milestones, no outcome-driven selection."""
import hashlib
import json
import time
import resource
import numpy as np
import torch
from mist_cove.phase11b.historical import ROOT,canonical,named_state_digest
from mist_cove.phase11b.security import sha256
from .provenance import PUBLIC,PRIVATE,isolated_rng,write_new
from .sampling import frozen_selection
from .data import load_cache
from .model_adequacy import WidthLedger
from .optimization import sampler,one_update,training_payload
from .seals import save_sealed
from .evaluation import evaluate
from .resources import measured,check_remaining


def run_identity(config,size,route,seed):
    return {'preregistration_sha256':sha256(canonical(config)),'capacity':size,'routing':route,'seed':seed}


def freeze():
    pilot=json.loads((PUBLIC/'resource_pilot.json').read_text())
    controls=json.loads((PUBLIC/'engineering_overfit.json').read_text())
    if pilot['status']!='PASS' or not all(r['finite_gradients'] and r['exact_resume'] for r in controls['runs']):
        raise ValueError('engineering prerequisites incomplete')
    files=['model_adequacy.py','optimization.py','training.py','evaluation.py','seals.py','support_operator.py','data.py','provenance.py','gates.py','resources.py']
    config={'version':'phase13-adequacy-v1','seeds':[123,456,789],'capacities':['S','L'],'routing':['R0','R1'],
        'horizon':3000,'milestones':[300,1000,3000],'batch_size':16,'optimizer':'Adam','learning_rate':.001,'schedule':'constant',
        'loss':'mean of clean and one local-HF paired classification CE; no regularizer',
        'source_sha256':{'src/mist_cove/phase13/'+f:sha256((Path(__file__).parent/f).read_bytes()) for f in files},
        'protocol_sha256':sha256((ROOT/'docs/mist_cove/PHASE_XIII_MODEL_ADEQUACY_PROTOCOL_V1.md').read_bytes()),
        'source_data_sha256':json.loads((PUBLIC/'source_scope_receipt.json').read_text())['data_sha256'],
        'primary_absolute_adequacy':{'family':'L_R0','participant_F1_min':.60,'N1_recall_min':.20,'every_class_F1_min':.10,'all_seeds':True},
        'primary_width':{'pair':'L_R0 minus S_R0 at3000','participant_F1_delta_min':.03,'other_class_F1_penalty_max':.10,'all_seeds':True},
        'primary_exposure':{'pair':'S_R0 step3000 minus300','participant_F1_delta_min':.03,'N1_recall_delta_min':.10,'all_seeds':True},
        'primary_routing':{'pair':'L_R1 minus L_R0 at3000','strong_local_HF_participant_F1_delta_min':.02,'clean_penalty_max':.02,'class_F1_penalty_max':.10,'all_seeds':True},
        'confirmatory_trajectories_max':12,'confirmatory_updates_max':36000,'scientific_CPU_seconds_max':14400,'main_CPU_seconds_max':10000,
        'selection':'fixed final3000; no early stopping, sweeps, M2, best-checkpoint selection or all15 final fit'}
    write_new(PUBLIC/'adequacy_preregistration.json',config)
    print('Twelve trajectories frozen at3000updates; protocol and implementation hashes recorded')


def run():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config=json.loads((PUBLIC/'adequacy_preregistration.json').read_text())
    for name,digest in config['source_sha256'].items():
        if sha256((ROOT/name).read_bytes())!=digest:raise ValueError('frozen scientific implementation changed')
    receipt=json.loads((PUBLIC/'source_scope_receipt.json').read_text())
    if receipt['data_sha256']!=config['source_data_sha256']:raise ValueError('frozen data identity differs')
    if (PRIVATE/'campaign_started.json').exists():raise ValueError('campaign already attempted; no automatic rerun')
    data=load_cache();n=int(data['n_fit']);x=torch.from_numpy(data['x']);y=torch.from_numpy(data['truth']).long();subjects=data['subjects']
    selection=frozen_selection();norm=selection['normalization'];monitor=[]
    for subject in sorted(set(subjects[:n])):
        for stage in range(5):monitor.extend(np.flatnonzero((subjects[:n]==subject)&(data['truth'][:n]==stage))[:10].tolist())
    monitor=np.array(monitor);results=[];streams={};updates=0;forward_examples=0
    write_new(PRIVATE/'campaign_started.json',{'preregistration_sha256':sha256(canonical(config)),'trajectories':12,'max_updates':36000})
    started=time.perf_counter()
    with measured('confirmatory_campaign','B',trajectories=12,optimizer_updates=36000) as counts:
        for seed in config['seeds']:
            for size in config['capacities']:
                for route in config['routing']:
                    with isolated_rng(seed):
                        identity=run_identity(config,size,route,seed);name=size+'_'+route+'_'+str(seed)
                        model=WidthLedger(size,route=='R1');optimizer=torch.optim.Adam(model.parameters(),lr=.001);g=sampler(seed)
                        initial={key:value.detach().clone() for key,value in model.state_dict().items()};history=[];trajectory_start=time.perf_counter()
                        for step in range(1,3001):
                            if step%25==1:
                                check_remaining(started)
                                if time.perf_counter()-started>10000:raise TimeoutError('main prospective CPU ceiling')
                            model.train();history.append(one_update(model,optimizer,x[:n],y[:n],g));updates+=1;forward_examples+=32
                            if step not in config['milestones']:continue
                            sealed=save_sealed(PRIVATE/'model_states'/name/('step_'+str(step).zfill(6)+'.pt'),
                                training_payload(model,optimizer,g,step,history,identity),{**identity,'step':step})
                            evaluation_start=time.perf_counter()
                            development,private=evaluate(model,x[n:],data['truth'][n:],subjects[n:],norm,views=True)
                            training_monitor,_=evaluate(model,x[monitor],data['truth'][monitor],subjects[monitor],norm,views=False)
                            forward_examples+=7*(len(x)-n)+len(monitor)
                            drift=float(torch.sqrt(sum((model.state_dict()[key]-value).square().sum() for key,value in initial.items())))
                            model_result={'capacity':size,'routing':route,'seed':seed,'step':step,'counts':model.counts(),
                                'training_loss_last100':float(np.mean([r['loss'] for r in history[-100:]])),
                                'gradient_norm_last100':float(np.mean([r['gradient_norm'] for r in history[-100:]])),
                                'parameter_drift_L2':drift,'fit_monitor':training_monitor['clean'],'development':development,
                                'checkpoint_byte_sha256':sealed['byte_sha256'],'state_digest':sealed['state_digest'],
                                'stream_sha256':sha256(''.join(r['draw_sha256'] for r in history).encode()),
                                'clean_exposures':step*16,'paired_exposures':step*16,'evaluation_seconds':time.perf_counter()-evaluation_start,
                                'trajectory_wall_to_milestone':time.perf_counter()-trajectory_start}
                            write_new(PRIVATE/'model_states'/name/('evaluation_'+str(step)+'.json'),{'by_view':{k:v['subject_metrics'] for k,v in private.items()}})
                            np.savez_compressed(PRIVATE/'model_states'/name/('predictions_'+str(step)+'.npz'),
                                                **{view+'_'+key:values[key] for view,values in private.items() for key in ('logits','contributions','reliability')})
                            write_new(PUBLIC/(name+'_'+str(step)+'.json'),model_result);results.append(model_result)
                            stream_key=seed,step
                            if stream_key in streams and streams[stream_key]!=model_result['stream_sha256']:raise ValueError('paired stream identity differs')
                            streams[stream_key]=model_result['stream_sha256']
                            print(json.dumps({'trajectory':name,'step':step,'participant_F1':development['clean']['metrics']['mean_subject_macro_f1'],
                                              'N1_recall':development['clean']['metrics']['per_stage_recall'][1]}),flush=True)
        counts.update(native_full_example_forwards=forward_examples,realized_updates=updates,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if updates!=36000 or len(results)!=36:raise ValueError('incomplete confirmatory denominator')
        write_new(PUBLIC/'campaign_execution.json',{'status':'COMPLETE','trajectories':12,'milestones':36,'optimizer_updates':updates,
            'clean_exposures':updates*16,'paired_exposures':updates*16,'native_full_example_forwards':forward_examples,
            'matched_streams':'PASS','CPU_wall_seconds':time.perf_counter()-started,'GPU_seconds':0,
            'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'fixed_final_checkpoint':3000})


from pathlib import Path

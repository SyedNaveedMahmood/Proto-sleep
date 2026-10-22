"""Generated throughput and at most two prespecified real fit-only overfit probes."""
import json
import time
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mist_cove.phase11b.historical import named_state_digest
from .provenance import PUBLIC,PRIVATE,OLD_PUBLIC,OLD_PRIVATE,isolated_rng,write_new
from .model_adequacy import WidthLedger
from .optimization import sampler,one_update,training_payload,restore_payload
from .seals import save_sealed,load_sealed
from .data import load_cache
from .resources import measured


def generated_pilot():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    results=[]
    with measured('generated_resource_pilot','ENGINEERING',generated_updates=100):
        for size in ('S','L'):
            for route in (False,True):
                with isolated_rng(13016):
                    model=WidthLedger(size,route);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                    x=torch.randn(64,1,3000);y=torch.arange(64)%5;g=sampler(13016)
                    started=time.perf_counter();history=[]
                    for _ in range(25):history.append(one_update(model,optimizer,x,y,g))
                    update_seconds=(time.perf_counter()-started)/25
                    model.eval();started=time.perf_counter()
                    with torch.no_grad():
                        for _ in range(10):model(x)
                    inference_seconds=(time.perf_counter()-started)/640
                    with tempfile.TemporaryDirectory(dir=PRIVATE) as directory:
                        path=Path(directory)/'generated.pt';identity={'domain':'generated','size':size,'routing':route}
                        started=time.perf_counter();seal=save_sealed(path,training_payload(model,optimizer,g,25,history,identity),identity)
                        payload,_=load_sealed(path,identity);checkpoint_seconds=time.perf_counter()-started
                    results.append({'capacity':size,'routing':route,'counts':model.counts(),'seconds_per_update':update_seconds,
                                    'inference_seconds_per_example':inference_seconds,'checkpoint_roundtrip_seconds':checkpoint_seconds,
                                    'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                                    'optimizer_tensor_bytes':sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)),
                                    'checkpoint_bytes':seal['bytes']})
        estimate=3*sum(r['seconds_per_update']*3000+r['inference_seconds_per_example']*(3*8000+3*500) for r in results)
        write_new(PUBLIC/'resource_pilot.json',{'status':'PASS' if estimate<10000 else 'SCOPE_AMENDMENT_REQUIRED','generated_only':True,
            'configurations':results,'projected_main_CPU_seconds':estimate,'full_horizon':3000,'uniform_reduction':False,
            'GPU_used':False,'CPU_threads':2,'proposed_main_CPU_ceiling':10000})
        print(json.dumps({'generated_pilot_seconds_projected':estimate,'full_horizon_3000_feasible':estimate<10000}))


def overfit():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=load_cache();n=int(data['n_fit']);y=data['truth'][:n]
    indices=np.concatenate([np.flatnonzero(y==stage)[:2] for stage in range(5)])
    if len(indices)!=10:raise ValueError('tiny fit panel missing a required stage; no replacement')
    write_new(PRIVATE/'engineering_tiny_selection.json',{'indices':indices.tolist(),'policy':'first2 per stage in frozen TRAIN-fit list','updates_per_run':1000})
    x=torch.from_numpy(data['x'][indices]);labels=torch.from_numpy(y[indices]).long();results=[]
    with measured('real_fit_engineering_overfit','B_ENGINEERING',real_overfit_runs=2,optimizer_updates=2000,native_examples=20000):
        for size in ('S','L'):
            with isolated_rng(13015):
                model=WidthLedger(size,False);optimizer=torch.optim.Adam(model.parameters(),lr=.001);g=sampler(13015)
                before=named_state_digest(model.state_dict());initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                with torch.no_grad():initial_loss=float(F.cross_entropy(model(x)['logits'],labels))
                history=[];resume_pass=False;start=time.perf_counter()
                for step in range(1,1001):
                    history.append(one_update(model,optimizer,x,labels,g,batch_size=10,paired=False))
                    if step==500:
                        identity={'domain':'engineering_real_fit_only','capacity':size,'seed':13015}
                        path=PRIVATE/('engineering_'+size+'_0500.pt')
                        save_sealed(path,training_payload(model,optimizer,g,step,history,identity),identity)
                        payload,_=load_sealed(path,identity)
                        # Compare uninterrupted next step with a restored clone,
                        # then continue the same run from that identical point.
                        cpu_rng=torch.get_rng_state().clone()
                        expected=one_update(model,optimizer,x,labels,g,batch_size=10,paired=False)
                        expected_state={k:v.clone() for k,v in model.state_dict().items()}
                        model2=WidthLedger(size,False);optimizer2=torch.optim.Adam(model2.parameters(),lr=.001);g2=sampler(13015)
                        restore_payload(payload,model2,optimizer2,g2,identity)
                        actual=one_update(model2,optimizer2,x,labels,g2,batch_size=10,paired=False)
                        if expected!=actual or any(not torch.equal(expected_state[k],model2.state_dict()[k]) for k in expected_state):raise ValueError('real tiny-run exact resume failed')
                        # Verification computes one extra step on each branch;
                        # restore step500 to avoid counting it as a trajectory step.
                        restore_payload(payload,model,optimizer,g,identity);resume_pass=True
                model.eval()
                with torch.no_grad():out=model(x);z=out['logits'];final_loss=float(F.cross_entropy(z,labels));accuracy=float((z.argmax(1)==labels).float().mean())
                changed=[name for name,value in model.state_dict().items() if not torch.equal(value,initial[name])]
                result={'capacity':size,'routing':'R0','updates':1000,'resume_verification_extra_updates':2,
                    'initial_CE':initial_loss,'final_CE':final_loss,'accuracy':accuracy,'predicted_classes':len(z.argmax(1).unique()),
                    'tiny_overfit_gate':final_loss<=.1 and accuracy>=.9,'finite_gradients':True,'intended_weights_updated':bool(changed),
                    'exact_additive_accounting':bool(torch.equal(z,model.bias+out['global']+out['contributions'].sum(1))),
                    'exact_resume':resume_pass,'state_changed':before!=named_state_digest(model.state_dict()),'wall_seconds':time.perf_counter()-start}
                results.append(result)
        write_new(PUBLIC/'engineering_overfit.json',{'domain':'TEN_TRAIN_FIT_EXAMPLES_ONLY','runs':results,'scientific_generalization_claim':False,
            'real_overfit_runs':2,'trajectory_updates':2000,'extra_resume_verification_updates':4,'total_engineering_real_updates':2004})
        print(json.dumps({'tiny_overfit':[{'capacity':r['capacity'],'CE':r['final_CE'],'accuracy':r['accuracy'],'exact_resume':r['exact_resume']} for r in results]}))


def previous_diagnostics():
    from mist_cove.phase11b.security import authenticated_snapshot,restricted_load
    anchors=json.loads((OLD_PUBLIC/'new_model_operator_audit.json').read_text())['rows'];results=[]
    with measured('previous_model_diagnostics','B'):
        for row in anchors:
            name=row['arm']+'_'+str(row['seed']);path=OLD_PRIVATE/(name+'.pt')
            with authenticated_snapshot(path,row['current_phase_checkpoint_sha256'],allowlist=[path]) as snapshot:payload=restricted_load(snapshot)
            old=json.loads((OLD_PUBLIC/(name+'_metrics.json')).read_text());state=payload['state_dict']
            results.append({'arm':row['arm'],'seed':row['seed'],'authenticated_bytes':True,'stored_slots':sum(v.numel() for v in state.values()),
                'final_training_CE':old['train_classification_loss_last20_mean'],'clean_N1_recall':old['evaluation']['clean']['metrics']['per_stage_recall'][1],
                'global_logit_magnitude':old['evaluation']['clean']['mean_absolute_global_logit'],
                'local_logit_magnitude':old['evaluation']['clean']['mean_absolute_local_logit_contribution'],
                'mean_reliability':old['evaluation']['clean']['mean_reliability'],'named_state_digest':named_state_digest(state)})
        write_new(PUBLIC/'previous_model_diagnostics.json',{'models':results,'historical_states_modified':False,
            'measured':['short300-update exposure','weak N1','remaining CE above1','fixed17-sample convolutional receptive field','early averaging of46 local positions','reported local/global magnitudes'],
            'hypotheses':['limited width','underoptimization','missing long waveform/temporal context','gate initialization/weak local gradient effects'],
            'not_recoverable':['Phase XII per-step gradients and optimizer states were not preserved; no retrospective gradient-flow proof']})

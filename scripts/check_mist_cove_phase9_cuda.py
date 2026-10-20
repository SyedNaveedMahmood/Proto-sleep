"""Bounded idle-device MIST-WIC CUDA correctness; never interrupts another job."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def availability():
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader'],text=True).strip()
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError) as e:return {'status':'BLOCKED','reason':str(e),'model_operations':0}
    return {'status':'BLOCKED' if jobs else 'IDLE','gpu_query':gpu,'compute_processes':jobs,'model_operations':0}


def run(path):
    from mist_cove.phase9.experiments import output
    from mist_evidence.runtime import atomic_json
    out=output(path,scope='SYNTHETIC_ONLY');receipt=availability()
    if receipt['status']=='BLOCKED':atomic_json(receipt,out/'cuda.json');return receipt
    import copy
    import torch
    from mist_cove.phase2.training import configure_precision
    from mist_cove.phase2.synthetic_runtime import assert_exact
    from mist_cove.phase4.model import make_model
    from mist_cove.phase6.generator import generate
    from mist_cove.phase7.prospective import optimizer,seal,restore
    from mist_cove.phase9.operators import internal_logits,sample_query_masks
    from mist_cove.phase9.objectives import paired_objective
    from mist_evidence.runtime import rng_state
    if not torch.cuda.is_available():
        receipt.update(status='BLOCKED',reason='CUDA-enabled interpreter/device unavailable');atomic_json(receipt,out/'cuda.json');return receipt
    configure_precision('cuda');torch.set_num_threads(2);started=time.perf_counter();rows=[]
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(.1)
    c={'learning_rate':.001,'weight_decay':.0001,'gradient_clip':5}
    x=generate(20,9317).x.cuda();y=generate(20,9317).y.cuda()
    for method in ('M1','M2'):
        if time.perf_counter()-started>120:raise TimeoutError('CUDA checker ceiling exceeded')
        torch.manual_seed(123);m=make_model(method).cuda();opt=optimizer(m,c);g=torch.Generator().manual_seed(711);torch.cuda.reset_peak_memory_stats()
        def step(model,optimizer,batch_rng):
            ix=torch.randperm(20,generator=batch_rng)[:16];optimizer.zero_grad(set_to_none=True)
            f,r=model.encode_candidates(torch.cat((x[ix],x[ix]+.1)));masks=sample_query_masks(16,device='cuda')
            loss,_=paired_objective(model,f[:16],r[:16],f[16:],r[16:],y[ix],masks,'C2',150,300)
            loss.backward();optimizer.step();assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
        step(m,opt,g);identity={'scope':'PHASE9_CUDA_ENGINEERING','method':method};creation=seal(out/f'{method}.pt',m,opt,g,1,identity)
        step(m,opt,g);expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),g.get_state(),rng_state())
        resumed=make_model(method).cuda();op=optimizer(resumed,c);batch=torch.Generator();restore(out/f'{method}.pt',creation,identity,resumed,op,batch);step(resumed,op,batch)
        assert_exact(expected,(resumed.state_dict(),op.state_dict(),batch.get_state(),rng_state()))
        with torch.no_grad():
            f,r=resumed.encode_candidates(x[:16]);native=resumed(x[:16]);full=internal_logits(resumed,f,r,torch.ones(16,15,dtype=torch.bool,device='cuda'))
            assert torch.equal(native,full);repeat=resumed(x[:16]);assert torch.equal(native,repeat)
            ev=resumed(x[:16],return_evidence=True);residual=float((ev.logits-ev.background_logits-ev.contributions.sum(1)).abs().max());assert residual<1e-5
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();assert reserved<=total*.1
        rows.append({'method':method,'batch_size':16,'finite_forward_backward':True,'exact_model_optimizer_global_batch_CUDA_rng_resume':True,'full_input_identity':True,'deterministic_FP32':True,'additive_residual':residual,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved})
    receipt.update(status='PASS',rows=rows,wall_seconds=time.perf_counter()-started,model_operations=18,device_memory_bytes=total,memory_fraction_limit=.1)
    atomic_json(receipt,out/'cuda.json');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output),indent=2))

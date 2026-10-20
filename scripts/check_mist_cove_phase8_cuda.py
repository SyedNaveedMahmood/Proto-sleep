"""Bounded Phase VIII CUDA check; refuses an occupied device before allocation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time


def availability():
    try:
        device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError) as e:return {'status':'BLOCKED','reason':str(e),'model_operations':0}
    return {'status':'BLOCKED' if processes else 'IDLE','device_query':device,'compute_processes':processes,'model_operations':0}


def run(output):
    from mist_cove.phase8.provenance import output_directory
    from mist_evidence.runtime import atomic_json
    out=output_directory(output);receipt=availability()
    if receipt['status']=='BLOCKED':atomic_json(receipt,out/'cuda_receipt.json');return receipt
    import torch
    from mist_cove.phase2.training import configure_precision
    from mist_cove.phase4.model import make_model
    from mist_cove.phase8.pooling import TrimmedModel
    from mist_cove.phase6.generator import generate
    from mist_cove.phase7.prospective import seal,restore,optimizer
    from mist_cove.phase2.synthetic_runtime import assert_exact
    from mist_evidence.runtime import rng_state,restore_rng
    configure_precision('cuda');torch.set_num_threads(2);started=time.perf_counter();rows=[]
    c={'learning_rate':.001,'weight_decay':.0001,'gradient_clip':5}
    for name,factory in [('meanmax',lambda:make_model('M1')),('trimmed',TrimmedModel),('additive',lambda:make_model('M2'))]:
        torch.manual_seed(123);m=factory().cuda();o=optimizer(m,c);g=torch.Generator().manual_seed(123)
        x=generate(20,97123).x[:16].cuda();y=generate(20,97123).y[:16].cuda()
        torch.cuda.reset_peak_memory_stats()
        def step(m,o):
            o.zero_grad(set_to_none=True);logits=m(x);loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();o.step()
            assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in m.parameters())
        step(m,o);identity={'scope':'GENERATED_PHASE8_CUDA','method':name};proof=seal(out/f'{name}.pt',m,o,g,1,identity)
        step(m,o);expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),rng_state())
        n=factory().cuda();op=optimizer(n,c);restore(out/f'{name}.pt',proof,identity,n,op,torch.Generator());step(n,op)
        assert_exact(expected,(n.state_dict(),op.state_dict(),rng_state()))
        with torch.no_grad():
            a=n(x,return_evidence=True);b=n(x);torch.testing.assert_close(a.logits,b,rtol=0,atol=0)
            residual=float((a.logits-a.background_logits-a.contributions.sum(1)).abs().max());assert residual<1e-5
        rows.append({'method':name,'exact_resume_all_rng':True,'additive_residual':residual,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    receipt.update(status='PASS',rows=rows,wall_seconds=time.perf_counter()-started,model_operations=18)
    atomic_json(receipt,out/'cuda_receipt.json');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output),indent=2))

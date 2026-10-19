"""Bounded synthetic CUDA measurements; competing GPU work is left untouched."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,time
from pathlib import Path
import torch
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.provenance import implementation_snapshot,environment
from mist_cove.phase4.model import make_model
from mist_cove.phase4.correctness import exact_resume

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous GPU receipt')
    a.output.mkdir(parents=True)
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:
        receipt={'status':'BLOCKED_GPU_IN_USE','hardware_query':identity,'other_compute_process_count':len(active.splitlines()),'gpu_allocation_attempted':False,'real_data_accessed':False}
    else:
        configure_precision('cuda');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
        rows=[]
        for method in ['M0','M1','M2','M3','M2_rms','M1_phase']:
            torch.manual_seed(123);torch.cuda.reset_peak_memory_stats();model=make_model(method).cuda();x=torch.randn(16,1,3000,device='cuda')
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(5):
                model.zero_grad(set_to_none=True);logits=model(x);loss=logits.square().mean();loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite GPU gradient')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            ev=model(x,return_evidence=True)
            if (ev.logits-ev.background_logits-ev.contributions.sum(1)).abs().max()>1e-5:raise ValueError('GPU additive failure')
            row={'method':method,'batch_size':16,'five_forward_backward_seconds':seconds,'examples_per_second':80/seconds,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'parameters':sum(p.numel() for p in model.parameters())}
            del model,x,ev;torch.cuda.empty_cache()
            row['exact_resume']=exact_resume(a.output/method,method,'cuda');rows.append(row)
        receipt={'status':'PASS','hardware_query':identity,'environment':environment('cuda'),'implementation':implementation_snapshot(),'rows':rows,'real_data_accessed':False}
    with (a.output/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

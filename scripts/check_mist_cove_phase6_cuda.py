"""Bounded Phase VI generated checks; never interrupt a competing GPU job."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import signal
import subprocess
import time
import torch
from mist_cove.phase2.provenance import environment
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget
from mist_evidence.runtime import atomic_json
from mist_cove.phase6.generator import generate
from mist_cove.phase6.runtime import private_output,frozen,historical
from mist_cove.phase6.correctness import exact_resume


def competitors():
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return [p for p in s.splitlines() if p.strip()!=str(os.getpid())]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=private_output(a.output,'SYNTHETIC_ONLY');proof=frozen();historical()
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    if competitors():atomic_json({'status':'BLOCKED_GPU_IN_USE','gpu_allocation_attempted':False,'real_data_accessed':False},output/'receipt.json');return
    def expire(s,f):raise TimeoutError('120-second synthetic CUDA ceiling')
    signal.signal(signal.SIGALRM,expire);signal.alarm(120);budget=Budget(120)
    try:
        configure_precision('cuda');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
        rows=[]
        for method in ['M1','M2','M3','M2_rms']:
            budget.check()
            if competitors():raise RuntimeError('competing GPU job appeared; stop')
            torch.manual_seed(8919);model=make_model(method).cuda();x=generate(20,8920,'multiple').x[:16].cuda()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(5):
                model.zero_grad(set_to_none=True);value=model(x)
                if value.shape!=(16,5) or not torch.isfinite(value).all():raise ValueError('CUDA output invalid')
                value.square().mean().backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('CUDA gradient invalid')
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            e=model(x,return_evidence=True);residual=float((e.logits-e.background_logits-e.contributions.sum(1)).abs().max())
            if residual>1e-5:raise ValueError('CUDA additive failure')
            for lo,hi in e.event_intervals[e.valid]:
                if not (0<=lo<hi<=3000 and hi-lo==400 and lo%200==0):raise ValueError('CUDA interval failure')
            row={'method':method,'batch_size':16,'five_forward_backward_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'max_additive_residual':residual}
            del model,x,value,e;torch.cuda.empty_cache()
            row['exact_resume']=exact_resume(output/method,method,'cuda');rows.append(row)
        frozen();historical()
        atomic_json({'status':'PASS','hardware_query':device,'environment':environment('cuda'),'frozen':proof,'rows':rows,'wall_seconds':budget.elapsed,'real_data_accessed':False},output/'receipt.json')
        print('PASS: four generated CUDA smoke and exact-resume probes')
    except BaseException as error:atomic_json({'status':'STOPPED','error':type(error).__name__,'real_data_accessed':False},output/'receipt.json');raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

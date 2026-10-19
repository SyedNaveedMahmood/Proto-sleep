"""Bounded generated-input CUDA check; refuse rather than interrupt other jobs."""
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
from mist_evidence.runtime import atomic_json
from mist_cove.phase5.models import WaveformControl,RestrictedProbe
from mist_cove.phase5.features import patches,zero_phase
from mist_cove.phase5.correctness import exact_resume
from mist_cove.phase5.runtime import private_output,execution_proofs,frozen_proofs,Budget


def competitors():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [line for line in output.strip().splitlines() if line.strip()!=str(os.getpid())]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=private_output(a.output,'SYNTHETIC_ONLY');proof=execution_proofs();frozen_proofs()
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    def stop(signum,frame): raise TimeoutError('Phase V CUDA 120-second ceiling')
    signal.signal(signal.SIGALRM,stop);signal.alarm(120)
    budget=Budget(120)
    try:
        if competitors():
            atomic_json({'status':'BLOCKED_GPU_IN_USE','gpu_allocation_attempted':False,'real_data_accessed':False},output/'receipt.json');return
        configure_precision('cuda');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
        rows=[]
        for method in ['original','patch_zero','coherent_zero','restricted_probe','additive_M2']:
            budget.check()
            if competitors(): raise RuntimeError('another GPU process appeared; stop without interrupting it')
            torch.manual_seed(8913)
            model=(RestrictedProbe(torch.randn(16,201)) if method=='restricted_probe' else make_model('M2') if method=='additive_M2' else WaveformControl(method)).cuda()
            x=torch.randn((16,201) if method=='restricted_probe' else (16,1,3000),device='cuda')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(5):
                model.zero_grad(set_to_none=True);value=model(x)
                if value.shape!=(16,5) or not torch.isfinite(value).all():raise FloatingPointError('invalid CUDA logits')
                value.square().mean().backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite CUDA gradient')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            residual=None
            if method!='restricted_probe':
                e=model(x,return_evidence=True)
                residual=float((e.logits-e.background_logits-e.contributions.sum(1)).abs().max())
                if residual>1e-5:raise ValueError('CUDA additive reconciliation failed')
                if method=='additive_M2':
                    for lo,hi in e.event_intervals[e.valid]:
                        if not (0<=lo<hi<=3000 and hi-lo==400 and lo%200==0):raise ValueError('CUDA source interval invalid')
                del e
            rows.append({'method':method,'batch_size':16,'five_forward_backward_seconds':seconds,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'max_additive_residual':residual,'parameters':sum(p.numel() for p in model.parameters())})
            del model,x,value;torch.cuda.empty_cache()
        resume=[]
        for control in ['original','patch_zero','coherent_zero']:
            budget.check()
            if competitors():raise RuntimeError('another GPU process appeared')
            resume.append(exact_resume(output/control,control,'cuda'))
        execution_proofs();frozen_proofs()
        atomic_json({'status':'PASS','hardware_query':hardware,'environment':environment('cuda'),'execution':proof,
            'rows':rows,'exact_resume':resume,'wall_seconds':budget.elapsed,'real_data_accessed':False},output/'receipt.json')
        print('PASS: five CUDA smoke checks and three exact-resume probes',flush=True)
    except BaseException as error:
        atomic_json({'status':'STOPPED','error_type':type(error).__name__,'real_data_accessed':False},output/'receipt.json');raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

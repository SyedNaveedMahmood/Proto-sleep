"""Generated exact checkpoint-recovery probe for the three information controls."""
import copy
from pathlib import Path
import random
import numpy as np
import torch
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_evidence.runtime import seed_all,rng_state
from .models import WaveformControl
from .signals import harmonic
from .runtime import save_checkpoint,restore_checkpoint


def exact_resume(output,control,device='cpu'):
    configure_precision(device);seed_all(8911,True)
    model=WaveformControl(control).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=harmonic(5,8912).x.to(device);y=torch.arange(5,device=device)
    def step(m,o):
        random.random();np.random.random()
        ix=torch.randperm(5,device=device)
        observed=x+torch.randn_like(x)*.01
        o.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(m(observed[ix]),y[ix]);loss.backward();o.step()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite resume probe')
    step(model,optimizer)
    path=Path(output)/'generated_resume.pt';identity={'scope':'SYNTHETIC_ONLY','transform':control,'device':device}
    save_checkpoint(path,model,optimizer,identity)
    step(model,optimizer)
    expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),rng_state())
    other=WaveformControl(control).to(device);opt=torch.optim.AdamW(other.parameters(),lr=.001)
    restore_checkpoint(path,other,opt,identity);step(other,opt)
    assert_exact(expected,(other.state_dict(),opt.state_dict(),rng_state()))
    return {'control':control,'device':device,'model_optimizer_python_numpy_cpu_cuda_rng_exact':True,'logical_optimizer_updates':2}

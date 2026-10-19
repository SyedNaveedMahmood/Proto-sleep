"""Exact generated checkpoint recovery, separate from scientific fits."""
import copy
from pathlib import Path
import random
import numpy as np
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_cove.phase2.provenance import environment,file_sha
from mist_cove.phase5.runtime import save_checkpoint,restore_checkpoint
from mist_evidence.runtime import seed_all,rng_state
from .generator import generate


def exact_resume(output,method,device='cpu'):
    configure_precision(device);seed_all(8917,True)
    model=make_model(method).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    data=generate(5,8918,'multiple');x=data.x.to(device);y=data.y.to(device)
    def step(m,o):
        random.random();np.random.random();ix=torch.randperm(5,device=device)
        observed=x+torch.randn_like(x)*.01
        o.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(observed[ix]),y[ix]);loss.backward();o.step()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite correctness probe')
    step(model,optimizer)
    identity={'scope':'SYNTHETIC_ONLY','model':model.cfg.dictionary(),'learning_rate':.001,
        'config_sha256':file_sha(Path(__file__).resolve().parents[3]/'configs/mist_cove_phase6_synthetic_v1.json'),
        'correctness_code_sha256':file_sha(__file__),'environment':environment(device)}
    path=Path(output)/'generated.pt';save_checkpoint(path,model,optimizer,identity)
    step(model,optimizer);expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),rng_state())
    other=make_model(method).to(device);opt=torch.optim.AdamW(other.parameters(),lr=.001)
    restore_checkpoint(path,other,opt,identity);step(other,opt)
    assert_exact(expected,(other.state_dict(),opt.state_dict(),rng_state()))
    return {'method':method,'device':device,'model_optimizer_python_numpy_cpu_cuda_rng_exact':True,'logical_updates':2}

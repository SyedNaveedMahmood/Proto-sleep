"""Matched stateful sampling and exact-resume optimizer payloads."""
import hashlib
import random
import torch
import torch.nn.functional as F
from mist_cove.phase11b.development import local_hf
from mist_cove.phase11b.historical import named_state_digest


def sampler(seed):
    return torch.Generator().manual_seed(seed+10000)


def one_update(model,optimizer,x,y,generator,batch_size=16,paired=True):
    indices=torch.randint(len(x),(batch_size,),generator=generator)
    atoms=torch.randint(15,(batch_size,),generator=generator)
    phase=2*torch.pi*torch.rand(batch_size,generator=generator)
    draw=hashlib.sha256(b''.join(v.numpy().tobytes() for v in (indices,atoms,phase))).hexdigest()
    a=model(x[indices]);loss=F.cross_entropy(a['logits'],y[indices])
    if paired:
        b=model(local_hf(x[indices],atoms,phase))
        loss=.5*(loss+F.cross_entropy(b['logits'],y[indices]))
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite classification loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    for p in model.parameters():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise FloatingPointError('intended parameter has missing/nonfinite gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient norm')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite updated parameter')
    return {'loss':float(loss.detach()),'gradient_norm':float(norm),'draw_sha256':draw}


def training_payload(model,optimizer,generator,step,history,identity):
    return {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler_state':generator.get_state(),
            'torch_CPU_RNG':torch.get_rng_state(),'python_RNG':random.getstate(),'step':step,
            'history':history,'identity':identity,'state_digest':named_state_digest(model.state_dict()),
            'numpy_RNG_used_in_training':False}


def restore_payload(payload,model,optimizer,generator,identity):
    if payload['identity']!=identity or len(payload['history'])!=payload['step']:
        raise ValueError('checkpoint trajectory identity/history mismatch')
    model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    if named_state_digest(model.state_dict())!=payload['state_digest']:raise ValueError('loaded parameter identity differs')
    generator.set_state(payload['sampler_state']);torch.set_rng_state(payload['torch_CPU_RNG']);random.setstate(payload['python_RNG'])
    return payload['step'],payload['history']

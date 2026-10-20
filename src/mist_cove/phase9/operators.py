"""Physical atom closure and unchanged native-head evidence withholding."""
from functools import lru_cache
import numpy as np
import torch

ATOMS=15
CANDIDATES=14


def atom_to_candidate_mask(atoms):
    if atoms.shape[-1]!=ATOMS or atoms.dtype!=torch.bool:raise ValueError('Boolean fifteen-atom availability required')
    return atoms[...,:-1]&atoms[...,1:]


def internal_logits(model,features,rms,atoms,*,neutral=0.):
    if neutral not in (0.,.01):raise ValueError('unregistered neutral control')
    if features.shape[-2:]!=(14,18) or rms.shape[-1]!=14:raise ValueError('native candidate geometry required')
    visible=atom_to_candidate_mask(atoms)
    # Broadcasting supports one cached example queried by many masks, matched
    # per-example masks and an explicit [batch,queries,atoms] training axis.
    if visible.ndim==3:
        features=features[:,None];rms=rms[:,None]
    f=torch.where(visible[...,None],features,torch.as_tensor(neutral,device=features.device,dtype=features.dtype))
    r=torch.where(visible,rms,torch.zeros((),device=rms.device,dtype=rms.dtype))
    shape=f.shape[:-2]
    return model.decision(f.reshape(-1,14,18),r.reshape(-1,14))[0].reshape(*shape,5)


@lru_cache(maxsize=1)
def masks_numpy():return ((np.arange(1<<ATOMS)[:,None]>>np.arange(ATOMS))&1).astype(bool)


def masks_tensor(device='cpu'):return torch.from_numpy(masks_numpy()).to(device)


def margin(logits,target):
    other=logits.clone();other[...,target]=-float('inf')
    return logits[...,target]-other.amax(-1)


@torch.no_grad()
def response_table(model,features,rms,target,*,batch_size=512,neutral=0.,temperature=1.,budget=None):
    if features.shape!=(1,14,18) or rms.shape!=(1,14):raise ValueError('one cached example per exact table')
    if not 0<=target<5:raise ValueError('fixed original predicted class required')
    masks=masks_tensor(features.device);margins=[];prob=[]
    for start in range(0,len(masks),batch_size):
        if budget is not None:budget.check()
        logits=internal_logits(model,features,rms,masks[start:start+batch_size],neutral=neutral)
        margins.append(margin(logits,target).cpu());prob.append((logits/temperature).softmax(-1)[:,target].cpu())
    return torch.cat(margins).numpy(),torch.cat(prob).numpy()


def sample_query_masks(batch,generator=None,device='cpu'):
    probs=torch.tensor([.5,.75],device=device)[None,:,None]
    masks=torch.rand(batch,2,15,generator=generator,device=device)<probs
    invalid=lambda m:(~atom_to_candidate_mask(m).any(-1))|atom_to_candidate_mask(m).all(-1)
    bad=invalid(masks)
    while bad.any():
        proposed=torch.rand(batch,2,15,generator=generator,device=device)<probs
        masks=torch.where(bad[:,:,None],proposed,masks);bad=invalid(masks)
    return masks


def component_contract(draw,index,target='diagnostic'):
    """Stored component recomposition for both diagnostic/nuisance controls."""
    original=draw.x[index:index+1]
    if target=='diagnostic':removed=draw.events[index:index+1].sum(1)
    elif target=='nuisance':removed=draw.nuisance[index:index+1]
    elif target=='identity':removed=torch.zeros_like(original)
    else:raise ValueError('unknown component target')
    changed=draw.base[index:index+1] if target=='diagnostic' else original-removed
    return changed,{'target_removal_fraction':0. if target=='identity' else 1.,
        'nontarget_max_residual':float((changed+removed-original).abs().max()),
        'target_present':bool(removed.abs().max()>0),'operator':'exact_'+target}

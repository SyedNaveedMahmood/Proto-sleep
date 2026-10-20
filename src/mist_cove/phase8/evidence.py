"""Matched frozen-encoder E0/E1/E2/E3 pipeline, collectively gated."""
import copy
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase5.experiment import CachedHead
from mist_cove.phase5.runtime import fit
from mist_cove.phase7.prospective import encode
from mist_evidence.runtime import tensor_digest

METHODS={'E0':'M1','E1':'M2','E2':'M3','E3':'M2_rms'}


def matched_heads(encoder,x,y,c,seed,budget,*,gate,engineering=False):
    if not engineering and gate.get('status')!='PASS':
        raise ValueError('complete primary robustness prerequisite required before confirmatory head fitting')
    if engineering and gate.get('scope')!='GENERATED_ENGINEERING_FIXTURE':
        raise ValueError('explicit generated engineering fixture identity required')
    expected=tensor_digest(encoder);models={};receipts={}
    for name,method in METHODS.items():
        torch.manual_seed(seed+402);m=make_model(method);m.encoder.load_state_dict(encoder);m.encoder.requires_grad_(False)
        f,r=encode(m,x);packed=torch.cat((f,r[:,:,None]),-1)
        training=fit(CachedHead(m),packed,y,c,seed,budget,updates=c['head_updates'])
        if tensor_digest(m.encoder.state_dict())!=expected:raise ValueError('frozen encoder changed')
        models[name]=m;receipts[name]={'encoder_sha256':expected,'training':training,
            'fixture_only':engineering,'effective_head_parameters':95 if name=='E2' else 185}
    return models,receipts

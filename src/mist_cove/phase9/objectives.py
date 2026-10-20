"""Matched paired classification/full-view/query JS losses; no masked CE."""
import torch
from .operators import internal_logits


def js_divergence(a,b):
    p=a.softmax(-1);q=b.softmax(-1);mid=(p+q)*.5
    return .5*((p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)+
               (q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)).mean()


def paired_objective(model,features_a,rms_a,features_b,rms_b,y,masks,arm,step,updates):
    if arm not in {'C0','C1','C2'}:raise ValueError('unregistered arm')
    a=model.decision(features_a,rms_a)[0];b=model.decision(features_b,rms_b)[0]
    cls=.5*(torch.nn.functional.cross_entropy(a,y)+torch.nn.functional.cross_entropy(b,y))
    full=js_divergence(a,b);query=torch.zeros((),device=a.device)
    if arm=='C2':query=js_divergence(internal_logits(model,features_a,rms_a,masks),internal_logits(model,features_b,rms_b,masks))
    warmup=min(1.,step/max(1,updates*.2))
    loss=cls+(warmup*.5*full if arm!='C0' else 0)+(warmup*.25*query if arm=='C2' else 0)
    return loss,{'classification':float(cls.detach()),'full_js':float(full.detach()),'query_js':float(query.detach()),'warmup':warmup}

"""Blind compact order statistics; encoder and 185-parameter head unchanged."""
import torch
from mist_cove.phase4.model import MorphModel, MorphConfig


def pool(features,method='meanmax',mask=None):
    if features.ndim!=3 or features.shape[1:]!=(14,18):raise ValueError('14 by 18 candidate geometry required')
    if method=='oracle':
        if mask is None or not mask.any(1).all():raise ValueError('nonempty privileged oracle mask required')
        return torch.cat(((features*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],
                          features.masked_fill(~mask[:,:,None],-float('inf')).amax(1)),1)
    if method=='trimmed':
        ordered=features.sort(1).values[:,2:-2]
        return torch.cat((ordered.mean(1),ordered.amax(1)),1)
    if method=='bounded':features=features.clamp(-3,3)
    if method=='mean':return torch.cat((features.mean(1),torch.zeros_like(features.mean(1))),1)
    if method=='max':return torch.cat((torch.zeros_like(features.mean(1)),features.amax(1)),1)
    if method not in {'meanmax','bounded'}:raise ValueError('unknown pooling')
    return torch.cat((features.mean(1),features.amax(1)),1)


class TrimmedModel(MorphModel):
    def __init__(self):super().__init__(MorphConfig(method='M1'))
    def decision(self,f,r,**kwargs):
        result=list(super().decision(f,r,**kwargs));p=pool(f,'trimmed')
        parts=p[:,:,None]*torch.cat((self.background_head.weight,self.event_head.weight),1).T
        bg=parts.sum(1)+self.background_head.bias
        result[0]=bg;result[1]=bg;result[2]=p;result[3]=parts
        return tuple(result)

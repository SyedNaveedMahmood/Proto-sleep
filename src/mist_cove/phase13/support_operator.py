"""Independent availability proof for this exact nonoverlapping model family."""
from dataclasses import dataclass
import numpy as np
import torch
from mist_cove.phase11b.historical import named_state_digest,canonical
from mist_cove.phase11b.security import sha256
from mist_cove.phase9.operators import masks_tensor,margin
from .model_adequacy import WidthLedger


@dataclass(frozen=True)
class LedgerCache:
    contributions:torch.Tensor
    global_features:torch.Tensor
    native:torch.Tensor
    identity:str
    content:str


def content_digest(*values):
    return sha256(b''.join(v.detach().contiguous().cpu().numpy().tobytes() for v in values))


class LedgerOperator:
    def __init__(self,model):
        if type(model) is not WidthLedger or model.training:raise ValueError('exact frozen eval WidthLedger required')
        self.model=model;self.capacity=model.capacity;self.routing=model.routing
        self.identity=named_state_digest(model.state_dict())
        self.support_map={'atoms':[[j*200,(j+1)*200] for j in range(15)],
            'local_branch':'one encoder, local evidence and scalar quality gate per atom; no cross-atom normalization',
            'global_features':7,'global_support':list(range(15)),
            'withholding':'withheld local contributions0; partial global INPUT features neutral0/.01 before head',
            'native_convolution_RF':17,'P3B_proof_reused':False}
        self.support_identity=sha256(canonical(self.support_map));self.check()

    def check(self,cache=None):
        if self.model.training or self.model.capacity!=self.capacity or self.model.routing!=self.routing or named_state_digest(self.model.state_dict())!=self.identity:raise ValueError('frozen state changed')
        if not hasattr(self,'topology'):
            with torch.random.fork_rng(devices=[]):reference=WidthLedger(self.capacity,self.routing).eval()
            self.topology=[(k,type(v),repr(v)) for k,v in reference.named_modules()]
        if [(k,type(v),repr(v)) for k,v in self.model.named_modules()]!=self.topology:raise ValueError('unreviewed dependency topology')
        for module in self.model.modules():
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or 'forward' in module.__dict__:raise ValueError('unreviewed dynamic hooks/override')
        if cache is not None:
            if cache.identity!=self.identity or content_digest(cache.contributions,cache.global_features,cache.native)!=cache.content:raise ValueError('cache/state identity differs')

    @torch.no_grad()
    def cache(self,x):
        self.check()
        if x.shape!=(1,1,3000) or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('one finite original FP32 epoch required')
        output=self.model(x);c=output['contributions'][0].clone();g=output['global_features'][0].clone();native=output['logits'][0].clone()
        if not all(torch.isfinite(v).all() for v in (c,g,native)):raise ValueError('nonfinite cached representation')
        return LedgerCache(c,g,native,self.identity,content_digest(c,g,native))

    @torch.no_grad()
    def _logits(self,cache,masks,neutral=0.):
        if masks.ndim!=2 or masks.shape[1]!=15 or masks.dtype!=torch.bool:raise ValueError('physical Boolean masks required')
        if neutral not in (0.,.01):raise ValueError('frozen neutral value required')
        local=(masks[:,:,None]*cache.contributions[None]).sum(1)
        global_inputs=torch.full((len(masks),7),neutral,dtype=torch.float32)
        global_inputs[masks.all(1)]=cache.global_features
        return self.model.bias+self.model.global_head(global_inputs)+local

    def logits(self,cache,masks,neutral=0.):
        self.check(cache)
        return self._logits(cache,masks,neutral)

    @torch.no_grad()
    def response_table(self,cache,target,neutral=0.):
        self.check(cache);values=[];masks=masks_tensor('cpu')
        for start in range(0,32768,512):values.append(margin(self._logits(cache,masks[start:start+512],neutral),target).numpy())
        self.check(cache);values=np.concatenate(values)
        return {'target':target,'margins':values,'truth':values>1e-6,'availability_queries':32768,'full_target_queries':0}

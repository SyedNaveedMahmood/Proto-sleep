"""Independent coarse17-variable support proof for offline context family."""
from dataclasses import dataclass
import numpy as np
import torch
from .context_model import ContextLedger
from mist_cove.phase13.support_operator import LedgerOperator,content_digest
from mist_cove.phase11b.historical import named_state_digest
from mist_cove.phase9.operators import margin
from mist_cove.phase9.circuits import compile_robdd,evaluate_robdd
from mist_cove.phase9.clauses import extract_readable_clauses,verify_clause


class ContextOperator:
    def __init__(self,model):
        if type(model) is not ContextLedger or model.training:raise ValueError('frozen reviewed context class required')
        self.model=model;self.arm=model.arm;self.identity=named_state_digest(model.state_dict());self.base=LedgerOperator(model.base)
        expected=torch.tensor([0.,1.,0.] if self.arm=='C0' else [.25,.5,.25])
        if not torch.equal(model.context_weights,expected):raise ValueError('unreviewed context aggregation')
        self.support={'variables':17,'order':['previous_whole30s']+['center_'+str(j) for j in range(15)]+['next_whole30s'],
            'seconds':[30]+[2]*15+[30],'neighbor_availability':'all samples or no local/global data','global_inputs':'per-epochall-or-neutral',
            'missing':'fixed physically valid weights; no joins; no neighbor labels','P3B_proof_reused':False}
    def check(self,cache=None):
        if self.model.training or self.model.arm!=self.arm or named_state_digest(self.model.state_dict())!=self.identity:raise ValueError('context state mutated')
        if 'forward' in self.model.__dict__ or self.model._forward_hooks or self.model._forward_pre_hooks:raise ValueError('dynamic context topology')
        self.base.check()
        if cache is not None and cache[-1]!=content_digest(*cache[:-1]):raise ValueError('context cache mutated')
    @torch.no_grad()
    def cache(self,x,valid):
        self.check();out=self.model(x,valid)
        c=self.model.base(x.reshape(3,1,3000));parts=(c['contributions'].clone(),c['global_features'].clone(),out['weights'][0].clone(),out['logits'][0].clone())
        return (*parts,content_digest(*parts))
    @torch.no_grad()
    def _logits(self,cache,masks,neutral=0.):
        if neutral not in (0.,.01) or masks.dtype!=torch.bool or masks.shape[1]!=17:raise ValueError('registered availability state required')
        c,g,w,native,_=cache
        physical=torch.stack([masks[:,0,None].expand(-1,15),masks[:,1:16],masks[:,16,None].expand(-1,15)],1)
        local=(physical[:,:,:,None]*c[None]*w[None,:,None,None]).sum((1,2))
        global_input=torch.full((len(masks),3,7),neutral,dtype=torch.float32)
        global_input=torch.where(physical.all(-1)[:,:,None],g[None],global_input)
        glob=self.model.base.global_head(global_input.reshape(-1,7)).reshape(-1,3,5)
        return self.model.base.bias+(glob*w[None,:,None]).sum(1)+local
    def logits(self,cache,masks,neutral=0.):self.check(cache);return self._logits(cache,masks,neutral)
    def table(self,cache,target):
        self.check(cache);ids=torch.arange(1<<17);bits=torch.arange(17);values=[]
        for start in range(0,len(ids),512):values.append(margin(self._logits(cache,((ids[start:start+512,None]>>bits)&1).bool()),target).numpy())
        self.check(cache);values=np.concatenate(values);return {'truth':values>1e-6,'margins':values,'target':target}


def summarize(table):
    truth=table['truth'];graph=compile_robdd(truth);signed=extract_readable_clauses(truth,margins=table['margins']);positive=extract_readable_clauses(truth,positive_only=True,margins=table['margins'])
    for group in (signed,positive):
        for clause in group['clauses']:
            if not verify_clause(truth,clause['positive_bits'],clause['negative_bits']):raise ValueError('unsound context clause')
            clause['retained_seconds']=sum(30 if bit in (0,16) else 2 for bit in clause['positive'])
    return {'exact_all_rows':bool(np.array_equal(evaluate_robdd(graph),truth)),'masks':len(truth),'BDD_nodes':len(graph['nodes'])-2,
            'BDD_sha256':graph['sha256'],'signed':signed,'positive':positive,'prior':bool(truth[0]),'full_response':bool(truth[-1])}

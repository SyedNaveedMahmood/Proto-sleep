"""Fixed endpoint and nuisance evaluations; original labels are not rescored."""
import numpy as np
import torch
from mist_cove.phase11b.development import local_hf
from mist_cove.phase11b.evaluation import public_metrics
from mist_transfer.metrics import classification_metrics

VIEWS=('clean','mild_local_HF','strong_local_HF','global_HF','drift','gain','broadband')


def nuisance_views(x,normalization):
    n=len(x);atoms=torch.arange(n)%15;rms=x.std(-1,unbiased=False).clamp_min(1e-8)
    t=torch.arange(3000,dtype=x.dtype)/100
    mild=x.clone();small=mild.reshape(n,15,200);t_atom=torch.arange(200,dtype=x.dtype)/100
    small[torch.arange(n),atoms]+=.3*rms[:,0,None]*torch.sin(2*torch.pi*35*t_atom)[None]
    noise=torch.stack([torch.randn((1,3000),generator=torch.Generator().manual_seed(13021+i)) for i in range(n)])
    return {'clean':x,'mild_local_HF':mild,'strong_local_HF':local_hf(x,atoms),
            'global_HF':x+.5*rms[:,:,None]*torch.sin(2*torch.pi*35*t)[None,None],
            'drift':x+.1*rms[:,:,None]*torch.sin(2*torch.pi*.2*t)[None,None],
            'gain':x*1.05+.05*normalization['mean_uV']/normalization['std_uV'],
            'broadband':x+.3*rms[:,:,None]*noise}


def class_collapse(logits):
    counts=np.bincount(np.asarray(logits).argmax(1),minlength=5)
    return {'predicted_counts':counts.tolist(),'predicted_classes':int((counts>0).sum()),
            'largest_fraction':float(counts.max()/counts.sum()),'N1_not_predicted':bool(counts[1]==0),
            'near_single_class':bool(counts.max()/counts.sum()>=.98)}


@torch.no_grad()
def evaluate(model,x,truth,subjects,normalization,views=True):
    model.eval();inputs=nuisance_views(x,normalization) if views else {'clean':x}
    public={};private={};reference=None
    for name,values in inputs.items():
        parts=[]
        for start in range(0,len(x),64):
            output=model(values[start:start+64])
            torch.testing.assert_close(output['logits'],model.bias+output['global']+output['contributions'].sum(1),atol=0,rtol=0)
            parts.append({k:v.detach().cpu() for k,v in output.items() if k!='embedding'})
        output={key:torch.cat([p[key] for p in parts]) for key in parts[0]}
        z=output['logits'].numpy();r=output['reliability'];e=output['evidence'];c=output['contributions']
        if reference is None:reference=output
        metrics=public_metrics(truth,z,subjects)
        public[name]={'metrics':metrics,'collapse':class_collapse(z),'prediction_stability':float(np.mean(z.argmax(1)==reference['logits'].numpy().argmax(1))),
            'mean_confidence':float(output['logits'].softmax(-1).max(-1).values.mean()),
            'mean_entropy':float(-(output['logits'].softmax(-1)*output['logits'].log_softmax(-1)).sum(-1).mean()),
            'gate_mean':float(r.mean()),'gate_std':float(r.std(unbiased=False)),
            'gate_near_zero':float((r<=.05).float().mean()),'gate_near_one':float((r>=.95).float().mean()),
            'gate_collapsed':bool(float(r.std(unbiased=False))<.01) if model.routing else None,
            'evidence_saturated_fraction':float((e.abs()>=2.85).float().mean()),
            'mean_abs_local_logit':float(c.sum(1).abs().mean()),'mean_sum_abs_local':float(c.abs().sum(1).mean()),
            'mean_abs_global_logit':float(output['global'].abs().mean()),'contribution_change':float((c-reference['contributions']).abs().mean()),
            'same_local_contributions_fraction':float(((c-reference['contributions']).abs().amax(-1)<=1e-6).float().mean()),
            'additive_identity':'BITWISE','label_preservation':'UNVERIFIED_PHYSICAL_SENSITIVITY' if name!='clean' else 'ORIGINAL'}
        private[name]={'logits':z,'contributions':c.numpy(),'reliability':r.numpy(),
            'subject_metrics':classification_metrics(truth,z.argmax(1),subjects)['subject_metrics']}
    return public,private

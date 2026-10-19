"""Atomic synthetic artifacts with explicit integrity and resource gates."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from mist_cove.experiment import metrics
from mist_cove.phase2.provenance import digest,file_sha
from mist_evidence.runtime import atomic_json,atomic_torch_save,rng_state,restore_rng,load_trusted

ROOT=Path(__file__).resolve().parents[3]


def verify_files(proof):
    for path,expected in proof.items():
        if file_sha(ROOT/path)!=expected: raise ValueError('integrity mismatch: '+path)


def historical_proofs():
    names=['experiments/mist_cove/phase4_preregistration_v2.json',
           'reports/mist_cove/phase4_v1/publication_manifest.json',
           'reports/mist_cove/phase4_v1/cuda_publication_manifest_v1.json']
    for name in names:
        value=json.loads((ROOT/name).read_text());verify_files(value.get('sha256',value.get('files',{})))
    source=json.loads((ROOT/'reports/mist_cove/phase4_v1/cuda_measurements_v1.json').read_text())['implementation']
    verify_files(source['files'])
    return {'files':{n:file_sha(ROOT/n) for n in names},'historical_implementation_sha256':source['sha256']}


def frozen_proofs():
    path=ROOT/'experiments/mist_cove/phase5_preregistration_v1.json'
    proof=json.loads(path.read_text());verify_files(proof['sha256'])
    return {'sha256':file_sha(path),'files':proof['sha256']}


def execution_proofs():
    path=ROOT/'experiments/mist_cove/phase5_execution_manifest_v1.json'
    value=json.loads(path.read_text());verify_files(value['files'])
    return {'manifest_sha256':file_sha(path),'implementation_sha256':digest(value['files']),'files':value['files']}


def private_output(path, scope):
    # Reject scope before filesystem inspection, including exists()/resolve().
    if scope!='SYNTHETIC_ONLY': raise ValueError('synthetic-only scope before file access')
    path=Path(path)
    resolved=path.resolve();allowed=(ROOT/'mist_transfer_runs/mist_cove/phase5_v1').resolve()
    if allowed not in resolved.parents: raise ValueError('output must be beneath ignored Phase V root')
    if path.exists(): raise FileExistsError('preserve previous run')
    path.mkdir(parents=True)
    return path


class Budget:
    def __init__(self,seconds): self.start=time.perf_counter();self.seconds=seconds
    def check(self):
        if time.perf_counter()-self.start>self.seconds: raise TimeoutError('prespecified synthetic wall budget exceeded')
    @property
    def elapsed(self): return time.perf_counter()-self.start


def wilson(correct,n,z=2.576):
    if not 0<=correct<=n or n<=0: raise ValueError('invalid accuracy count')
    p=correct/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,middle-half),min(1.,middle+half)]


@torch.no_grad()
def predict(model,x,budget=None):
    model.eval();out=[]
    for start in range(0,len(x),64):
        if budget: budget.check()
        value=model(x[start:start+64])
        if value.shape!=(min(64,len(x)-start),5) or not torch.isfinite(value).all():
            raise FloatingPointError('finite five-output prediction required')
        out.append(value.cpu())
    return torch.cat(out)


def summarize(logits,y,c):
    r=metrics(logits,y);pred=logits.argmax(-1)
    interval=wilson(int((pred==y).sum()),len(y),c['criteria']['wilson_z'])
    r['accuracy_ci99']=interval
    r['restricted_equivalence']=bool(c['criteria']['restricted_equivalence_low']<=interval[0] and interval[1]<=c['criteria']['restricted_equivalence_high'])
    r['leakage_lower_bound_pass']=bool(interval[0]>c['criteria']['historical_leakage_accuracy_lower_bound'])
    r['per_class_accuracy_is_recall']=r['per_stage_recall']
    return r


def chance_reference(y,seed,c):
    from mist_transfer.metrics import confusion_metrics
    g=torch.Generator().manual_seed(seed+912731)
    pred=(torch.arange(len(y))%5)[torch.randperm(len(y),generator=g)]
    r=metrics(torch.zeros(len(y),5),y)
    r.update(confusion_metrics(torch.bincount(y*5+pred,minlength=25).reshape(5,5).numpy()))
    r.update(confidence=.2,ece15=abs(.2-r['accuracy']),prediction_counts=torch.bincount(pred,minlength=5).tolist(),collapsed=False,
             expected_balanced_accuracy=.2,asymptotic_expected_macro_f1=.2,tie_policy='independent balanced random tie break; uniform probabilities')
    r['accuracy_ci99']=wilson(int((pred==y).sum()),len(y),c['criteria']['wilson_z'])
    return r


def fit(model,x,y,c,seed,budget,*,updates=None):
    model.train();g=torch.Generator().manual_seed(seed+711)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c['weight_decay'])
    start=time.perf_counter();losses=[]
    for step in range(updates or c['updates']):
        budget.check();ix=torch.randperm(len(y),generator=g)[:c['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x[ix]),y[ix])
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite synthetic loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('nonfinite parameters')
    return {'optimizer_updates':len(losses),'example_exposures':len(losses)*c['batch_size'],
        'training_seconds':time.perf_counter()-start,'initial_loss':losses[0],'final_loss':losses[-1],
        'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def paired_summary(values,seed,replicates=500):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed+9382)
    means=a[rng.integers(0,len(a),size=(replicates,len(a)))].mean(1)
    return {'mean':float(a.mean()),'bootstrap_ci95':np.quantile(means,[.025,.975]).tolist(),'independent_nuisance_groups':len(a)}


def save_checkpoint(path,model,optimizer,identity):
    path=Path(path)
    if path.exists() or path.with_suffix('.seal.json').exists(): raise FileExistsError('preserve checkpoint')
    atomic_torch_save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()},path)
    atomic_json({'identity':identity,'sha256':file_sha(path)},path.with_suffix('.seal.json'))


def restore_checkpoint(path,model,optimizer,identity):
    path=Path(path);seal=json.loads(path.with_suffix('.seal.json').read_text())
    if seal['identity']!=identity or file_sha(path)!=seal['sha256']: raise ValueError('checkpoint identity/seal mismatch before deserialization')
    data=load_trusted(path)
    if data['identity']!=identity: raise ValueError('checkpoint payload identity mismatch')
    model.load_state_dict(data['model']);optimizer.load_state_dict(data['optimizer']);restore_rng(data['rng'])

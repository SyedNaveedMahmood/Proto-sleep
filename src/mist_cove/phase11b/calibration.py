"""TRAIN-only scalar calibration and a single frozen mechanism follow-up."""
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
from mist_cove.phase10.source_adapter import SourceAdapter
from .campaign import PUBLIC,PRIVATE,setup,write_new
from .security import sha256
from .historical import canonical
from .evaluation import public_metrics,clustered_mean
from .development import local_hf

def verified_partition():
    allocation=json.loads((PRIVATE/'participant_allocation.json').read_text());receipt=json.loads((PUBLIC/'participant_partition.json').read_text())
    if sha256(canonical(allocation))!=receipt['allocation_sha256'] or set(allocation['fit'])&set(allocation['internal_dev']):raise ValueError('participant fitting boundary')
    if len(allocation['fit'])!=10 or len(allocation['internal_dev'])!=5:raise ValueError('participant counts')
    return allocation

def fit_temperature(logits,truth,roles):
    if set(roles)!={'train_fit'}:raise ValueError('calibration fitting requires TRAIN fit only')
    z=np.asarray(logits,dtype=float);y=np.asarray(truth,dtype=int)
    if len(z)!=len(roles) or (y<0).any():raise ValueError('scored fit rows required')
    solution=minimize_scalar(lambda t:float(np.mean(logsumexp(z/t,axis=1)-z[np.arange(len(y)),y]/t)),bounds=(.1,10.),method='bounded',options={'xatol':1e-6,'maxiter':100})
    if not solution.success:raise ValueError('temperature optimization failed')
    return float(solution.x)

def run():
    started=time.perf_counter();allocation=verified_partition();data=np.load(PRIVATE/'train_predictions.npz',allow_pickle=False)
    y=data['truth'];z=data['logits'];s=data['subjects'];fit=np.isin(s,allocation['fit'])&(y>=0);dev=np.isin(s,allocation['internal_dev'])&(y>=0)
    temperature=fit_temperature(z[fit],y[fit],['train_fit']*int(fit.sum()))
    confidence_fit=softmax(z[fit]/temperature,axis=1).max(1);confidence_dev=softmax(z[dev]/temperature,axis=1).max(1)
    thresholds={};rules=[]
    for coverage in (.5,.8,.9,.95):
        threshold=float(np.quantile(confidence_fit,1-coverage));thresholds[str(coverage)]=threshold;keep=confidence_dev>=threshold
        rules.append({'target_fit_coverage':coverage,'actual_dev_coverage':float(keep.mean()),'retained_dev_errors':int((z[dev][keep].argmax(1)!=y[dev][keep]).sum()),
                      'retained_dev_n':int(keep.sum()),'risk':float((z[dev][keep].argmax(1)!=y[dev][keep]).mean()) if keep.any() else None})
    write_new(PRIVATE/'calibration_fit.json',{'temperature':temperature,'thresholds':thresholds,'fit_role':'train_fit','partition_sha256':sha256(canonical(allocation))})
    write_new(PUBLIC/'calibration_metrics.json',{'internal_dev_before':public_metrics(y[dev],z[dev],s[dev]),'internal_dev_after':public_metrics(y[dev],z[dev]/temperature,s[dev]),'abstention':rules,
        'fitted_values_public':False,'fit_scored_epochs':int(fit.sum()),'dev_scored_epochs':int(dev.sum()),'historical_model_saw_all_participants':True,'wall_seconds':time.perf_counter()-started})
    # Additional exploratory mechanism probe was frozen before its responses.
    model,_=setup();adapter=SourceAdapter(model);inputs=np.load(PRIVATE/'audit_inputs.npz',allow_pickle=False)
    rows=[json.loads(v) for v in (PRIVATE/'audit_rows.jsonl').read_text().splitlines()];selected=[r for r in rows if r['primary']]
    ordinals=np.array([r['ordinal'] for r in selected]);x=torch.from_numpy(inputs['normalized'][ordinals]);atoms=torch.tensor(ordinals%15)
    changed=local_hf(x,atoms);outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),32):outputs.append(adapter.native(changed[start:start+32]).numpy())
    out=np.concatenate(outputs);truth=np.array([r['coordinate']['stage'] for r in selected]);subjects=[r['coordinate']['subject'] for r in selected]
    difference=(out.argmax(1)==truth).astype(float)-np.array([r['correct'] for r in selected])
    np.savez_compressed(PRIVATE/'local_hf_followup.npz',logits=out,ordinals=ordinals,atoms=atoms.numpy())
    write_new(PUBLIC/'local_hf_followup.json',{'status':'EXPLORATORY_TRAIN_MECHANISM','metrics':public_metrics(truth,out,subjects),
        'accuracy_change':clustered_mean(difference,subjects),'prediction_changes':int(sum(out.argmax(1)!=np.array([r['native']['prediction'] for r in selected]))),
        'native_full_example_forwards':150,'total_wall_seconds_including_calibration':time.perf_counter()-started,'label_preservation':'ASSUMPTION_NOT_VALIDATED'})

if __name__=='__main__':run()

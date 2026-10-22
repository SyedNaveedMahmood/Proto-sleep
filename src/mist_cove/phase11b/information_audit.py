"""Predefined remaining WP15/WP22 sensitivity controls; no fitting or selection."""
import json
import time
import numpy as np
import torch
from mist_cove.phase10.source_adapter import SourceAdapter
from .campaign import PUBLIC,PRIVATE,setup,write_new
from .evaluation import distribution,clustered_mean,public_metrics,QUALITY_NAMES

def run():
    started=time.perf_counter();rows=[json.loads(v) for v in (PRIVATE/'audit_rows.jsonl').read_text().splitlines() if json.loads(v)['primary']]
    inputs=np.load(PRIVATE/'audit_inputs.npz',allow_pickle=False)['normalized'];model,_=setup();adapter=SourceAdapter(model)
    states={k:[] for k in ('empty','local_only','global_only','greedy_canonical','minimum_positive')};native=np.array([r['native']['confidence'] for r in rows]);subjects=[r['coordinate']['subject'] for r in rows]
    for r in rows:
        for name in ('empty','local_only','global_only'):states[name].append(r['feature_ablations'][name])
        x=torch.from_numpy(inputs[r['ordinal']:r['ordinal']+1]);cache=adapter.cache(x)
        ids=torch.tensor([r['summary']['greedy']['mask'],r['summary']['positive_minimum_mask']]);masks=((ids[:,None]>>torch.arange(15))&1).bool()
        with torch.no_grad():z=adapter.logits(cache,masks)[:,0];p=z.softmax(1)
        for i,name in enumerate(('greedy_canonical','minimum_positive')):
            states[name].append({'confidence':float(p[i].max()),'prediction':int(z[i].argmax()),'entropy':float(-(p[i]*p[i].log()).sum())})
    result={}
    for name,values in states.items():
        confidence=np.array([v['confidence'] for v in values]);changed=np.array([v['prediction']!=r['native']['prediction'] for v,r in zip(values,rows)])
        result[name]={'confidence':distribution(confidence),'entropy':distribution([v['entropy'] for v in values]),'prediction_changes':int(changed.sum()),
                      'confidence_ge_09':int((confidence>=.9).sum()),'changed_prediction_confidence_ge_09':int((changed&(confidence>=.9)).sum()),
                      'confidence_change':clustered_mean(confidence-native,subjects)}
    # Fixed diagnostic normalization mis-specification, never a fitted replacement.
    x=torch.from_numpy(inputs[[r['ordinal'] for r in rows]]);y=np.array([r['coordinate']['stage'] for r in rows]);sensitivity={}
    for name,values in [('mean_offset_01_normalized',x+.1),('scale_overestimate_10percent',x/1.1)]:
        with torch.no_grad():z=torch.cat([adapter.native(values[j:j+32]) for j in range(0,len(x),32)]).numpy()
        sensitivity[name]=public_metrics(y,z,subjects)
    write_new(PUBLIC/'information_loss_metrics.json',{'n':150,'states':result,'normalization_sensitivity':sensitivity,'sensitivity_status':'EXPLORATORY_FIXED_CONTROLS_NO_NORMALIZER_REFIT',
        'native_full_example_forwards':300,'feature_cache_example_forwards':150,'cached_head_queries':300,'wall_seconds':time.perf_counter()-started,
        'missing_information_states_are_physical_EEG':False})

if __name__=='__main__':run()

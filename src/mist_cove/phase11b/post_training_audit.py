"""Exploratory post-fit operator check on fixed first-two internal-dev strata.

All nine existing models are included. No fitting, model selection or new gate.
"""
import json
import os
import stat
import time
import numpy as np
import torch
from .campaign import PUBLIC,PRIVATE,write_new
from .security import AuthenticatedBytes,restricted_load,sha256,stamp
from .development import ReliabilityLedger
from .triad import replacements

def new_phase_state(path):
    # These files were produced by this campaign, not historical artifacts.
    # Their current byte digest is recorded as such; it is not an old trust anchor.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NOATIME)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=1024*1024:raise ValueError('new-state file contract')
        data=b''
        while len(data)<before.st_size:
            block=os.read(fd,min(65536,before.st_size-len(data)))
            if not block:raise ValueError('new-state truncated')
            data+=block
        if stamp(before)!=stamp(os.fstat(fd)) or stamp(before)!=stamp(os.stat(path,follow_symlinks=False)):raise ValueError('new-state substitution')
    finally:os.close(fd)
    return restricted_load(AuthenticatedBytes(data,sha256(data))),sha256(data)

def run():
    started=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    selection=json.loads((PRIVATE/'new_model_selection.json').read_text())['internal_dev'];seen={};indices=[]
    for i,c in enumerate(selection):
        key=(c['subject'],c['stage']);seen[key]=seen.get(key,0)+1
        if seen[key]<=2:indices.append(i)
    audit_selection=json.loads((PRIVATE/'selection.json').read_text())['extended'];lookup={(c['recording_id'],c['epoch']):i for i,c in enumerate(audit_selection)}
    raw=np.load(PRIVATE/'audit_inputs.npz',allow_pickle=False)['wave_uV'];norm=json.loads((PRIVATE/'new_model_normalization.json').read_text())
    original=[];modified=[];metadata=[]
    for i in indices:
        c=selection[i];wave=raw[lookup[(c['recording_id'],c['epoch'])],0];original.append(wave)
        values,rows=replacements(wave)
        for row in rows:
            if row['kind']=='bridge':modified.append(values[row['output_index'],0]);metadata.append(row)
    x=torch.from_numpy((np.array(original,dtype=np.float32)[:,None]-norm['mean_uV'])/norm['std_uV'])
    changed=torch.from_numpy((np.array(modified,dtype=np.float32)[:,None]-norm['mean_uV'])/norm['std_uV']);results=[]
    for seed in (123,456,789):
        for arm in ('M0','M1','M2'):
            payload,digest=new_phase_state(PRIVATE/(arm+'_'+str(seed)+'.pt'))
            if set(payload)!={'state_dict','arm','seed','updates'} or (payload['arm'],payload['seed'],payload['updates'])!=(arm,seed,300):raise ValueError('new-phase state identity')
            model=ReliabilityLedger(arm!='M0').eval();model.load_state_dict(payload['state_dict'],strict=True)
            with torch.no_grad():
                base=model(x);z=torch.cat([model(changed[j:j+64])['logits'] for j in range(0,len(changed),64)])
                internal=model.bias+base['contributions'].sum(1)[:,None]-base['contributions']
            target=base['logits'].argmax(1);original_eval=np.load(PRIVATE/(arm+'_'+str(seed)+'_evaluation.npz'),allow_pickle=False)['clean_logits'][indices]
            torch.testing.assert_close(base['logits'],torch.from_numpy(original_eval),atol=1e-6,rtol=1e-6)
            internal_lost=(internal.argmax(-1)!=target[:,None]).numpy().reshape(-1);physical_lost=(z.argmax(1)!=target.repeat_interleave(15)).numpy()
            compatible=np.array([r['historical_compatibility'] for r in metadata]);correct=(target.numpy()==np.array([selection[i]['stage'] for i in indices]))
            result={'arm':arm,'seed':seed,'examples':len(indices),'correct':int(correct.sum()),'bridge_attempts':len(metadata),'compatible':int(compatible.sum()),
                    'compatible_internal_target_loss':int(internal_lost[compatible].sum()),'compatible_waveform_target_loss':int(physical_lost[compatible].sum()),
                    'compatible_operator_disagreement':int((internal_lost[compatible]!=physical_lost[compatible]).sum()),
                    'all_internal_target_loss':int(internal_lost.sum()),'all_bridge_target_loss':int(physical_lost.sum()),
                    'current_phase_checkpoint_sha256':digest,'restricted_loading':True,'saved_evaluation_identity':True}
            results.append(result)
    write_new(PUBLIC/'new_model_operator_audit.json',{'status':'EXPLORATORY_POST_FIT_ALL_ARMS','selection':'first2 per stage/person in fixed internal-dev coordinates',
        'rows':results,'native_full_example_forwards':9*(len(x)+len(changed)),'cached_additive_head_queries':9*len(changed),
        'wall_seconds':time.perf_counter()-started,'internal_operator':'remove selected local contribution and entire global logit branch; retain bias and other local contributions',
        'original_WIC_operator_identical':False,'checkpoint_digests':'current newly produced campaign artifacts, not historical provenance anchors','no_new_fits':True})

if __name__=='__main__':run()

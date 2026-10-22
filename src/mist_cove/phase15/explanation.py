"""Exact fixed-model ST certificates and equally sized proxy comparators.

Reads only authenticated model and sealed private panel crops. Journal charges
all 65,536 planned head states before a case, including interrupted attempts.
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch

from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase10.response_circuits import summarize_response
from mist_cove.phase11b.historical import authenticate_model, named_state_digest, STATE_SHA
from mist_cove.phase11b.triad import exact_support, feature_rankings
from mist_cove.phase14.event_alignment import matched_candidates
from mist_cove.phase14.event_schema import event_intervals
from mist_cove.phase13.annotation_metrics import mask_intervals,overlap_metrics
from mist_cove.phase9.operators import masks_tensor,margin
from .governance import PRIVATE,PUBLIC,OLD_PRIVATE,LABEL,read,new,sha,canonical,verify_plan_commit


def _write(path,obj):
    path.write_text(json.dumps(obj,allow_nan=False,indent=2)+'\n')


def _truth_table(operator,cache,target,neutral):
    masks=masks_tensor('cpu');pieces=[]
    for i in range(0,32768,512):
        z=operator.logits(cache,masks[i:i+512],neutral=neutral)[:,0]
        pieces.append(margin(z,target).numpy())
    values=np.concatenate(pieces)
    return {'target':target,'margins':values,'truth':values>1e-6}


def _align(events,epoch,masks):
    output={}
    for key in ('EEG Pz-Oz|spindle','EEG Pz-Oz|spindle_strict','EEG Pz-Oz|slow_wave',
                'EEG Fpz-Cz|spindle','EOG horizontal|excursion','EMG submental|low'):
        intervals=event_intervals(events.get(key,[]),epoch)
        output[key]={'event_count':len(intervals),
                     'methods':{name:{str(t):overlap_metrics(mask_intervals(int(mask)),intervals,t)
                                       for t in (0,.1,.25)} for name,mask in masks.items()}}
    return output


def run():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verify_plan_commit();started=time.perf_counter()
    panel=read(PRIVATE/'frozen_panels.json');seal=read(PUBLIC/'panel_seal.json')
    if sha(canonical(panel))!=seal['private_panel_canonical_sha256']:
        raise ValueError('panel identity changed')
    coords=read(PRIVATE/'ST_panel_wave_coordinates.json')
    if coords['panel_sha256']!=seal['private_panel_canonical_sha256']:
        raise ValueError('wave crop panel identity changed')
    if sha((PRIVATE/'ST_panel_waves.npz').read_bytes())!=read(PUBLIC/'rem_wave_diagnostics.json')['private_wave_crop_sha256']:
        raise ValueError('wave crop bytes changed')
    with np.load(PRIVATE/'ST_panel_waves.npz',allow_pickle=False) as z:
        waves=z['EEG Fpz-Cz'].copy()
    cases=[(str(rec),int(epoch)) for rec,epoch in coords['coordinates']]
    if len(cases)!=len(set(cases)) or len(cases)!=len(waves):raise ValueError('coordinate/crop mismatch')
    selected={(c['recording'],c['epoch']):c for name in ('E1','E3') for c in panel[name]}
    if set(cases)!=set(selected):raise ValueError('sealed cases differ')
    model,payload,root,child,auth=authenticate_model();operator=SourceAdapter(model)
    norm=read(OLD_PRIVATE/'source_manifest.json')['normalization']
    maximum=read(__import__('pathlib').Path(__file__).resolve().parents[3]/'configs/mist_cove_phase15_v1.json')['resources']['new_head_queries_attempted_plus_successful_max']
    journal=PRIVATE/'explanation_journal.json';out=PRIVATE/'explanation_rows.jsonl'
    if journal.exists():
        state=read(journal)
        if state['panel_sha256']!=seal['private_panel_canonical_sha256']:raise ValueError('journal panel changed')
    else:
        state={'panel_sha256':seal['private_panel_canonical_sha256'],'completed':0,
               'attempted_head_queries_including_successes':0,'interrupted_cases':0,
               'exact_tables':0,'base_table_full_identity_max_abs':0.}
        _write(journal,state)
    if out.exists():
        with out.open() as stream: completed_lines=sum(1 for _ in stream)
        if completed_lines!=state['completed']:raise ValueError('journal and result row count differ')
    elif state['completed']:raise ValueError('missing private result rows')
    event_cache={}
    with out.open('a') as stream:
        for i in range(state['completed'],len(cases)):
            rec,epoch=cases[i];wave=waves[i]
            # Charge the entire attempted case before the first new masked query.
            planned=65536+5
            if state['attempted_head_queries_including_successes']+planned>maximum:
                raise TimeoutError('Phase XV head-query ceiling would be exceeded')
            state['attempted_head_queries_including_successes']+=planned
            state['interrupted_cases']+=1
            _write(journal,state)
            x=torch.from_numpy(((wave-norm['mean_uV'])/norm['std_uV'])[None,None].astype(np.float32))
            with torch.no_grad():
                native=operator.native(x)[0];target=int(native.argmax())
                cache=operator.cache(x)
                full=operator.logits(cache,torch.ones(1,15,dtype=torch.bool))[0,0]
                err=float((full-native).abs().max())
                if err>1e-6:raise ValueError('exact full-input identity failed')
                if target!=selected[(rec,epoch)]['pred']:raise ValueError('original ST prediction differs')
                base=_truth_table(operator,cache,target,0.)
                neutral=_truth_table(operator,cache,target,.01)
            summary=summarize_response(base);summary.update(exact_support(base['truth']))
            ns=exact_support(neutral['truth'])
            with torch.enable_grad():ranks=feature_rankings(operator,cache,target)
            positive=summary['positive_minimum_mask'];greedy=summary['greedy']['mask']
            candidates=None
            if positive is not None:
                candidates=matched_candidates(wave,positive,greedy,ranks['absolute_gradient'],15011+i)
                k=positive.bit_count();candidates['absolute_gradient_matched']=sum(1<<j for j in ranks['absolute_gradient'][:k])
                candidates['signed_active_positive']=next((c['positive_bits'] for c in summary['signed']['clauses'] if c['negative_bits']==0),positive)
                # Negative and interfering literals are represented separately;
                # no signed clause is silently treated as waveform-ground truth.
            if rec not in event_cache:event_cache[rec]=read(OLD_PRIVATE/'ST_EVAL'/(rec+'_events.json'))['events']
            align=_align(event_cache[rec],epoch,candidates) if candidates is not None else None
            row={'coordinate':{'recording':rec,'epoch':epoch},'status':LABEL,
                 'truth':selected[(rec,epoch)]['truth'],'pred':target,'full_identity_max_abs':err,
                 'summary':summary,'neutral':{'table_disagreement_fraction':float(np.mean(base['truth']!=neutral['truth'])),
                                               'positive_minimum_mask_changed':positive!=ns['positive_minimum_mask'],
                                               'prior_changed':summary['prior_response']!=ns['prior_response']},
                 'candidates':candidates,'proxy_alignment':align,
                 'base_truth_sha256':sha(np.packbits(base['truth']).tobytes()),
                 'neutral_truth_sha256':sha(np.packbits(neutral['truth']).tobytes()),
                 'head_states_charged':planned,'human_event_validation':'NOT_TESTED'}
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            state['completed']+=1;state['interrupted_cases']-=1;state['exact_tables']+=2
            state['base_table_full_identity_max_abs']=max(state['base_table_full_identity_max_abs'],err)
            _write(journal,state)
            if (i+1)%20==0:print(json.dumps({'exact_cases':i+1,'queries_charged':state['attempted_head_queries_including_successes'],
                                             'wall_seconds':round(time.perf_counter()-started,1)}),flush=True)
    if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('original model state changed')
    if state['completed']!=len(cases):raise ValueError('incomplete exact circuit campaign')
    receipt={'status':LABEL,'cases':len(cases),'exact_tables':state['exact_tables'],
             'head_queries_attempted_including_successes':state['attempted_head_queries_including_successes'],
             'head_query_ceiling':maximum,'interrupted_cases':state['interrupted_cases'],
             'base_table_full_identity_max_abs':state['base_table_full_identity_max_abs'],
             'model_named_state_unchanged':True,'scientific_CPU_wall_seconds':time.perf_counter()-started,
             'private_rows_sha256':sha(out.read_bytes()),'query_unit':'cached head state, not full-network training exposure',
             'new_neural_fits':0,'optimizer_updates':0}
    new(PUBLIC/'explanation_execution.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':run()

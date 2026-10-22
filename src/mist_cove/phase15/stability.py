"""Outcome-blind E1 nuisance audit of unchanged original response clauses."""
import json
import time
from collections import defaultdict

import numpy as np
import torch

from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase11b.historical import authenticate_model,named_state_digest,STATE_SHA
from mist_cove.phase11b.triad import waveform_probes,fixed_clause_stability
from .explanation import _truth_table
from .governance import PRIVATE,PUBLIC,OLD_PRIVATE,LABEL,read,new,sha,canonical,verify_plan_commit


def run():
    verify_plan_commit();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    t=time.perf_counter();panel=read(PRIVATE/'frozen_panels.json')
    if sha(canonical(panel))!=read(PUBLIC/'panel_seal.json')['private_panel_canonical_sha256']:
        raise ValueError('panel changed')
    chosen=panel['E1'][:100]  # Frozen E1 hash order, no outcome or proxy reselection.
    cases=[(c['recording'],c['epoch']) for c in chosen]
    crops=read(PRIVATE/'ST_panel_wave_coordinates.json')['coordinates']
    index={(r,int(e)):j for j,(r,e) in enumerate(crops)}
    with np.load(PRIVATE/'ST_panel_waves.npz',allow_pickle=False) as z:waves=z['EEG Fpz-Cz'].copy()
    original={}
    for line in (PRIVATE/'explanation_rows.jsonl').open():
        r=json.loads(line);original[(r['coordinate']['recording'],r['coordinate']['epoch'])]=r
    if len(original)!=360:raise ValueError('missing original exact table rows')
    norm=read(OLD_PRIVATE/'source_manifest.json')['normalization']
    model,*_=authenticate_model();operator=SourceAdapter(model)
    journal=read(PRIVATE/'explanation_journal.json')
    charge_per_case=3*32768+3
    if journal['attempted_head_queries_including_successes']+len(cases)*charge_per_case>38000000:
        raise TimeoutError('nuisance audit would exceed total Phase XV head budget')
    details=[]
    # This is a single bounded pass. Charge the full planned cost before the
    # first query, including any case lost to an interruption.
    total=journal['attempted_head_queries_including_successes']+len(cases)*charge_per_case
    new(PRIVATE/'stability_attempt.json',{'charged_total_including_prior':total,
          'planned_cases':len(cases),'views':['mild_hf','drift','gain']})
    for j,key in enumerate(cases):
        wave=waves[index[key]];row=original[key];target=row['pred']
        probes=waveform_probes(wave,15011+j);views={}
        for name in ('mild_hf','drift','gain'):
            x=torch.from_numpy(((probes[name]-norm['mean_uV'])/norm['std_uV'])[None,None].astype(np.float32))
            with torch.no_grad():
                pred=int(operator.native(x)[0].argmax());cache=operator.cache(x)
                truth=_truth_table(operator,cache,target,0.)['truth']
            views[name]=fixed_clause_stability(row['summary'],truth,pred,target)
        details.append({'coordinate':{'recording':key[0],'epoch':key[1]},'truth':row['truth'],
                        'pred':target,'views':views})
        if (j+1)%25==0:print(json.dumps({'nuisance_cases':j+1,'seconds':round(time.perf_counter()-t,1)}),flush=True)
    if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('historical model mutated')
    new(PRIVATE/'stability_rows.json',{'rows':details})
    summary={}
    for name in ('mild_hf','drift','gain'):
        values=[r['views'][name] for r in details]
        summary[name]={'cases':len(values),'prediction_stable':sum(v['prediction_stable'] for v in values),
             'fixed_primary_clause_valid':sum(v['fixed_clause_valid'] is True for v in values),
             'primary_clause_not_available':sum(v['fixed_clause_valid'] is None for v in values),
             'any_original_active_clause_valid':sum(v['any_original_active_clause_valid'] for v in values),
             'active_substitution':sum(v['active_substitution'] for v in values)}
    new(PUBLIC/'stability_results.json',{'status':LABEL,'E1_cases':len(details),'views':summary,
        'selection':'first 100 in already frozen outcome-independent E1 hash order',
        'new_head_queries_attempted_including_successes':len(cases)*charge_per_case,
        'total_phase15_head_queries_charged':total,'private_rows_sha256':sha((PRIVATE/'stability_rows.json').read_bytes()),
        'model_state_unchanged':True,'scientific_CPU_wall_seconds':time.perf_counter()-t,
        'interpretation':'fixed original clause versus alternative original clause; model prediction tracked separately'})
    print(json.dumps({'cases':len(details),'total_head_queries':total,'views':summary}))


if __name__=='__main__':run()

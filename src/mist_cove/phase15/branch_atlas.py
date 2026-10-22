"""Private multichannel case atlas and source-head branch diagnostics."""
import json
import time
from collections import defaultdict

import numpy as np
import torch

from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase11b.historical import authenticate_model,named_state_digest,STATE_SHA
from .governance import PRIVATE,PUBLIC,OLD_PRIVATE,LABEL,read,new,sha,canonical,verify_plan_commit
from .forensics import collect


def run():
    verify_plan_commit();torch.set_num_threads(2);t=time.perf_counter()
    panel=read(PRIVATE/'frozen_panels.json')
    if sha(canonical(panel))!=read(PUBLIC/'panel_seal.json')['private_panel_canonical_sha256']:
        raise ValueError('panel changed')
    crop_receipt=read(PRIVATE/'ST_panel_wave_coordinates.json')
    if sha(canonical(crop_receipt))!=read(PUBLIC/'rem_wave_diagnostics.json')['private_wave_coordinate_sha256']:
        raise ValueError('ordered wave-crop coordinates changed')
    coords=[(r,int(e)) for r,e in crop_receipt['coordinates']]
    with np.load(PRIVATE/'ST_panel_waves.npz',allow_pickle=False) as z:
        waves={k:z[k].copy() for k in z.files}
    summaries={}
    for s in (PRIVATE/'explanation_rows.jsonl').open():
        r=json.loads(s);c=r['coordinate'];summaries[(c['recording'],c['epoch'])]=r
    norm=read(OLD_PRIVATE/'source_manifest.json')['normalization']
    model,*_=authenticate_model();adapter=SourceAdapter(model)
    rows=[];charge=3*len(coords)
    base=read(PUBLIC/'stability_results.json')['total_phase15_head_queries_charged']
    if base+charge>38000000:raise TimeoutError('head budget exceeded')
    new(PRIVATE/'branch_attempt.json',{'charged_queries':charge,'prior_charged':base})
    for i,key in enumerate(coords):
        wave=waves['EEG Fpz-Cz'][i]
        x=torch.from_numpy(((wave-norm['mean_uV'])/norm['std_uV'])[None,None].astype(np.float32))
        with torch.no_grad():
            cache=adapter.cache(x)
            features=adapter.features(cache,torch.ones(1,15,dtype=torch.bool)).features[0]
            if features.shape[1]<=192:raise ValueError('historical local/global feature split differs')
            ab=torch.cat((features.clone(),features.clone(),torch.zeros_like(features)))
            ab[0,192:]=0;ab[1,:192]=0
            z=adapter.model.emissions(ab[None])[0]
        target=summaries[key]['pred']
        rows.append({'coordinate':key,'truth':summaries[key]['truth'],'pred':target,
                     'local_only_pred':int(z[0].argmax()),'global_only_pred':int(z[1].argmax()),
                     'bias_only_pred':int(z[2].argmax())})
    if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('original state changed')
    new(PRIVATE/'branch_rows.json',{'rows':rows})
    index={tuple(r['coordinate']):r for r in rows};result={}
    for name in ('E1','E2','E3'):
        sub=[index[(c['recording'],c['epoch'])] for c in panel[name]]
        result[name]={'cases':len(sub),'local_only_changes':sum(r['local_only_pred']!=r['pred'] for r in sub),
            'global_only_preserves':sum(r['global_only_pred']==r['pred'] for r in sub),
            'global_preserves_local_changes':sum(r['global_only_pred']==r['pred'] and r['local_only_pred']!=r['pred'] for r in sub),
            'global_preserves_local_changes_and_wrong':sum(r['global_only_pred']==r['pred'] and r['local_only_pred']!=r['pred'] and r['pred']!=r['truth'] for r in sub)}
    new(PUBLIC/'branch_results.json',{'status':LABEL,'panels':result,'head_queries_charged':charge,
        'total_phase15_charged':base+charge,'feature_cache_computations':len(rows),
        'model_state_unchanged':True,'wall_seconds':time.perf_counter()-t,
        'private_rows_sha256':sha((PRIVATE/'branch_rows.json').read_bytes()),
        'interpretation':'feature-branch ablation; not an EEG-channel causal effect'})
    # Reproducible private atlas selects category examples from already frozen E3.
    pool=[(i,c,index[(c['recording'],c['epoch'])],summaries[(c['recording'],c['epoch'])])
          for i,c in enumerate(panel['E3'])]
    grid,_,_=collect();confidence={(r['recording'],r['epoch']):r['confidence'] for r in grid}
    categories={
        'high_confidence_REM_to_W':lambda c,b,e:c['truth']==4 and c['pred']==0 and confidence[(c['recording'],c['epoch'])]>=.9,
        'high_confidence_REM_to_N1':lambda c,b,e:c['truth']==4 and c['pred']==1 and confidence[(c['recording'],c['epoch'])]>=.9,
        'high_confidence_REM_to_N2':lambda c,b,e:c['truth']==4 and c['pred']==2 and confidence[(c['recording'],c['epoch'])]>=.9,
        'low_confidence_REM_error':lambda c,b,e:c['truth']==4 and c['pred']!=4 and confidence[(c['recording'],c['epoch'])]<.5,
        'correct_REM':lambda c,b,e:c['truth']==4 and c['pred']==4,
        'wrong_global_reliant':lambda c,b,e:c['truth']!=c['pred'] and b['global_only_pred']==c['pred'] and b['local_only_pred']!=c['pred'],
        'neutral_sensitive':lambda c,b,e:e['neutral']['table_disagreement_fraction']>0,
        'interfering_clause':lambda c,b,e:any(x['negative_bits'] for x in e['summary']['signed']['clauses'])}
    atlas=[];counts={};selected=set()
    for name,condition in categories.items():
        choices=[(i,c,b,e) for i,c,b,e in pool if condition(c,b,e) and i not in selected]
        if name.startswith('high_confidence'):
            choices.sort(key=lambda v:-confidence[(v[1]['recording'],v[1]['epoch'])])
        if choices:
            i,c,b,e=choices[0];selected.add(i)
            j=coords.index((c['recording'],c['epoch']))
            atlas.append({'category':name,'coordinate':c,'model_branches':b,
                          'explanation':{'positive_mask':e['summary']['positive_minimum_mask'],
                                         'BDD_nodes':e['summary']['BDD_nonterminal_nodes'],
                                         'nonmonotone_edges':e['summary']['nonmonotone_edges']['response_lost_when_adding_atom'],
                                         'neutral_disagreement':e['neutral']['table_disagreement_fraction']},
                          'calibrated_uV_signals':{ch:waves[ch][j].astype(float).tolist() for ch in waves}})
        counts[name]=len(choices)
    new(PRIVATE/'REM_failure_atlas.json',{'status':LABEL,'cases':atlas,'categories_available':counts,
        'private_only':True,'human_review':'NOT_TESTED'})
    new(PUBLIC/'atlas_receipt.json',{'status':LABEL,'private_case_reports':len(atlas),
        'category_candidates':counts,'private_atlas_sha256':sha((PRIVATE/'REM_failure_atlas.json').read_bytes()),
        'raw_waveforms_public':False,'expert_adjudication':'NOT_TESTED'})
    print(json.dumps({'branch':result,'atlas_cases':len(atlas),'queries':charge}))


if __name__=='__main__':run()


def extend_atlas():
    """Add unqueried EOG/quality exemplars from sealed E3 to the private atlas."""
    verify_plan_commit();panel=read(PRIVATE/'frozen_panels.json')
    atlas_path=PRIVATE/'REM_failure_atlas.json';atlas=read(atlas_path)
    waves_path=PRIVATE/'ST_panel_waves.npz'
    if sha(waves_path.read_bytes())!=read(PUBLIC/'rem_wave_diagnostics.json')['private_wave_crop_sha256']:
        raise ValueError('crop bytes changed')
    crop_receipt=read(PRIVATE/'ST_panel_wave_coordinates.json')
    if sha(canonical(crop_receipt))!=read(PUBLIC/'rem_wave_diagnostics.json')['private_wave_coordinate_sha256']:
        raise ValueError('crop order changed')
    coords=[(r,int(e)) for r,e in crop_receipt['coordinates']];lookup={k:i for i,k in enumerate(coords)}
    with np.load(waves_path,allow_pickle=False) as z:waves={k:z[k].copy() for k in z.files}
    branches={tuple(r['coordinate']):r for r in read(PRIVATE/'branch_rows.json')['rows']}
    summaries={}
    for s in (PRIVATE/'explanation_rows.jsonl').open():
        row=json.loads(s);c=row['coordinate'];summaries[(c['recording'],c['epoch'])]=row
    wf={(r['recording'],r['epoch']):r for r in read(PRIVATE/'ST_REM_wave_features.json')['rows']}
    used={(c['coordinate']['recording'],c['coordinate']['epoch']) for c in atlas['cases']}
    event_cache={}
    def quality(c):
        rec=c['recording'];i=c['epoch']
        if rec not in event_cache:event_cache[rec]=read(OLD_PRIVATE/'ST_EVAL'/(rec+'_events.json'))['events']
        return any(e['start']<(i+1)*30 and e['end']>i*30 for k,v in event_cache[rec].items()
                   if k in ('EEG Fpz-Cz|step','EEG Pz-Oz|extreme','EEG Pz-Oz|clip') for e in v)
    pool=[c for c in panel['E3'] if (c['recording'],c['epoch']) not in used and c['truth']==4 and c['pred']!=4]
    choices={
        'strong_EOG_proxy_wrong':sorted([c for c in pool if wf[(c['recording'],c['epoch'])]['EOG_excursion_count']>0],
             key=lambda c:-wf[(c['recording'],c['epoch'])]['EOG_excursion_count']),
        'EEG_EOG_model_disagreement':sorted([c for c in pool if c['pred']==2 and wf[(c['recording'],c['epoch'])]['EOG_excursion_count']>=5],
             key=lambda c:-wf[(c['recording'],c['epoch'])]['EOG_excursion_count']),
        'fixed_signal_quality_flag':sorted([c for c in pool if quality(c)],key=lambda c:(c['recording'],c['epoch']))}
    for name,cc in choices.items():
        available=[c for c in cc if (c['recording'],c['epoch']) not in used]
        atlas['categories_available'][name]=len(available)
        if not available:continue
        c=available[0];key=c['recording'],c['epoch'];used.add(key);i=lookup[key];e=summaries[key]
        atlas['cases'].append({'category':name,'coordinate':c,'model_branches':branches[key],
          'explanation':{'positive_mask':e['summary']['positive_minimum_mask'],
             'BDD_nodes':e['summary']['BDD_nonterminal_nodes'],
             'nonmonotone_edges':e['summary']['nonmonotone_edges']['response_lost_when_adding_atom'],
             'neutral_disagreement':e['neutral']['table_disagreement_fraction']},
          'calibrated_uV_signals':{ch:waves[ch][i].astype(float).tolist() for ch in waves}})
    atlas_path.write_text(json.dumps(atlas,indent=2,allow_nan=False)+'\n')
    receipt_path=PUBLIC/'atlas_receipt.json';receipt=read(receipt_path)
    receipt.update(private_case_reports=len(atlas['cases']),category_candidates=atlas['categories_available'],
        private_atlas_sha256=sha(atlas_path.read_bytes()),extension_new_head_queries=0)
    receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'private_atlas_cases':len(atlas['cases']),'extension_head_queries':0}))

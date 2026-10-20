"""Sequential bounded probes, exact tables and same-table controls."""
import hashlib
import json
import time
import numpy as np
import torch
from mist_cove.phase9.operators import masks_tensor
from .mock_models import make_mock,mock_input
from .mock_controls import CONTROL_NAMES,ControlAdapter
from .source_adapter import SourceAdapter
from .response_circuits import response_table,summarize_response,canonical_comparators
from .safety import PUBLIC,PRIVATE,mock_request,write_json,no_sensitive_exports
from .verification import require_gates,verify_frozen_mock_core


def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture(name):
    if name.startswith('native_'):
        seed=int(name.split('_')[1]);adapter=SourceAdapter(make_mock(seed).network);x=mock_input(n=1)
    else:
        adapter=ControlAdapter(name);x=adapter.waveform()
    return adapter,x,adapter.cache(x)


def behavior_probe():
    verify_frozen_mock_core()
    mock_request();require_gates(json.loads((PUBLIC/'correctness_receipt.json').read_text()),['identity','noninterference'])
    started=time.perf_counter();rows=[]
    ids=[0,32767,1<<2,21845,10922]+[32767^(1<<j) for j in range(15)]
    masks=((torch.tensor(ids)[:,None]>>torch.arange(15))&1).bool()
    for name in CONTROL_NAMES+('native_123','native_456','native_789'):
        adapter,x,cache=fixture(name);logits=adapter.logits(cache,masks);trace=adapter.features(cache,masks)
        base=int(logits[1,0].argmax());classes=logits[:,0].argmax(-1)
        prior=int(logits[0,0].argmax())
        # Feature-path diagnostics, not additional physical intervention operators.
        f=trace.features[1:2].clone();f[...,-len(adapter.graph['global_features']):]=0
        if isinstance(adapter,SourceAdapter):
            isolated=adapter.model.emissions(f.reshape(1,1,-1))[0,0]
        else:
            score=adapter.score(f)[0,0];isolated=torch.stack((score,-score,score.new_tensor(-3),score.new_tensor(-4),score.new_tensor(-5)))
        changed_mean=trace.features[2:3].clone()
        for j,scale in enumerate(adapter.graph['scales']):
            start=scale['feature_start'];d=scale['embedding_dim'];count=trace.valid_window_counts[2,j]
            changed_mean[...,start:start+d]*=len(scale['windows'])/max(1,int(count))
        if isinstance(adapter,SourceAdapter):alt=adapter.model.emissions(changed_mean.reshape(1,1,-1))[0,0]
        else:
            s=adapter.score(changed_mean)[0,0];alt=torch.stack((s,-s,s.new_tensor(-3),s.new_tensor(-4),s.new_tensor(-5)))
        rows.append({'name':name,'kind':'random_historical_architecture' if name.startswith('native_') else 'privileged_analytic_control',
                     'full_class':base,'empty_class':prior,'sampled_masks':len(ids),'sampled_same_class_fraction':float((classes==base).float().mean()),
                     'sampled_distinct_classes':int(classes.unique().numel()),'global_suppression_only_class':int(isolated.argmax()),
                     'global_suppression_only_max_logit_change':float((isolated-logits[1,0]).abs().max()),
                     'single_atom_fixed_mean_class':int(logits[2,0].argmax()),'single_atom_valid_count_mean_diagnostic_class':int(alt.argmax()),
                     'mean_semantics_max_logit_change':float((alt-logits[2,0]).abs().max()),
                     'oracle_atoms':getattr(adapter,'oracle_atoms',None)})
    result={'version':'phase10-behavior-v1','status':'COMPLETE','rows':rows,'wall_seconds':time.perf_counter()-started,
            'scope':'mock-only; isolated feature removal and alternative denominator are diagnostics, not policy switching','real_artifacts_opened':0}
    no_sensitive_exports(result);write_json(PUBLIC/'behavior_receipt.json',result);return result


def circuit_probe():
    verify_frozen_mock_core()
    mock_request();require_gates(json.loads((PUBLIC/'correctness_receipt.json').read_text()),['identity','noninterference'])
    if not (PUBLIC/'behavior_receipt.json').exists():raise ValueError('X07 required first')
    started=time.perf_counter();deadline=started+600;rows=[];out=PRIVATE/'circuits_v1'
    out.mkdir(exist_ok=False)
    for name in CONTROL_NAMES+('native_123','native_456','native_789'):
        adapter,x,cache=fixture(name);t=time.perf_counter();table=response_table(adapter,cache,deadline=deadline)
        summary=summarize_response(table)
        path=out/(name+'.npz')
        np.savez_compressed(path,truth=table['truth'],margin=table['margins'])
        row={'name':name,'kind':'random_historical_architecture' if name.startswith('native_') else 'privileged_analytic_control',
             **summary,'truth_archive_sha256':file_sha(path),'wall_seconds':time.perf_counter()-t,
             'dependency_identity':adapter.graph['identity'],'cache_bytes':cache.nbytes,
             'availability_queries':32768,'full_target_queries':1,'truth_label':getattr(adapter,'truth_label',None)}
        rows.append(row)
    result={'version':'phase10-circuits-v1','status':'COMPLETE','rows':rows,'wall_seconds':time.perf_counter()-started,
            'response_margin_strictly_greater_than':1e-6,'mask_batch_size':512,'complete_per_mask_tables_public':False,'real_artifacts_opened':0}
    no_sensitive_exports(result);write_json(PUBLIC/'circuit_receipt.json',result);return result


def gradient_ranking(adapter,cache,target):
    f=adapter.features(cache,torch.ones(1,15,dtype=torch.bool)).features.detach().requires_grad_(True)
    if isinstance(adapter,SourceAdapter):
        logits=adapter.model.emissions(f.reshape(1,1,-1))[0,0]
        others=torch.cat((logits[:target],logits[target+1:]));score=logits[target]-others.max()
    else:
        s=adapter.score(f).sum();score=2*s if target==0 else -2*s
    # A literal constant control has derivative zero but no autograd graph.
    grad=(torch.autograd.grad(score,f)[0] if score.requires_grad else torch.zeros_like(f))[0,0]
    scores=torch.zeros(15)
    for embed,scale in zip(cache.local_embeddings,adapter.graph['scales']):
        d=scale['embedding_dim'];start=scale['feature_start'];e=embed[0]
        influence=(e*grad[start:start+d]/len(e)).sum(-1)
        argmax=e.argmax(0)
        for j in range(d):influence[argmax[j]]+=e[argmax[j],j]*grad[start+d+j]
        for w,value in zip(scale['windows'],influence):
            scores[w['atoms']]+=value.detach()/len(w['atoms'])
    global_start=adapter.graph['global_features'][0]['feature_index']
    global_absolute=float((f.detach()[0,0,global_start:]*grad[global_start:]).abs().sum())
    ranking=sorted(range(15),key=lambda j:(-float(scores[j].abs()),j))
    return ranking,{'method':'native-head gradient-times-feature distributed through mean/max to physical atoms; approximate, not additive reconciliation',
                    'uses_model_internals':True,'uses_oracle_to_select':False,'head_backward_queries':1,
                    'global_attribution_absolute_sum':global_absolute,'atom_attribution_absolute_sum':float(scores.abs().sum()),
                    'global_attribution_not_falsely_localized':True}


def comparator_probe():
    verify_frozen_mock_core()
    mock_request();receipt=json.loads((PUBLIC/'circuit_receipt.json').read_text());started=time.perf_counter();rows=[]
    for row in receipt['rows']:
        name=row['name'];path=PRIVATE/'circuits_v1'/(name+'.npz')
        if file_sha(path)!=row['truth_archive_sha256']:raise ValueError('mock table tampered')
        with np.load(path,allow_pickle=False) as archive:truth=archive['truth']
        adapter,x,cache=fixture(name);ranking,method=gradient_ranking(adapter,cache,row['target_class'])
        oracle=getattr(adapter,'oracle_atoms',[])
        comparisons=canonical_comparators(truth,ranking,oracle)
        if name.startswith('native_'):
            for v in (comparisons['ranking'],comparisons['random_matched_size']):
                v['privileged_oracle_precision']=None;v['privileged_oracle_coverage']=None
        rows.append({'name':name,'native_ranking':method,**comparisons,'greedy':row['greedy'],
                     'positive_summary_coverage':row['positive_only']['coverage'],'signed_summary_coverage':row['signed']['coverage'],
                     'signed_prior_only':row['signed']['prior_only'],'original_prediction_correct_for_constructed_label':row['target_class']==row['truth_label'] if row['truth_label'] is not None else None,
                     'fidelity_domain':'canonical mask for ranking/greedy/random; universal completions for clauses; full truth table for ROBDD',
                     'oracle_role':'privileged fixture interval scoring only; no oracle for random historical architectures'})
    result={'version':'phase10-comparators-v1','status':'COMPLETE','rows':rows,'wall_seconds':time.perf_counter()-started,
            'external_explainer_implementations_run':False,'real_artifacts_opened':0}
    no_sensitive_exports(result);write_json(PUBLIC/'comparator_receipt.json',result);return result

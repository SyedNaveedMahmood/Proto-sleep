"""Generated-input correctness on authenticated trained state, before real EEG."""
import time
import torch
from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase10.verification import hidden_change,errors
from mist_cove.phase10.response_circuits import response_table,summarize_response
from .historical import named_state_digest,STATE_SHA


@torch.no_grad()
def trained_adapter_checks(model):
    start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    adapter=SourceAdapter(model);g=torch.Generator().manual_seed(44112);t=torch.arange(3000,dtype=torch.float32)/100
    x=torch.stack((torch.sin(2*torch.pi*2.3*t),torch.sin(2*torch.pi*12.7*t)) )[:,None]+.15*torch.randn((2,1,3000),generator=g)
    cache=adapter.cache(x);full=torch.ones(1,15,dtype=torch.bool)
    features=adapter.features(cache,full).features[0];native_features=model.encode(x)
    actual=adapter.logits(cache,full)[0];native=adapter.native(x)
    torch.testing.assert_close(features,native_features,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(actual,native,atol=1e-6,rtol=1e-6)
    maximum=0.;comparisons=0;controls=0
    for mask_id in (0,1,128,21845,32767^128,32767):
        mask=((torch.tensor(mask_id)>>torch.arange(15))&1).bool()
        for kind in ('random','impulses','sinusoid','rms','dc','broadband','boundary'):
            changed=adapter.cache(hidden_change(x,mask,kind))
            for neutral in (0.,.01):
                original_f=adapter.features(cache,mask[None],neutral=neutral);changed_f=adapter.features(changed,mask[None],neutral=neutral)
                a=adapter.logits(cache,mask[None],neutral=neutral);b=adapter.logits(changed,mask[None],neutral=neutral)
                torch.testing.assert_close(original_f.features,changed_f.features,atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
                if mask_id!=32767:assert torch.equal(original_f.features[...,192:],torch.full_like(original_f.features[...,192:],neutral))
                if mask_id==0:assert int(original_f.valid_window_counts.sum())==0
                maximum=max(maximum,float((a-b).abs().max()));comparisons+=1
            if mask_id!=32767 and kind=='dc':
                assert float((cache.global_features-changed.global_features).abs().max())>1e-6;controls+=1
    rows=[]
    for i in range(2):
        table=response_table(adapter,adapter.cache(x[i:i+1]),deadline=start+120)
        summary=summarize_response(table)
        rows.append({'generated_index':i,'exact_all_rows':summary['exact_all_rows'],'masks':32768,'BDD_nonterminal_nodes':summary['BDD_nonterminal_nodes'],'full_response':summary['full_response'],'zero_response':summary['zero_response']})
    if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('trained state mutated')
    return {'status':'PASS','input_domain':'TWO_FIXED_GENERATED_INPUTS_ON_AUTHENTICATED_TRAINED_STATE','seed':44112,'feature_identity':errors(features,native_features),'logit_identity':errors(actual,native),'atol':1e-6,'rtol':1e-6,'noninterference_comparisons':comparisons,'noninterference_max_absolute':maximum,'broken_global_negative_controls_detected':controls,'windows':169,'features':209,'global_features':17,'exhaustive_tables':rows,'table_head_queries':65538,'state_unchanged':True,'wall_seconds':time.perf_counter()-start,'real_EEG_accessed':False}

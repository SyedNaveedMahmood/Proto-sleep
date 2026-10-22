"""Frozen exact tables, unchanged clauses and guarded waveform comparisons."""
import json
import time
import numpy as np
import torch
from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase10.response_circuits import summarize_response
from mist_cove.phase9.operators import masks_tensor,margin
from mist_cove.phase11b.triad import (waveform_probes,VIEW_NAMES,replacements,exact_support,
    fixed_clause_stability,feature_rankings,triangulation_graph)
from mist_cove.phase11b.historical import named_state_digest,STATE_SHA
from .provenance import PUBLIC,PRIVATE,setup_source,write_new
from .sampling import frozen_selection
from .data import load_cache
from .support_operator import LedgerOperator
from .model_adequacy import WidthLedger
from .training import run_identity
from .seals import load_sealed
from .evaluation import nuisance_views
from .evidence_alignment import candidate_regions
from .resources import measured,check_remaining


class Budget:
    def __init__(self):
        self.started=time.perf_counter();self.queries=0;self.native=0;self.cache=0;self.tables=0;self.exact=0;self.maximum_error=0.
    def check(self,queries=0,native=0,cache=0):
        check_remaining(self.started)
        if self.queries+queries+400000>160000000:raise TimeoutError('prospective cached-state ceiling')
        self.queries+=queries;self.native+=native;self.cache+=cache
    def report(self):
        return {'cached_head_queries':self.queries,'feature_cache_example_forwards':self.cache,
                'native_full_example_forwards':self.native,'tables':self.tables,'exact_tables':self.exact,
                'full_identity_max_abs':self.maximum_error,'wall_seconds':time.perf_counter()-self.started,'GPU_seconds':0,
                'query_unit':'one cached availability state, not a full-network training exposure'}


def compile_table(table,budget):
    s=summarize_response(table);s.update(exact_support(table['truth']));budget.exact+=1
    return s


@torch.no_grad()
def table_for(operator,cache,target,budget,neutral=0.):
    budget.check(queries=32768);budget.tables+=1
    if isinstance(operator,LedgerOperator):return operator.response_table(cache,target,neutral)
    masks=masks_tensor('cpu');values=[]
    for start in range(0,32768,512):values.append(margin(operator.logits(cache,masks[start:start+512],neutral=neutral)[:,0],target).numpy())
    values=np.concatenate(values)
    return {'target':target,'margins':values,'truth':values>1e-6}


@torch.no_grad()
def forward(operator,x,budget):
    budget.check(native=len(x))
    return operator.model(x)['logits'] if isinstance(operator,LedgerOperator) else operator.native(x)


@torch.no_grad()
def analyze_case(operator,x,wave,stage,views,budget,neutral=False,physical=True):
    z=forward(operator,x,budget)[0];target=int(z.argmax());budget.check(cache=1);cache=operator.cache(x)
    budget.check(queries=1);full=operator.logits(cache,torch.ones(1,15,dtype=torch.bool))
    if not isinstance(operator,LedgerOperator):full=full[:,0]
    err=float((full[0]-z).abs().max());budget.maximum_error=max(budget.maximum_error,err)
    torch.testing.assert_close(full[0],z,atol=1e-6,rtol=1e-6)
    table=table_for(operator,cache,target,budget);s=compile_table(table,budget)
    row={'correct':target==stage,'prediction':target,'stage':int(stage),'confidence':float(z.softmax(-1).max()),
         'summary':s,'full_identity_max_abs':err,'views':{}}
    if isinstance(operator,LedgerOperator):
        c=cache.contributions.sum(0);g=operator.model.global_head(cache.global_features);bias=operator.model.bias
        ablations={'local_only':bias+c,'global_only':bias+g,'bias_only':bias}
        row['local_magnitude']=float(c.abs().mean());row['global_magnitude']=float(g.abs().mean())
        budget.check(queries=3)
    else:
        f=operator.features(cache,torch.ones(1,15,dtype=torch.bool)).features[0]
        ab=torch.cat((f.clone(),f.clone(),torch.zeros_like(f)));ab[0,192:]=0;ab[1,:192]=0
        budget.check(queries=3)
        ablations={k:v for k,v in zip(('local_only','global_only','bias_only'),operator.model.emissions(ab[None])[0])}
        with torch.enable_grad():ranking=feature_rankings(operator,cache,target)['absolute_gradient']
        budget.check(queries=1)
        positive=s['positive_minimum_mask'];greedy=s['greedy']['mask']
        row['alignment_candidates']=None if positive is None else candidate_regions(wave,positive,greedy,ranking)
    row['branch_predictions']={k:int(v.argmax()) for k,v in ablations.items()}
    all_tables=[table['margins']]
    for name,vx in views.items():
        vz=forward(operator,vx,budget)[0];budget.check(cache=1);vcache=operator.cache(vx)
        vt=table_for(operator,vcache,target,budget);vs=compile_table(vt,budget);all_tables.append(vt['margins'])
        row['views'][name]={**fixed_clause_stability(s,vt['truth'],int(vz.argmax()),target),
                            'exact_all_rows':vs['exact_all_rows'],'BDD_nonterminal_nodes':vs['BDD_nonterminal_nodes']}
    nt=None
    if neutral:
        nt=table_for(operator,cache,target,budget,.01);ns=compile_table(nt,budget);all_tables.append(nt['margins'])
        row['neutral']={'table_disagreement_fraction':float(np.mean(table['truth']!=nt['truth'])),
            'positive_minimum_mask_changed':s['positive_minimum_mask']!=ns['positive_minimum_mask'],
            'prior_changed':s['prior_response']!=ns['prior_response'],'summary':ns}
    metadata=[]
    if physical:
        repl,metadata=replacements(wave)
        norm=analyze_case.normalization
        rx=torch.from_numpy((repl-norm['mean_uV'])/norm['std_uV'])
        rz=forward(operator,rx,budget)
        for item in metadata:
            if item['output_index'] is None:continue
            response=rz[item['output_index']];item['target_retained']=bool(margin(response[None],target)[0]>1e-6)
            item['prediction']=int(response.argmax())
        row['triad']=triangulation_graph(s,table['truth'],metadata,row['views'],table['truth'] if nt is None else nt['truth'],row['confidence'])
    row['replacements']=metadata
    return row,np.stack(all_tables)


def verified_new_model(config,size,route,seed,step):
    name=f'{size}_{route}_{seed}';path=PRIVATE/'model_states'/name/f'step_{step:06d}.pt'
    public=json.loads((PUBLIC/f'{name}_{step}.json').read_text())
    seal=json.loads(path.with_suffix('.seal.json').read_text())
    if seal['byte_sha256']!=public['checkpoint_byte_sha256'] or seal['state_digest']!=public['state_digest']:
        raise ValueError('new-state seal differs from published milestone digest')
    payload,_=load_sealed(path,{**run_identity(config,size,route,seed),'step':step})
    if named_state_digest(payload['state_dict'])!=public['state_digest']:raise ValueError('named state differs')
    with torch.random.fork_rng(devices=[]):model=WidthLedger(size,route=='R1').eval()
    model.load_state_dict(payload['state_dict'],strict=True)
    return model,LedgerOperator(model)


def run(resume_empty=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if (PRIVATE/'mechanistic_started.json').exists():
        if not resume_empty:raise ValueError('mechanistic campaign already attempted')
        rows=PRIVATE/'original_primary_mechanistic.jsonl'
        if rows.exists() and rows.stat().st_size!=0:raise ValueError('only empty engineering-failed attempt can continue here')
        failure=PUBLIC/'mechanistic_engineering_failure.json'
        if not failure.exists():
            if not rows.exists():raise ValueError('expected empty initial attempt')
            for name in ('original_primary_mechanistic.jsonl','original_primary_tables.npy','mechanistic_started.json'):
                (PRIVATE/name).rename(PRIVATE/(name+'.gradient_context_failure'))
            write_new(failure,{'status':'ENGINEERING_BUG_FIXED','completed_cases':0,'cached_states_computed_before_failure':32768,
                'reason':'native gradient comparator was called within no_grad; enable_grad required for feature ranking',
                'original_attempt_preserved':True,'no_fit_rerun':True,'scientific_conditions_or_thresholds_changed':False})
        else:
            if rows.exists():raise ValueError('unexpected repeated attempt')
            (PRIVATE/'mechanistic_started.json').rename(PRIVATE/'mechanistic_started.json.entry_guard_failure')
    selection=frozen_selection();data=load_cache();contexts=load_cache('annotation_context.npz');primary=selection['primary']
    budget=Budget();write_new(PRIVATE/'mechanistic_started.json',{'primary':150,'new_models':36,'new_model_cases':25,'maximum_queries':160000000})
    with measured('explanation_campaign_gradient_fix' if resume_empty else 'explanation_campaign','A_AND_B') as counts:
        model,access=setup_source();operator=SourceAdapter(model);norm=access.normalization;analyze_case.normalization=norm
        margins=np.lib.format.open_memmap(PRIVATE/'original_primary_tables.npy',mode='w+',dtype=np.float32,shape=(len(primary),5,32768))
        with (PRIVATE/'original_primary_mechanistic.jsonl').open('x') as stream:
            for i,c in enumerate(primary):
                wave=contexts['wave_uV'][i,1];x=torch.from_numpy(((wave-norm['mean_uV'])/norm['std_uV'])[None,None])
                probes=waveform_probes(wave,123+i)
                views={k:torch.from_numpy(((probes[k]-norm['mean_uV'])/norm['std_uV'])[None,None]) for k in VIEW_NAMES}
                row,tables=analyze_case(operator,x,wave,c['stage'],views,budget,neutral=True)
                row.update(ordinal=i,coordinate=c);margins[i]=tables;stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if (i+1)%25==0:print(json.dumps({'original_completed':i+1,'seconds':time.perf_counter()-budget.started}),flush=True)
        margins.flush()
        if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('historical state changed')
        config=json.loads((PUBLIC/'adequacy_preregistration.json').read_text());norm=selection['normalization'];analyze_case.normalization=norm
        lookup={(c['recording_id'],c['epoch']):i for i,c in enumerate(selection['training']['fit']+selection['training']['internal_dev'])}
        panel=selection['mechanistic'];indices=[lookup[c['recording_id'],c['epoch']] for c in panel]
        xx=torch.from_numpy(data['x'][indices]);waves=data['wave_uV'][indices,0];vv=nuisance_views(xx,norm)
        for seed in (123,456,789):
            for size in ('S','L'):
                for route in ('R0','R1'):
                    for step in (300,1000,3000):
                        model,operator=verified_new_model(config,size,route,seed,step);name=f'{size}_{route}_{seed}_{step}'
                        nviews=4 if step==3000 else 3
                        mm=np.lib.format.open_memmap(PRIVATE/(name+'_tables.npy'),mode='w+',dtype=np.float32,shape=(len(panel),nviews,32768))
                        with (PRIVATE/(name+'_mechanistic.jsonl')).open('x') as stream:
                            for i,c in enumerate(panel):
                                views={k:vv[k][i:i+1] for k in ('mild_local_HF','gain')}
                                row,tables=analyze_case(operator,xx[i:i+1],waves[i],c['stage'],views,budget,neutral=step==3000)
                                row.update(ordinal=i,coordinate=c);mm[i]=tables;stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                        mm.flush();operator.check()
                        print(json.dumps({'new_state_completed':name,'tables':budget.tables,'seconds':time.perf_counter()-budget.started}),flush=True)
        counts.update(**budget.report())
        write_new(PUBLIC/'mechanistic_execution.json',{'status':'COMPLETE','original_primary':150,'new_state_cases':900,
            'historical_state_unchanged':True,'source_access':access.counts,'resources':budget.report(),
            'secondary':'annotation-only prior-enriched cases, no new mechanistic pooling',
            'new_view_namespace':'25-case mechanistic ordinal, matched across all36states; independent from1000-case development ordinal',
            'between_family_caution':'original whole-epoch mild32Hz differs from new localized35Hz; no fair cross-family nuisance superiority claim'})

if __name__=='__main__':run()

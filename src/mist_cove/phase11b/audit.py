"""Frozen real-source audit. Detailed outputs remain in the ignored run directory."""
import json
import time
import resource
import numpy as np
import torch
from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase10.response_circuits import summarize_response,canonical_comparators
from mist_cove.phase9.operators import masks_tensor,margin
from .campaign import PUBLIC,PRIVATE,setup,write_new
from .historical import canonical,named_state_digest,STATE_SHA
from .security import sha256
from .triad import (waveform_probes,VIEW_NAMES,distortion,replacements,exact_support,
                    feature_rankings,fixed_clause_stability,triangulation_graph)

class Budget:
    def __init__(self):
        self.start=time.perf_counter();self.head=131078;self.full=4;self.cache=4
        self.table_seconds=0.;self.compile_seconds=0.;self.native_seconds=0.;self.cache_seconds=0.;self.intervention_seconds=0.
    def check(self,head=0,full=0,cache=0):
        if time.perf_counter()-self.start>7000 or self.head+head>50000000 or self.full+self.cache+full+cache>250000:
            raise TimeoutError('frozen resource ceiling would be exceeded')
        self.head+=head;self.full+=full;self.cache+=cache
    def report(self):
        return {'cached_head_queries_including_prior_checks':self.head,'native_full_example_forwards':self.full,
                'feature_cache_example_forwards':self.cache,'table_seconds':self.table_seconds,'compile_seconds':self.compile_seconds,
                'native_seconds':self.native_seconds,'cache_seconds':self.cache_seconds,'intervention_seconds':self.intervention_seconds,
                'wall_seconds':time.perf_counter()-self.start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'optimizer_updates':0,'new_fits':0,'GPU_seconds':0}

@torch.no_grad()
def fixed_table(adapter,cache,target,budget,neutral=0.):
    budget.check(head=32768);start=time.perf_counter();masks=masks_tensor('cpu');values=[]
    for offset in range(0,32768,512):
        budget.check();values.append(margin(adapter.logits(cache,masks[offset:offset+512],neutral=neutral)[:,0],target).numpy())
    values=np.concatenate(values);budget.table_seconds+=time.perf_counter()-start
    return {'target':target,'margins':values,'truth':values>1e-6,'availability_queries':32768,'full_target_queries':0}

def summary(table,budget):
    start=time.perf_counter();result=summarize_response(table);result.update(exact_support(table['truth']))
    budget.compile_seconds+=time.perf_counter()-start
    return result

@torch.no_grad()
def native(adapter,x,budget):
    budget.check(full=len(x));start=time.perf_counter();z=adapter.native(x);budget.native_seconds+=time.perf_counter()-start;return z

def cache_input(adapter,x,budget):
    budget.check(cache=len(x));start=time.perf_counter();c=adapter.cache(x);budget.cache_seconds+=time.perf_counter()-start;return c

def logits_record(z,target):
    p=z.softmax(-1);return {'logits':z.tolist(),'prediction':int(z.argmax()),'confidence':float(p.max()),
                           'entropy':float(-(p*p.clamp_min(1e-30).log()).sum()),'target_margin':float(margin(z[None],target)[0])}

def run():
    budget=Budget();model,access=setup();adapter=SourceAdapter(model)
    coordinates=json.loads((PRIVATE/'selection.json').read_text());selection=json.loads((PUBLIC/'sample_selection.json').read_text())
    if sha256(canonical(coordinates))!=selection['selection_sha256']:raise ValueError('sample identity differs')
    if (PRIVATE/'audit_rows.jsonl').exists():raise ValueError('audit already attempted; no silent rerun')
    samples=coordinates['extended'];primary={(x['recording_id'],x['epoch']) for x in coordinates['primary']}
    normalized=np.empty((len(samples),1,3000),np.float32);waves=np.empty_like(normalized)
    records=access.records('train')
    for record in records:
        indices=[i for i,c in enumerate(samples) if c['recording_id']==record['recording_id']]
        if not indices:continue
        with access.reader(record) as reader:
            for start in range(0,len(indices),256):
                group=indices[start:start+256];x,w=access.read(record,reader,[samples[i]['epoch'] for i in group])
                normalized[group]=x;waves[group]=w
    np.savez_compressed(PRIVATE/'audit_inputs.npz',normalized=normalized,wave_uV=waves)
    all_margins=np.lib.format.open_memmap(PRIVATE/'audit_margins.npy',mode='w+',dtype=np.float32,shape=(len(samples),32768))
    view_margins=np.lib.format.open_memmap(PRIVATE/'primary_view_margins.npy',mode='w+',dtype=np.float32,shape=(len(primary),4,32768))
    primary_index=0;count=0;status='COMPLETE'
    try:
        with (PRIVATE/'audit_rows.jsonl').open('x') as stream:
            for ordinal,coordinate in enumerate(samples):
                x=torch.from_numpy(normalized[ordinal:ordinal+1]);wave=waves[ordinal,0]
                z=native(adapter,x,budget)[0];target=int(z.argmax());cache=cache_input(adapter,x,budget)
                budget.check(head=1);full=adapter.logits(cache,torch.ones(1,15,dtype=torch.bool))[0,0]
                torch.testing.assert_close(z,full,atol=1e-6,rtol=1e-6)
                table=fixed_table(adapter,cache,target,budget);s=summary(table,budget);all_margins[ordinal]=table['margins']
                row={'ordinal':ordinal,'coordinate':coordinate,'primary':(coordinate['recording_id'],coordinate['epoch']) in primary,
                     'native':logits_record(z,target),'correct':target==coordinate['stage'],'summary':s,
                     'full_identity_max_abs':float((full-z).abs().max())}
                # Explicit feature-space local/global ablations, separate from A.
                f=adapter.features(cache,torch.ones(1,15,dtype=torch.bool)).features[0]
                ablations=torch.cat((f.clone(),f.clone(),torch.zeros_like(f)))
                ablations[0,192:]=0;ablations[1,:192]=0
                budget.check(head=3)
                with torch.no_grad():abz=model.emissions(ablations[None])[0]
                row['feature_ablations']={name:logits_record(v,target) for name,v in zip(('local_only','global_only','empty'),abz)}
                if row['primary']:
                    ranks=feature_rankings(adapter,cache,target);budget.check(head=1)
                    row['comparators']={name:canonical_comparators(table['truth'],rank,[]) for name,rank in ranks.items()}
                    neutral=fixed_table(adapter,cache,target,budget,.01);view_margins[primary_index,0]=neutral['margins']
                    row['neutral_summary']=summary(neutral,budget)
                    probes=waveform_probes(wave,123+ordinal);names=list(probes)
                    px=np.stack([probes[k] for k in names])[:,None]
                    pnorm=(px-access.normalization['mean_uV'])/access.normalization['std_uV']
                    pz=native(adapter,torch.from_numpy(pnorm),budget)
                    row['probes']={name:{**logits_record(v,target),'distortion':distortion(wave,probes[name]),'compatibility_claim':'CONTINUOUS_DISTORTION_ONLY'} for name,v in zip(names,pz)}
                    torch.testing.assert_close(pz[0],z,atol=1e-6,rtol=1e-6)
                    stability={}
                    for vi,name in enumerate(VIEW_NAMES):
                        pi=names.index(name);vcache=cache_input(adapter,torch.from_numpy(pnorm[pi:pi+1]),budget)
                        vtable=fixed_table(adapter,vcache,target,budget);view_margins[primary_index,vi+1]=vtable['margins']
                        # Every nuisance table is also compiled and verified, even if
                        # original target no longer wins the new full input.
                        vs=summary(vtable,budget)
                        stability[name]={**fixed_clause_stability(s,vtable['truth'],int(pz[pi].argmax()),target),
                                         'exact_all_rows':vs['exact_all_rows'],'BDD_nonterminal_nodes':vs['BDD_nonterminal_nodes']}
                    start=time.perf_counter();repl,metadata=replacements(wave);budget.intervention_seconds+=time.perf_counter()-start
                    rnorm=(repl-access.normalization['mean_uV'])/access.normalization['std_uV']
                    rz=native(adapter,torch.from_numpy(rnorm),budget)
                    for met in metadata:
                        if met['output_index'] is None:continue
                        response=logits_record(rz[met['output_index']],target);met.update(response)
                        met['target_retained']=response['target_margin']>1e-6
                    row['replacements']=metadata
                    row['triad']=triangulation_graph(s,table['truth'],metadata,stability,neutral['truth'],row['native']['confidence'])
                    primary_index+=1
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();count+=1
                if count%50==0:print(json.dumps({'completed_extended':count,'completed_primary':primary_index,'wall_seconds':time.perf_counter()-budget.start}),flush=True)
    except Exception as exc:
        status='INCOMPLETE';write_new(PUBLIC/'audit_failure.json',{'status':status,'exception_type':type(exc).__name__,'completed':count,'no_case_replacement':True})
        raise
    finally:
        all_margins.flush();view_margins.flush()
        write_new(PUBLIC/'audit_execution.json',{'status':status,'completed':count,'primary_completed':primary_index,
                  'selection_sha256':selection['selection_sha256'],'source_access':access.counts,'resources':budget.report(),
                  'state_unchanged':named_state_digest(model.state_dict())==STATE_SHA})

if __name__=='__main__':run()

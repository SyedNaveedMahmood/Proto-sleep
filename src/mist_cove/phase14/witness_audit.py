"""Fixed source/ST witnesses; exact reconstruction, guarded physical views."""
import json
import time
from functools import lru_cache
import numpy as np
import torch
from mist_cove.phase13.mechanistic import analyze_case
from mist_cove.phase10.source_adapter import SourceAdapter
from mist_cove.phase11b.historical import authenticate_model,named_state_digest,STATE_SHA
from mist_cove.phase11b.triad import waveform_probes,VIEW_NAMES
from mist_cove.phase9.operators import masks_tensor
from .common import ROOT,PRIVATE,PUBLIC,read_json,write_new,measured,ledger,check_budget,sha256,canonical
from .source_audit import cached_selection
from .event_alignment import matched_candidates,align
from .event_schema import event_intervals,validate_record_events


class Budget:
    def __init__(self):
        self.started=time.perf_counter();self.queries=0;self.native=0;self.cache=0;self.tables=0;self.exact=0;self.maximum_error=0.
    def check(self,queries=0,native=0,cache=0):
        prior=sum(r.get('cached_head_queries',0) for r in read_json(PUBLIC/'resource_stages.json').values())
        check_budget(self.started,prior+self.queries+queries+1000000)
        self.queries+=queries;self.native+=native;self.cache+=cache
    def report(self):return {'cached_head_queries':self.queries,'native_full_example_forwards':self.native,'feature_cache_example_forwards':self.cache,'tables':self.tables,'exact_tables':self.exact,'full_identity_max_abs':self.maximum_error}


@lru_cache(maxsize=4)
def record_events(role,recording):
    value=read_json(PRIVATE/role/(recording+'_events.json'))
    for rows in value['events'].values():validate_record_events(rows,value['duration'])
    return value


def run(role):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if role=='ST_EVAL':
        from .protocol import verify_st_seal
        verify_st_seal()
    if (PRIVATE/(role+'_witness_rows.jsonl')).exists():raise ValueError('witness audit already attempted')
    receipt=read_json(PUBLIC/(role+'_execution.json'));path=PRIVATE/(role+'_audit_inputs.npz')
    if sha256(path.read_bytes())!=receipt['audit_input_sha256']:raise ValueError('audit input identity differs')
    with np.load(path,allow_pickle=False) as archive:waves={k:archive[k] for k in archive.files}
    coords=read_json(PRIVATE/(role+'_audit_coordinates.json'))
    if sha256(canonical(coords))!=receipt['coordinates_sha256']:raise ValueError('audit coordinates changed')
    samples=coords['samples'];model,payload,root,child,auth=authenticate_model();operator=SourceAdapter(model);norm=read_json(PRIVATE/'source_manifest.json')['normalization'];analyze_case.normalization=norm
    budget=Budget();nprimary=150 if role=='SC_TRAIN' else len(samples)
    margins=np.lib.format.open_memmap(PRIVATE/(role+'_base_tables.npy'),mode='w+',dtype=np.float32,shape=(len(samples),32768))
    vv=np.lib.format.open_memmap(PRIVATE/(role+'_primary_view_tables.npy'),mode='w+',dtype=np.float32,shape=(nprimary,4,32768))
    with measured(role+'_witness_audit') as counts:
        with (PRIVATE/(role+'_witness_rows.jsonl')).open('x') as stream:
            for i,c in enumerate(samples):
                wave=waves['EEG Fpz-Cz'][i];x=torch.from_numpy(((wave-norm['mean_uV'])/norm['std_uV'])[None,None]);probes=waveform_probes(wave,123+i)
                views={name:torch.from_numpy(((probes[name]-norm['mean_uV'])/norm['std_uV'])[None,None]) for name in VIEW_NAMES} if i<nprimary else {}
                row,tables=analyze_case(operator,x,wave,c['stage'],views,budget,neutral=i<nprimary,physical=i<nprimary)
                margins[i]=tables[0]
                if i<nprimary:vv[i]=tables[1:]
                rankings=row['alignment_candidates'];ranking_mask=rankings['gradient_matched_budget'] if rankings else 0
                # Recover a full deterministic native gradient ranking for a
                # differently registered spectral comparator; no proxy used.
                from mist_cove.phase11b.triad import feature_rankings
                with torch.enable_grad():
                    budget.check(cache=1,queries=1);cache=operator.cache(x);rank=feature_rankings(operator,cache,row['prediction'])['absolute_gradient']
                candidate=matched_candidates(wave,row['summary']['positive_minimum_mask'],row['summary']['greedy']['mask'],rank,14024+i)
                event_record=record_events(role,c['recording_id']);events=event_record['events']
                # Permute Pz-Oz event epochs within the SAME recording, using a
                # fixed cyclic1epoch permutation; no cross-person/night joins.
                other_epoch=(c['epoch']+1)%event_record['n_epochs'];permuted={key:event_intervals(value,other_epoch) for key,value in events.items() if key.startswith('EEG Pz-Oz')}
                row['proxy_alignment']=align(events,c['epoch'],candidate,permuted)
                row['detector_record_statuses']=event_record['statuses'];row['coordinate']=c;row['ordinal']=i;row['primary']=i<nprimary;row['matched_candidates']=candidate
                masks=[0,row['summary']['positive_minimum_mask'],row['summary']['greedy']['mask']];validm=[m for m in masks if m is not None]
                budget.check(queries=len(validm));states=torch.tensor([[(m>>j)&1 for j in range(15)] for m in validm],dtype=torch.bool)
                with torch.no_grad():logits=operator.logits(cache,states)[:,0]
                row['partial_confidence']={label:float(z.softmax(-1).max()) for label,z in zip([label for label,m in zip(('empty','exact_positive','greedy'),masks) if m is not None],logits)}
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if (i+1)%25==0:print(json.dumps({'role':role,'witness_cases':i+1,'queries':budget.queries,'elapsed':time.perf_counter()-budget.started}),flush=True)
        margins.flush();vv.flush()
        if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('historical model mutated')
        result={'status':'COMPLETE','role':role,'primary_cases':nprimary,'prevalence_cases':len(samples)-nprimary,'resources':budget.report(),'historical_state_unchanged':True,
            'query_accounting':'cached availability/head operations, not complete neural inference','margin_columns':['base','mild_hf','drift','gain','neutral_.01'],
            'proxy_alignment':'agreement with automated proxies; no expert event accuracy','cross_channel_permutation':'Pz-Oznextphysicalepoch cyclicwithinrecording, fixed policy',
            'private_rows_sha256':sha256((PRIVATE/(role+'_witness_rows.jsonl')).read_bytes())}
        counts.update(**budget.report());write_new(PUBLIC/(role+'_witness_execution.json'),result)
        if role=='SC_TRAIN':ledger([23,24,25,26],'COMPLETE','SCbalanced/prevalence exact circuits, matched proxy controls, correctness and guarded physical views complete',['reports/mist_cove/phase14_v1/SC_TRAIN_witness_execution.json'],science='PENDING_AGGREGATION')
        else:ledger([41,42,43],'COMPLETE','Frozen STstage-balanced full circuits, proxy/control and fixed-clause comparisons complete',['reports/mist_cove/phase14_v1/ST_EVAL_witness_execution.json'],science='PENDING_AGGREGATION')

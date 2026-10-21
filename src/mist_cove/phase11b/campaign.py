"""Explicit phase-owned research entrypoints; never discover source files."""
import json
import time
from pathlib import Path
import torch
from .historical import ROOT, authenticate_model, lineage, canonical
from .security import sha256
from .source import SourceAccess, select_epochs

PUBLIC=ROOT/'reports/mist_cove/phase11b_v1'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase11b_v1'

def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def setup():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,payload,root,child,_=authenticate_model()
    frozen,norm,_=lineage(payload,root,child)
    adapter=json.loads((PUBLIC/'trained_adapter_correctness.json').read_text())
    return model,SourceAccess(frozen,norm,adapter)

def freeze_sample():
    start=time.perf_counter();model,access=setup()
    try:
        records=access.records('train')
        primary,missing_primary=select_epochs(records,2)
        extended,missing_extended=select_epochs(records,10)
        coordinates={'primary':primary,'extended':extended,'missing_primary':missing_primary,'missing_extended':missing_extended}
        write_new(PRIVATE/'selection.json',coordinates)
        receipt={'status':'FROZEN_BEFORE_PREDICTIONS','selection_sha256':sha256(canonical(coordinates)),
                 'primary_count':len(primary),'extended_count':len(extended),
                 'missing_primary':sum(x['missing'] for x in missing_primary),'missing_extended':sum(x['missing'] for x in missing_extended),
                 'source_access':access.counts,'wall_seconds':time.perf_counter()-start,'waveforms_read':False}
    except (FileNotFoundError,PermissionError) as exc:
        receipt={'status':'BLOCKED','reason':'approved TRAIN descriptor unavailable at its exact frozen path',
                 'exception_type':type(exc).__name__,'source_access':access.counts,'wall_seconds':time.perf_counter()-start,'waveforms_read':False}
    write_new(PUBLIC/'sample_selection.json',receipt)
    print(json.dumps(receipt))

def pilot():
    from mist_cove.phase10.source_adapter import SourceAdapter
    from mist_cove.phase10.response_circuits import response_table,summarize_response
    from .historical import named_state_digest,STATE_SHA
    import resource
    start=time.perf_counter();model,access=setup();adapter=SourceAdapter(model)
    coordinates=json.loads((PRIVATE/'selection.json').read_text())
    receipt=json.loads((PUBLIC/'sample_selection.json').read_text())
    if sha256(canonical(coordinates))!=receipt['selection_sha256']:raise ValueError('frozen sample changed')
    records=access.records('train');rows=[]
    try:
        for coordinate in coordinates['primary'][:2]:
            record=next(r for r in records if r['recording_id']==coordinate['recording_id'])
            with access.reader(record) as reader:x,_=access.read(record,reader,[coordinate['epoch']])
            x=torch.from_numpy(x);times={}
            with torch.no_grad():
                t=time.perf_counter();native=adapter.native(x);times['native_seconds']=time.perf_counter()-t
                t=time.perf_counter();cache=adapter.cache(x);times['cache_seconds']=time.perf_counter()-t
                t=time.perf_counter();full=adapter.logits(cache,torch.ones(1,15,dtype=torch.bool))[0]
                torch.testing.assert_close(full,native,atol=1e-6,rtol=1e-6);times['full_identity_seconds']=time.perf_counter()-t
                t=time.perf_counter();table=response_table(adapter,cache,deadline=start+120);times['table_seconds']=time.perf_counter()-t
                t=time.perf_counter();summary=summarize_response(table);times['circuit_clause_seconds']=time.perf_counter()-t
            rows.append({'full_identity_max_abs':float((full-native).abs().max()),'exact_all_rows':summary['exact_all_rows'],'mask_count':32768,'timing':times})
        if named_state_digest(model.state_dict())!=STATE_SHA:raise ValueError('model mutation')
        result={'status':'PASS','epochs':rows,'head_queries':65540,'native_example_forwards':2,'feature_cache_examples':2,
                'projected_1350_tables_seconds':1350*sum(sum(x['timing'].values()) for x in rows)/2,
                'cuda':'NOT_TESTED_CPU_FIRST','state_unchanged':True}
    except (FileNotFoundError,PermissionError) as exc:
        result={'status':'BLOCKED','reason':'approved source unavailable at exact frozen path; no alternate locations attempted',
                'exception_type':type(exc).__name__,'completed_epochs':len(rows)}
    result.update(source_access=access.counts,wall_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_new(PUBLIC/'source_pilot.json',result);print(json.dumps(result))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze-sample','pilot'])
    args=parser.parse_args()
    if args.action=='freeze-sample':freeze_sample()
    elif args.action=='pilot':pilot()

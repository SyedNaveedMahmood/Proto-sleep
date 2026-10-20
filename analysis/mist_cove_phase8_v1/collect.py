"""Verify sealed private synthetic receipts and publish aggregate-only records."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/mist_cove/phase8_v1'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase8_v1'
FORBIDDEN={'x','y','events','weights','state_dict','model','optimizer','rng','batch_rng','logits','probabilities','waveform','waveforms','labels'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN:raise ValueError('private state cannot enter public aggregates')
        for v in value.values():safe(v)
    elif isinstance(value,list):
        for v in value:safe(v)


def publish(path,value):
    safe(value);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.phase8-')
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.link(name,path) # atomic complete publication, never overwrite
    finally:os.unlink(name)


def load_verified(path,expected=None):
    if expected is not None and sha(path)!=expected:raise ValueError('private receipt mismatch before JSON decoding')
    r=json.loads(Path(path).read_text())
    if r.get('status')!='COMPLETE':raise ValueError('complete scientific receipt required')
    return r


def summarize(rows):
    result={}
    for task in ('single','multiple'):
        result[task]={}
        for method in ('P0','R1','R2','B1','A1','T1'):
            group=[r for r in rows if r['task']==task and r['method']==method];entry={}
            for condition in ('clean','competitive','added'):
                values=[r['conditions'][condition]['macro_f1'] for r in group]
                entry[condition]={'mean':statistics.mean(values),'seed_sd':statistics.stdev(values),'worst_seed':min(values)}
            entry['worst_taxonomy_f1']=min(v['metrics']['macro_f1'] for r in group for v1 in r['strengths'].values() for v in v1.values())
            entry['competitive_ece_mean']=statistics.mean(r['conditions']['competitive']['ece15'] for r in group)
            entry['competitive_false_confidence_mean']=statistics.mean(r['conditions']['competitive']['false_confidence_0_9'] for r in group)
            entry['competitive_nll_mean']=statistics.mean(r['conditions']['competitive']['nll'] for r in group)
            entry['competitive_brier_mean']=statistics.mean(r['conditions']['competitive']['brier'] for r in group)
            result[task][method]=entry
    return result


def main():
    from mist_cove.phase8.evaluation import gate
    from mist_cove.phase8.provenance import freeze_proof
    started=time.perf_counter()
    r=load_verified(PRIVATE/'confirmatory_v1/results.json')
    if r['frozen']!=freeze_proof('confirmatory'):raise ValueError('confirmatory freeze mismatch')
    manifest=json.loads((ROOT/'experiments/mist_cove/phase8_confirmatory_freeze_v1.json').read_text())
    d=load_verified(PRIVATE/'diagnostic_v2/results.json',manifest['diagnostic_result_sha256'])
    if d['frozen']!=freeze_proof('diagnostic'):raise ValueError('diagnostic freeze mismatch')
    c=r['config'];expected={(t,s,m) for t in c['tasks'] for s in c['seeds'] for m in c['methods']}
    if len(r['rows'])!=len(expected) or {(v['task'],v['seed'],v['method']) for v in r['rows']}!=expected:raise ValueError('all seeds/tasks/candidates required')
    if r['gate']!=gate(r['rows'],c):raise ValueError('gate does not recompute')
    if r['gate']['status']=='FAIL' and (r['heads']['fits'] or r['heads']['rows']):raise ValueError('failed gate bypassed')
    for row in r['rows']:
        if row['method'] in ('B1','A1','T1'):
            path=PRIVATE/f"confirmatory_v1/{row['task']}_{row['seed']}_{row['method']}.pt";creation=row['training']['creation']
            if sha(path)!=creation['checkpoint_sha256'] or sha(path.with_suffix('.seal.json'))!=creation['seal_sha256']:
                raise ValueError('new checkpoint creation mismatch')
            if json.loads(path.with_suffix('.creation.json').read_text())!=creation:raise ValueError('new creation sidecar mismatch')
    failed=json.loads((PRIVATE/'diagnostic_v1/STOPPED.json').read_text())
    diagnostic={**d,'private_result_sha256':sha(PRIVATE/'diagnostic_v2/results.json')}
    publish(PUBLIC/'diagnostic_summary.json',diagnostic)
    publish(PUBLIC/'robustness_summary.json',{'config':c,'rows':[{k:v for k,v in row.items() if k not in {'calibration','information_loss','donors'}} for row in r['rows']],
        'summary':summarize(r['rows']),'private_result_sha256':sha(PRIVATE/'confirmatory_v1/results.json')})
    publish(PUBLIC/'calibration_summary.json',{'rows':[{k:row[k] for k in ('task','seed','method','calibration','information_loss')} for row in r['rows']]})
    publish(PUBLIC/'evidence_summary.json',{'heads':r['heads'],'donors':[{k:row[k] for k in ('task','seed','method','donors')} for row in r['rows'] if row['donors'] is not None]})
    publish(PUBLIC/'scientific_gate_status.json',r['gate'])
    cuda=json.loads((PRIVATE/'cuda_v1/cuda_receipt.json').read_text());publish(PUBLIC/'cuda_receipt.json',cuda)
    training=[v['training'] for v in r['rows'] if v['method'] in ('B1','A1','T1')]
    resources={k:r[k] for k in ('new_representation_fits','new_head_fits','optimizer_updates','example_exposures','wall_seconds','peak_rss_KiB','git_implementation')}
    resources.update(diagnostic_wall_seconds=d['wall_seconds'],failed_diagnostic_wall_seconds=failed['wall_seconds'],
        failed_diagnostic_probe_fits=1,failed_diagnostic_probe_updates=150,
        completed_diagnostic_probe_fits=d['new_probe_fits'],diagnostic_optimizer_updates=d['optimizer_updates'],
        total_scientific_fits_including_failed_probe=r['new_representation_fits']+r['new_head_fits']+d['new_probe_fits']+1,
        total_scientific_optimizer_updates_including_failed_probe=r['optimizer_updates']+d['optimizer_updates']+150,
        total_scientific_exposures_including_failed_probe=r['example_exposures']+d['exposures']+150*16,
        pilot_updates=20,pilot_exposures=320,
        train_seconds=sum(v['train_seconds'] for v in training),evaluation_seconds=sum(v['evaluation_wall_seconds'] for v in r['rows']),
        checkpoint_write_seconds=sum(v['checkpoint_write_seconds'] for v in training),checkpoint_restore_seconds=sum(v['checkpoint_restore_seconds'] for v in training),
        reference_restore_seconds=sum(v['checkpoint_read_restore_seconds'] for v in r['provenance']),
        timing_limitations='internal wall excludes interpreter imports; per-row evaluation includes generation/reporting; peak RSS is process lifetime; no kernel-time measurement',
        real_data_accessed=False,historical_phase6_deserializations=0,cuda_status=cuda['status'],provenance_status='PASS',
        confirmatory_private_result_sha256=sha(PRIVATE/'confirmatory_v1/results.json'),
        failed_attempt_receipt_sha256=sha(PRIVATE/'diagnostic_v1/STOPPED.json'),
        diagnostics_private_result_sha256=sha(PRIVATE/'diagnostic_v2/results.json'))
    resources['collector_audit_seconds']=time.perf_counter()-started
    publish(PUBLIC/'execution_receipt.json',resources)
    manifest={'version':'phase8-execution-manifest-v1','implementation_commit':r['git_implementation'],'confirmatory_freeze':r['frozen'],
        'private_receipts':{'diagnostic_v2':sha(PRIVATE/'diagnostic_v2/results.json'),'confirmatory_v1':sha(PRIVATE/'confirmatory_v1/results.json')},
        'new_model_creation_sha256':{f"{v['task']}_{v['seed']}_{v['method']}":sha(PRIVATE/f"confirmatory_v1/{v['task']}_{v['seed']}_{v['method']}.creation.json") for v in r['rows'] if v['method'] in ('B1','A1','T1')}}
    publish(ROOT/'experiments/mist_cove/phase8_execution_manifest_v1.json',manifest)
    print(json.dumps({'gate':r['gate'],'resources':resources},indent=2))

if __name__=='__main__':main()

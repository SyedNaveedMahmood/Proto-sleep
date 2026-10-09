"""Atomic SC manifest transactions and mandatory deterministic real-data pilot."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import resource
import time
import mne
import numpy as np
import pyedflib
from .integrity import annotation_grid
from .preprocessing import (SCHEMA, CHANNEL, GATE_SHA256, SourceProofs, load_sc_scope, validate_scope,
                            build_record, epoch_targets, iter_recording)
from .provenance import AuditRun, atomic_json, atomic_csv, atomic_text, fingerprint, sha256_file

DEFAULT_GATE=Path('reports/data_audit/p1_5/gate_decisions.json')
DEFAULT_PRIOR=Path('reports/data_audit/audit.json')
DEFAULT_RESOLUTION=Path('reports/data_audit/p1_5/revised_audit.json')
DEFAULT_RECEIPT=Path('mist_transfer_runs/p1_5_input_verification.json')


def pilot_selection(rows, seed=123):
    """QA selection only, based on variants and previously audited gap positions."""
    ordered=sorted(rows,key=lambda r:r['recording_id']);selected=[]
    def add(row):
        if row['subject_id'] not in {r['subject_id'] for r in selected}:selected.append(row)
    for variant in sorted({r['recording_variant'] for r in ordered}):
        add(next(r for r in ordered if r['recording_variant']==variant))
    for rid in ['SC4092E0','SC4571F0']:
        add(next(r for r in ordered if r['recording_id']==rid))
    for reason in ['movement','unknown','unannotated']:
        candidates=[r for r in ordered if reason in r['excluded_epoch_ranges']]
        if candidates:add(candidates[0])
    for i in np.random.default_rng(seed).permutation(len(ordered)):
        if len(selected)>=10:break
        add(ordered[int(i)])
    if len(selected)!=10:raise ValueError('ten distinct participant-night QA pairs required')
    return selected


def validate_real_record(record):
    """New-loader differential checks, not a repeat of NPZ waveform alignment.

    Stream every physical epoch once for shape/finiteness. Compare bounded sample
    epochs against independent MNE volts->uV and EDF digital calibration oracle.
    All labels are checked independently against pyEDFlib on the physical grid.
    """
    labels,mask,codes,_,reasons=epoch_targets(record);n=len(labels)
    positions={0,n-1,n//2}
    for s in record['annotation_segments']:
        i=int(s['onset_seconds']//30)
        if 0<i<n:positions.update([i-1,i]);break
    for reason in ['movement','unknown','unannotated']:
        gap=np.flatnonzero(reasons==reason)
        if len(gap):positions.update(int(i) for i in [max(0,gap[0]-1),gap[0],min(n-1,gap[-1]+1)])
    errors={'mne_max_abs_error_uV':0.,'digital_max_abs_error_uV':0.}
    observed=0;checked=[];peak_batch=0
    with pyedflib.EdfReader(record['annotation_source']) as hyp:
        on,dur,text=hyp.readAnnotations()
    reference,_=annotation_grid(on,dur,text,record['signal_contract']['duration_seconds'])
    if not np.array_equal(reference,labels):raise ValueError(record['recording_id']+': independent stage-grid disagreement')
    h=record['signal_contract']['calibration'];scale=(h['physical_max']-h['physical_min'])/(h['digital_max']-h['digital_min'])
    # EDFLib forbids two concurrent readers of one path. Read only bounded digital
    # references first and close that handle before streaming the new loader.
    digital_references={}
    with pyedflib.EdfReader(record['path']) as raw:
        for i in sorted(positions):
            digital=raw.readSignal(record['signal_contract']['channel_index'],start=i*3000,n=3000,digital=True)
            digital_references[i]=(digital.astype(np.float64)-h['digital_min'])*scale+h['physical_min']
    mraw=mne.io.read_raw_edf(record['path'],preload=False,include=[CHANNEL],verbose='ERROR')
    try:
        for batch in iter_recording(record,include_labels=True):
            idx=batch['original_epoch_index'];expected=np.arange(observed,observed+len(idx))
            if not np.array_equal(idx,expected) or batch['x'].shape!=(len(idx),1,3000) or batch['x'].dtype!=np.float32:
                raise ValueError(record['recording_id']+': streamed physical epoch topology/shape/dtype disagreement')
            if not np.array_equal(batch['scoring_mask'],mask[idx]) or not np.array_equal(batch['y'],labels[idx]):
                raise ValueError(record['recording_id']+': chunk label/mask disagreement')
            peak_batch=max(peak_batch,batch['x'].nbytes);observed+=len(idx)
            for j,i in enumerate(idx):
                if int(i) not in positions:continue
                start=int(i)*3000;wave=batch['x'][j,0]
                oracle=digital_references[int(i)]
                mne_uv=mraw.get_data(start=start,stop=start+3000)[0]*1e6
                # Two float32 eps relative error plus 2e-5 uV absolute floor;
                # fixed before data, accommodates the required float32 cast.
                for key,want in [('digital_max_abs_error_uV',oracle),('mne_max_abs_error_uV',mne_uv)]:
                    error=float(np.max(np.abs(wave.astype(np.float64)-want)))
                    errors[key]=max(errors[key],error)
                    if not np.allclose(wave,want,rtol=2*np.finfo(np.float32).eps,atol=2e-5):
                        raise ValueError(record['recording_id']+f': waveform/unit differential failure at epoch {i}: {key}={error}')
                checked.append(int(i))
    finally:mraw.close()
    if observed!=n:raise ValueError(record['recording_id']+': stream omitted physical epochs')
    return {'recording_id':record['recording_id'],'subject_id':record['subject_id'],'variant':record['recording_variant'],
            'status':'PASS','streamed_epochs':observed,'waveform_differential_original_indices':sorted(checked),
            'scored_epochs':int(np.sum(mask)),'unscored_epochs':int(np.sum(~mask)),
            'original_annotation_codes':sorted(set(c for c in codes if c is not None)),
            'unscored_reasons':dict(Counter(reasons[~mask])), 'peak_float32_batch_bytes':peak_batch,
            'label_decoder':'independent pyEDFlib vs strict TAL/MNE on physical grid',**errors}


def build_manifest(output, mode='pilot', seed=123, resume=False, dry_run=False,
                   gate_path=DEFAULT_GATE, prior_path=DEFAULT_PRIOR, resolution_path=DEFAULT_RESOLUTION,
                   receipt_path=DEFAULT_RECEIPT, pilot_dir=None):
    started=time.perf_counter()
    if mode not in {'pilot','complete'}:raise ValueError('mode must be pilot or complete')
    rows,scope=load_sc_scope(gate_path,prior_path,resolution_path)
    selected=pilot_selection(rows,seed)
    chosen=selected if mode=='pilot' else rows
    config={'phase':'P2-SC','schema_version':SCHEMA,'mode':mode,'scope':scope,
            'selection':'all complete physical epochs, before annotation access','channel':CHANNEL,
            'seed':seed,'pilot_recording_ids':[r['recording_id'] for r in selected],
            'split_assignments':None,'normalization':'none; future source-TRAIN only',
            'st_access':False,'shhs_access':False,'waveform_exports':False}
    if dry_run:
        from .provenance import git_info, environment
        print(json.dumps({'config':config,'git':git_info(),'environment':environment(),
            'recordings':len(chosen),'subjects':len({r['subject_id'] for r in chosen}),
            'expected_physical_epochs':sum(r['n_epochs'] for r in chosen),
            'output_dir':str(Path(output).resolve()),'source_data_opened':False},indent=2))
        return 0
    pilot_proof=None
    if mode=='complete':
        if not pilot_dir:raise ValueError('complete manifest requires a passing --pilot-dir')
        pilot_proof=load_complete(pilot_dir)
        p=pilot_proof['validation']
        if p['gate']!='PASS' or p['mode']!='pilot' or p['scope']!=scope or p['pilot_recording_ids']!=config['pilot_recording_ids']:
            raise ValueError('pilot scope/seed/validation mismatch')
        config['pilot_proof_signature']=pilot_proof['provenance']['signature']
    proofs=SourceProofs(receipt_path,scope['historical_signature'])
    # Check only the selected/eligible SC EDF/hyp hashes, never NPZ/ST/SHHS.
    sources={}
    for row in chosen:
        for path_key,hash_key in [('path','source_sha256'),('annotation_source','annotation_sha256')]:
            proofs.verify(row[path_key],row[hash_key]);sources[row[path_key]]=row[hash_key]
    run=AuditRun(output,config,sources,seed,resume)
    if run.complete.exists():
        print('Verified completed atomic run; no source waveform or annotation reload.',flush=True)
        print((run.output/'FINAL_COPY_PASTE.txt').read_text(),flush=True)
        return 0
    if mode=='complete':
        if pilot_proof['provenance']['git']['implementation_hash']!=run.identity['git']['implementation_hash'] or pilot_proof['provenance']['environment']!=run.identity['environment']:
            raise ValueError('pilot implementation/environment changed; revalidate before complete build')
        if pilot_proof['provenance']['source_recording_hashes']!={k:v for k,v in sources.items() if k in pilot_proof['provenance']['source_recording_hashes']}:
            raise ValueError('pilot source proof mismatch')
    records=[];results=[];reused=0
    for row in chosen:
        rid=row['recording_id'];name='records/'+rid+'.json';path=run.output/name
        token=fingerprint({'row':row,'run_signature':run.signature})
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['input_signature']!=token or fingerprint(saved['record'])!=saved['record_sha256']:
                raise ValueError(rid+': interrupted record digest/signature mismatch')
            record=saved['record'];reused+=1
        else:
            record=build_record(row,proofs)
            atomic_json({'input_signature':token,'record_sha256':fingerprint(record),'record':record},path)
        epoch_targets(record)
        if mode=='pilot':results.append(validate_real_record(record))
        records.append(record)
        print(f'{rid}: {record["epoch_grid"]["n_epochs"]} physical epochs; {record["scoring_mask"]["unscored_epochs"]} unscored; PASS',flush=True)
    validate_scope(records,{r['recording_id'] for r in chosen},{r['subject_id'] for r in chosen})
    proofs.postflight()
    totals={'subjects':len({r['subject_id'] for r in records}),'recordings':len(records),
            'physical_epochs':sum(r['epoch_grid']['n_epochs'] for r in records),
            'scored_epochs':sum(r['scoring_mask']['scored_epochs'] for r in records),
            'unscored_epochs':sum(r['scoring_mask']['unscored_epochs'] for r in records),
            'partial_tail_samples':sum(r['epoch_grid']['partial_tail_samples'] for r in records)}
    validation={'gate':'PASS','mode':mode,'schema_version':SCHEMA,'scope':scope,**totals,
        'pilot_recording_ids':config['pilot_recording_ids'],'pilot_results':results if mode=='pilot' else pilot_proof['validation']['pilot_results'],
        'pilot_signature':None if mode=='pilot' else pilot_proof['provenance']['signature'],
        'source_files_verified':len(proofs.checked),'source_files_rehashed':proofs.files_rehashed,
        'cached_waveform_alignment_reused':True,'npz_arrays_opened':0,'st_opened':False,'shhs_opened':False,
        'record_descriptors_resumed':reused,'elapsed_seconds':time.perf_counter()-started,
        'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'split_chosen':False,'fitted_normalization':False,'training_performed':False,
        'p3_data_eligibility':'VERIFIED_SC_ONLY' if mode=='complete' else 'PILOT_ONLY',
        'p3_conditions':['freeze explicit subject-grouped source train/val/reserved-test roles',
                         'fit eventual normalization using source TRAIN only; use scoring masks',
                         'ST/SHHS eligibility remains unchanged; no external transfer claim']}
    manifest={'schema_version':SCHEMA,'mode':mode,'signature':run.signature,'scope':scope,
              'totals':totals,'split_assignments':None,'records':[
                  {'recording_id':r['recording_id'],'subject_id':r['subject_id'],'descriptor':'records/'+r['recording_id']+'.json',
                   'descriptor_sha256':sha256_file(run.output/('records/'+r['recording_id']+'.json'))} for r in records]}
    atomic_json(manifest,run.output/'manifest.json');atomic_json(validation,run.output/'validation.json')
    atomic_csv([{'recording_id':r['recording_id'],'subject_id':r['subject_id'],'night':r['night'],
        'variant':r['recording_variant'],'physical_epochs':r['epoch_grid']['n_epochs'],
        'scored_epochs':r['scoring_mask']['scored_epochs'],'unscored_epochs':r['scoring_mask']['unscored_epochs'],
        'source_sha256':r['source_sha256'],'annotation_sha256':r['annotation_sha256']} for r in records],
        ['recording_id','subject_id','night','variant','physical_epochs','scored_epochs','unscored_epochs','source_sha256','annotation_sha256'],run.output/'recording_summary.csv')
    text='\n'.join(['MIST-Transfer v3 P2-SC; STOP BEFORE P3',f'Code SHA: {run.identity["git"]["sha"]}',
        f'Mode: {mode}; PASS; '+str(totals),
        f'Independent real pilot: {len(validation["pilot_results"])} distinct subjects; E/F/G and audited gaps.',
        'Fpz-Cz; native 100 Hz/uV; output float32 [N,1,3000]; no filters/resampling/normalization.',
        'Every full physical epoch retained; movement/unknown/missing labels are unscored, available for inference.',
        'Original indices/time, raw codes, calibration, source/annotation hashes and record boundaries preserved.',
        f'SC source files verified: {len(proofs.checked)}; rehashed: {proofs.files_rehashed}; NPZ alignment repeated: 0.',
        f'Elapsed seconds: {validation["elapsed_seconds"]:.3f}; process peak RSS KiB: {validation["peak_process_rss_kib"]}.',
        f'P3 data eligibility: {validation["p3_data_eligibility"]}; explicit subject roles and TRAIN-only normalization required.',
        'No final split chosen, model trained, ST/SHHS processed or waveform arrays exported.',
        'Remaining limitations: quarantined SC sources; unknown NPZ producer; SC/ST linkage; unconfirmed SHHS access.',
        'NEXT SINGLE SAFE COMMAND: git status --short --branch'])+'\n'
    atomic_text(text,run.output/'FINAL_COPY_PASTE.txt')
    run.finish(['manifest.json','validation.json','recording_summary.csv','FINAL_COPY_PASTE.txt']+
               ['records/'+r['recording_id']+'.json' for r in records],'PASS_SC_ONLY')
    print(text,flush=True)
    return 0


def load_complete(output):
    output=Path(output); marker=json.loads((output/'COMPLETE.json').read_text())
    for name,digest in marker['outputs'].items():
        if sha256_file(output/name)!=digest:raise ValueError('completed artifact digest mismatch: '+name)
    manifest=json.loads((output/'manifest.json').read_text());provenance=json.loads((output/'provenance.json').read_text())
    validation=json.loads((output/'validation.json').read_text())
    if marker['gate']!='PASS_SC_ONLY' or validation['gate']!='PASS':raise ValueError('completion gate failed')
    if marker['signature']!=manifest['signature'] or marker['signature']!=provenance['signature']:
        raise ValueError('completed manifest/provenance signature mismatch')
    if manifest['schema_version']!=SCHEMA or manifest['mode']!=validation['mode']:
        raise ValueError('manifest version/mode mismatch')
    records=[]
    for r in manifest['records']:
        path=output/r['descriptor']
        if sha256_file(path)!=r['descriptor_sha256']:raise ValueError('record descriptor digest mismatch')
        saved=json.loads(path.read_text());record=saved['record']
        if fingerprint(record)!=saved['record_sha256'] or r['recording_id']!=record['recording_id'] or r['subject_id']!=record['subject_id']:
            raise ValueError('record descriptor content/identity mismatch')
        records.append(record)
    if sha256_file(DEFAULT_GATE)!=GATE_SHA256:raise ValueError('reviewed eligibility artifact changed')
    reviewed,scope=load_sc_scope(DEFAULT_GATE,DEFAULT_PRIOR,DEFAULT_RESOLUTION)
    allowed={r['recording_id'] for r in reviewed}
    originals={r['recording_id']:r for r in reviewed}
    if manifest['scope']!=scope:raise ValueError('manifest scope differs from reviewed proof')
    for record in records:
        original=originals.get(record['recording_id'])
        if original is None or any(record[k]!=original[k] for k in ['path','source_sha256','annotation_source','annotation_sha256','subject_id','night']):
            raise ValueError(record['recording_id']+': descriptor source differs from reviewed proof')
    if {r['recording_id'] for r in records}-allowed:raise ValueError('manifest includes unreviewed recording')
    validate_scope(records,{r['recording_id'] for r in records},{r['subject_id'] for r in records})
    if manifest['mode']=='complete' and (len(records)!=146 or len({r['subject_id'] for r in records})!=74 or {r['recording_id'] for r in records}!=allowed):
        raise ValueError('complete manifest does not cover reviewed scope')
    return {'manifest':manifest,'validation':validation,'provenance':provenance,'records':records}

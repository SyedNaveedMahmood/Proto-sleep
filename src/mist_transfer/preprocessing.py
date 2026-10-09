"""Versioned SC-only physical epoch grids and lazy calibrated EDF access.

No filtering, resampling, fitted normalization, waveform export or split choice.
The reviewed P1.5 artifacts freeze the permitted person/recording scope.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pyedflib
from .annotation_compat import fixed_header, paired_annotations, read_annotation_profile
from .integrity import STAGE_MAP, STAGES, annotation_grid, calibration, identity, validate_splits
from .provenance import sha256_file

SCHEMA = 'mist-transfer-sc-epochs-v1'
CHANNEL = 'EEG Fpz-Cz'
GATE_SHA256 = '71773546eca2c8e6cc88e6a91cd5a703861c0e4f746ee1a5731b58263948912b'
RESOLUTION_SHA256 = 'e7e81576802a0ba6fe959842ccaeba8251ba6e024f2e4c7346be63b3785a96b8'
EXCLUDED = {'SC:06','SC:23','SC:36','SC:74'}


def load_sc_scope(gate_path, prior_path, resolution_path):
    """Validate reviewed, hash-bound artifacts without opening ST/SHHS/NPZ data."""
    gate_path,prior_path,resolution_path = map(Path,(gate_path,prior_path,resolution_path))
    if sha256_file(gate_path)!=GATE_SHA256 or sha256_file(resolution_path)!=RESOLUTION_SHA256:
        raise ValueError('reviewed P1.5 eligibility/resolution artifact changed')
    gates=json.loads(gate_path.read_text()); prior=json.loads(prior_path.read_text())
    resolution=json.loads(resolution_path.read_text())
    if sha256_file(prior_path)!=resolution['historical_audit_sha256']:
        raise ValueError('historical P1 proof artifact changed')
    gate=gates['sc_within_cohort']
    if gate['p2_eligibility']!='VERIFIED' or set(gate['quarantined_subjects'])!=EXCLUDED:
        raise ValueError('SC eligibility gate is not the reviewed scope')
    allowed=set(gate['eligible_recordings']); people=set(gate['eligible_subjects'])
    if len(allowed)!=146 or len(people)!=74 or people & EXCLUDED:
        raise ValueError('frozen SC scope counts/exclusions disagree')
    rows=[r for r in prior['records'] if r['dataset']=='raw_sc' and r['recording_id'] in allowed]
    validate_scope(rows,allowed,people)
    proofs={r['recording_id']:r for r in resolution['annotation_resolutions'] if r['study']=='SC'}
    for row in rows:
        p=proofs[row['recording_id']]
        if p['status']!='VERIFIED' or p['proposed_scope_eligibility']!='VERIFIED':
            raise ValueError(row['recording_id']+': P1.5 resolution is ineligible')
        for key in ['path','source_sha256','annotation_source','annotation_sha256','subject_id','night']:
            if p[key]!=row[key]:raise ValueError(row['recording_id']+': inconsistent audit '+key)
    return sorted(rows,key=lambda r:r['recording_id']), {
        'gate_sha256':GATE_SHA256,'resolution_sha256':RESOLUTION_SHA256,
        'historical_audit_sha256':resolution['historical_audit_sha256'],
        'historical_signature':prior['signature'],'reviewed_commit':'19351edfd79807ba1716189bd3cc9671f2c97499'}


def validate_scope(rows, allowed, people):
    if {r['recording_id'] for r in rows}!=set(allowed) or {r['subject_id'] for r in rows}!=set(people):
        raise ValueError('recording/person identities disagree with frozen eligibility')
    seen={k:set() for k in ['recording_id','path','source_sha256']}
    nights=set()
    for row in rows:
        ident=identity(Path(row['path']).name,'psg')
        if ident['study']!='SC' or row['subject_id'] in EXCLUDED or row.get('split') is not None:
            raise ValueError(row['recording_id']+': excluded person, non-SC study or preassigned split')
        if any(row[k]!=ident[k] for k in ['study','subject_id','night','recording_id']) or row.get('recording_variant',ident['recording_variant'])!=ident['recording_variant']:
            raise ValueError(row['recording_id']+': corrupted canonical metadata')
        if identity(Path(row['annotation_source']).name,'hyp')['recording_id']!=row['recording_id']:
            raise ValueError(row['recording_id']+': mismatched annotation identity')
        night=(row['subject_id'],row['night'])
        if night in nights:raise ValueError(row['recording_id']+': duplicate person/night')
        nights.add(night)
        for key,values in seen.items():
            value=str(Path(row[key]).resolve()) if key=='path' else row[key]
            if not value or value in values:raise ValueError(row['recording_id']+': duplicate/missing '+key)
            values.add(value)


class SourceProofs:
    """Reuse local P1.5 stat/hash receipts; changed inputs require digest validation.

    This cache is trusted local evidence, not an adversarial tamper-proof store.
    Only explicitly supplied SC paths are accessed. No dataset discovery occurs.
    """
    def __init__(self, receipt=None, prior_signature=None):
        saved=json.loads(Path(receipt).read_text()) if receipt and Path(receipt).exists() else {}
        self.cache=saved.get('files',{}) if saved.get('prior_signature')==prior_signature else {}
        self.checked={}; self.files_rehashed=0

    def verify(self,path,digest):
        path=Path(path); name=str(path.resolve()); s=path.stat()
        stamp=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
        proof=self.checked.get(name,self.cache.get(name,{}))
        if proof.get('stat')!=stamp or proof.get('sha256')!=digest:
            actual=sha256_file(path);self.files_rehashed+=1
            if actual!=digest:raise ValueError(path.name+': input SHA256 changed; P1 proof invalid')
        proof={'stat':stamp,'sha256':digest};self.checked[name]=proof
        return proof

    def postflight(self):
        for name,proof in list(self.checked.items()):
            if self.verify(name,proof['sha256'])!=proof:
                raise ValueError(Path(name).name+': source changed during execution')


def physical_grid(n_samples, fs=100):
    """Geometry only. This function cannot consult annotations or stage labels."""
    if fs!=100 or int(n_samples)!=n_samples or n_samples<3000:
        raise ValueError('physical grid requires measured 100 Hz and at least one full epoch')
    n=int(n_samples)//3000
    return {'n_epochs':n,'n_samples_native':int(n_samples),'samples_per_epoch':3000,
            'original_epoch_index':{'start':0,'stop':n,'step':1},
            'start_seconds':{'start':0,'step':30},'start_sample':{'start':0,'step':3000},
            'partial_tail_samples':int(n_samples)-n*3000,
            'partial_tail_seconds':(int(n_samples)-n*3000)/fs,
            'selection':'complete physical epochs, independent of labels',
            'physical_continuity':'validated contiguous EDF; no joins across recordings'}


def signal_contract(row):
    """Verify file extent, raw header and chosen channel before constructing grid."""
    path=Path(row['path']);h,start=fixed_header(path)
    ns=int(h[252:256]); nrecords=int(h[236:244]); duration=float(h[244:252]); size=int(h[184:192])
    if ns<=0 or nrecords<=0 or not np.isfinite(duration) or duration<=0 or size!=256*(ns+1):
        raise ValueError(path.name+': invalid EDF signal geometry')
    if h[192:236].strip() not in {b'',b'EDF+C'}:
        raise ValueError(path.name+': discontinuous/unsupported EDF; do not infer continuity')
    with path.open('rb') as f:f.seek(256);sh=f.read(ns*256)
    if len(sh)!=ns*256 or any(x<32 or x>126 for x in sh):
        raise ValueError(path.name+': truncated/nonprintable signal header')
    counts=[int(sh[216*ns+8*i:216*ns+8*(i+1)]) for i in range(ns)]
    if min(counts)<=0 or path.stat().st_size!=size+2*nrecords*sum(counts):
        raise ValueError(path.name+': truncated or extra EDF signal bytes')
    with pyedflib.EdfReader(str(path)) as reader:
        headers=reader.getSignalHeaders(); names=[c['label'] for c in headers]
        if names.count(CHANNEL)!=1:raise ValueError(path.name+': Fpz-Cz missing/ambiguous')
        channel_index=names.index(CHANNEL);channel=headers[channel_index];calibration(channel)
        if channel['dimension']!='uV' or channel['sample_frequency']!=100:
            raise ValueError(path.name+': measured Fpz-Cz must be 100 Hz in uV')
        expected=[c for c in row['channels'] if c['label']==CHANNEL]
        if len(expected)!=1 or channel!=expected[0]:
            raise ValueError(path.name+': channel/calibration metadata differs from audited source')
        samples=int(reader.getNSamples()[channel_index])
        if samples!=nrecords*counts[channel_index] or samples/100!=nrecords*duration:
            raise ValueError(path.name+': measured EEG samples and physical duration disagree')
        if reader.file_duration!=row['duration_seconds'] or reader.getStartdatetime().isoformat()!=row['start_datetime'] or reader.getStartdatetime()!=start:
            raise ValueError(path.name+': audited clock/duration differs')
    grid=physical_grid(samples)
    if grid['n_epochs']!=row['n_epochs']:
        raise ValueError(path.name+': audited epoch count differs from physical grid')
    return grid, {'channel':CHANNEL,'montage':'Fpz-Cz','reference':'Cz',
        'channel_index':channel_index,'fs_native':100,'fs_output':100,'raw_units':'uV',
        'output_units':'uV','amplitude_conversion':1.0,'calibration':channel,
        'normalization':'none; eventual fitted statistics source TRAIN only',
        'filtering':'none added; native acquisition prefilter retained','resampling':'none',
        'start_datetime':start.isoformat(),'timezone':'UNKNOWN; preserve native EDF local clock',
        'duration_seconds':samples/100,'edf_header_bytes':size,'data_record_count':nrecords,
        'data_record_duration_seconds':duration}


def annotation_segments(path):
    _,_,(onsets,durations,descriptions),_=read_annotation_profile(path)
    return [{'onset_seconds':float(on),'duration_seconds':float(dur),
             'original_description':str(text),
             'original_code':'M' if text=='Movement time' else str(text).rsplit(' ',1)[-1]}
            for on,dur,text in zip(onsets,durations,descriptions)]


def build_record(row,proofs):
    rid=row['recording_id']
    try:
        if row['study']!='SC' or row['subject_id'] in EXCLUDED:raise ValueError('ineligible SC identity')
        proofs.verify(row['path'],row['source_sha256'])
        grid,contract=signal_contract(row)  # MUST precede annotation parsing.
        proofs.verify(row['annotation_source'],row['annotation_sha256'])
        labels,detail=paired_annotations(row['annotation_source'],row['path'],contract['duration_seconds'])
        if len(labels)!=grid['n_epochs'] or {name:int(np.sum(labels==i)) for i,name in enumerate(STAGES)}!=row['mapped_stage_counts']:
            raise ValueError('annotation grid differs from geometry/historical stage counts')
        return {**{k:row[k] for k in ['study','subject_id','night','recording_id','recording_variant','path','source_sha256',
                    'annotation_source','annotation_sha256','signal_sha256','shape_sha256']},
                'schema_version':SCHEMA,'split':None,'signal_contract':contract,'epoch_grid':grid,
                'annotation_segments':annotation_segments(row['annotation_source']),
                'scoring_mask':{'scored_epochs':int(np.sum(labels>=0)),'unscored_epochs':int(np.sum(labels<0)),
                                'unscored_original_epoch_runs':detail['excluded_epoch_ranges']},
                'annotation_validation':detail,'source_stat_proof':proofs.checked[str(Path(row['path']).resolve())],
                'annotation_stat_proof':proofs.checked[str(Path(row['annotation_source']).resolve())]}
    except (ValueError,OSError,KeyError) as exc:
        raise ValueError(rid+': '+str(exc)) from exc


def epoch_targets(record):
    """Attach scoring metadata on the fixed physical grid; never select inputs."""
    segments=record['annotation_segments'];n=record['epoch_grid']['n_epochs']
    labels,detail=annotation_grid([s['onset_seconds'] for s in segments],
        [s['duration_seconds'] for s in segments],[s['original_description'] for s in segments],
        record['signal_contract']['duration_seconds'])
    if len(labels)!=n:raise ValueError(record['recording_id']+': target/grid length disagreement')
    codes=np.full(n,None,dtype=object);desc=np.full(n,None,dtype=object);reason=np.full(n,'unannotated',dtype=object)
    for s in segments:
        expected_code='M' if s['original_description']=='Movement time' else s['original_description'].rsplit(' ',1)[-1]
        if s['original_code']!=expected_code:raise ValueError(record['recording_id']+': corrupted original annotation code')
        lo=min(n,int(round(s['onset_seconds']/30)));hi=min(n,int(round((s['onset_seconds']+s['duration_seconds'])/30)))
        codes[lo:hi]=s['original_code'];desc[lo:hi]=s['original_description']
        reason[lo:hi]='' if STAGE_MAP[s['original_description']]>=0 else 'movement' if s['original_code']=='M' else 'unknown'
    if record['scoring_mask']['unscored_original_epoch_runs']!=detail['excluded_epoch_ranges'] or record['scoring_mask']['scored_epochs']!=int(np.sum(labels>=0)):
        raise ValueError(record['recording_id']+': corrupted scoring metadata')
    return labels,labels>=0,codes,desc,reason


def iter_recording(record, chunk_epochs=16, include_labels=False):
    """Bounded CPU batches [N,1,3000], all physical positions including unscored.

    Annotation files are not opened by inference. Every batch carries a normalized
    join to immutable recording calibration/source metadata plus per-epoch time.
    """
    if isinstance(chunk_epochs,bool) or int(chunk_epochs)!=chunk_epochs or chunk_epochs<=0:
        raise ValueError('positive integer chunk_epochs required')
    if record['schema_version']!=SCHEMA or record['study']!='SC' or record['subject_id'] in EXCLUDED or record.get('split')=='test':
        raise ValueError('unsupported/ineligible record')
    proof=SourceProofs();proof.checked[str(Path(record['path']).resolve())]=record['source_stat_proof']
    proof.verify(record['path'],record['source_sha256'])
    grid,contract=signal_contract({**record,'channels':[record['signal_contract']['calibration']],
        'start_datetime':record['signal_contract']['start_datetime'],'duration_seconds':record['signal_contract']['duration_seconds'],
        'n_epochs':record['epoch_grid']['n_epochs']})
    if grid!=record['epoch_grid'] or contract!=record['signal_contract']:
        raise ValueError(record['recording_id']+': corrupt physical grid/signal contract')
    if include_labels:
        proof.checked[str(Path(record['annotation_source']).resolve())]=record['annotation_stat_proof']
        proof.verify(record['annotation_source'],record['annotation_sha256'])
    targets=epoch_targets(record) if include_labels else None
    with pyedflib.EdfReader(record['path']) as reader:
        for lo in range(0,grid['n_epochs'],int(chunk_epochs)):
            hi=min(grid['n_epochs'],lo+int(chunk_epochs)); count=(hi-lo)*3000
            wave=reader.readSignal(contract['channel_index'],start=lo*3000,n=count)
            if len(wave)!=count or not np.isfinite(wave).all():
                raise ValueError(record['recording_id']+': short/nonfinite calibrated waveform read')
            indices=np.arange(lo,hi,dtype=np.int64)
            x=wave.astype(np.float32).reshape(hi-lo,1,3000)
            if not np.isfinite(x).all():raise ValueError(record['recording_id']+': nonfinite float32 waveform')
            batch={'x':x,
                'original_epoch_index':indices,'start_seconds':indices*30,'start_sample':indices*3000,
                'physical_start_time':[ (dt.datetime.fromisoformat(contract['start_datetime'])+dt.timedelta(seconds=int(i)*30)).isoformat() for i in indices],
                'recording':record}
            if targets:
                batch.update(zip(['y','scoring_mask','original_annotation_code','original_annotation_description','unscored_reason'],[v[lo:hi] for v in targets]))
            yield batch
    proof.postflight()


def assign_subject_roles(records, roles):
    """Future interface only; caller supplies roles, P2 chooses none."""
    if set(roles)!={r['subject_id'] for r in records} or set(roles.values())-{'train','val','test'}:
        raise ValueError('explicit complete SC person-to-role map required')
    if any(r.get('split') is not None and r['split']!=roles[r['subject_id']] for r in records):
        raise ValueError('existing recording role cannot be reassigned')
    validate_scope([{**r,'split':None} for r in records],{r['recording_id'] for r in records},set(roles))
    assigned=[{**r,'split':roles[r['subject_id']]} for r in records]
    validate_splits(assigned)
    return assigned


def iter_development_split(records,roles,split,chunk_epochs=16):
    if split not in {'train','val'}:raise ValueError('reserved test opening prohibited')
    assigned=assign_subject_roles(records,roles)
    for record in assigned:
        if record['split']==split:
            yield from iter_recording(record,chunk_epochs,include_labels=True)

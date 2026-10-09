"""Frozen label-independent SC roles and fail-closed development access."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .integrity import validate_splits, identity
from .preprocessing import GATE_SHA256, SCHEMA, EXCLUDED
from .provenance import atomic_json, fingerprint, sha256_file, git_info

SPLIT_VERSION='mist-transfer-sc-source-split-v1'
P2_COMPLETION='3088497baa4f769986245206f932797d4b18280b313ef12481015f39fa3beab6'
DEFAULT_CANONICAL=Path('mist_transfer_runs/p2_sc/v1/canonical')
DEFAULT_SPLIT=Path('mist_transfer_runs/p3a/frozen_source_split.json')
IDENTITY_KEYS=('study','subject_id','night','recording_id','recording_variant','path','source_sha256',
               'annotation_source','annotation_sha256','signal_sha256','shape_sha256')


def descriptor_identity(path):
    """Read only the scalar identity prefix; never read the annotation section.

    P2 v1 writes these fields before schema_version/signal_contract/annotations.
    Unbuffered byte reads avoid fetching a later annotation block incidentally.
    The whole descriptor hash is already frozen by the reviewed P2 completion.
    """
    found={};line=bytearray();consumed=0
    with Path(path).open('rb',buffering=0) as f:
        while consumed<16384:
            b=f.read(1);consumed+=len(b)
            if not b:break
            line.extend(b)
            if b!=b'\n':continue
            text=line.decode('utf-8').strip();line.clear()
            if text.startswith('"schema_version":'):break
            if not text.startswith('"') or ':' not in text:continue
            key,value=text.split(':',1);key=json.loads(key)
            if key in IDENTITY_KEYS:
                if key in found:raise ValueError('duplicate descriptor identity field')
                found[key]=json.loads(value.rstrip(','))
    if set(found)!=set(IDENTITY_KEYS):raise ValueError('P2 identity prefix incomplete; do not read annotations')
    return found


def split_policy(records, extension_subjects, seed=123):
    """19 historical people -> 15 TRAIN/4 VAL; 55 extension -> reserved TEST."""
    people={r['subject_id'] for r in records};extension=set(extension_subjects)
    development=sorted(people-extension)
    if len(people)!=74 or len(records)!=146 or len(extension)!=55 or len(development)!=19 or people&EXCLUDED:
        raise ValueError('reviewed 74/146 historical/extension strata disagree')
    perm=np.random.default_rng(seed).permutation(len(development))
    validation={development[int(i)] for i in perm[:4]}
    roles={sid:'test' if sid in extension else 'val' if sid in validation else 'train' for sid in sorted(people)}
    assigned=[{**r,'split':roles[r['subject_id']]} for r in records]
    validate_role_records(assigned)
    return {'schema_version':SPLIT_VERSION,'seed':seed,'policy':{
        'historical_development_subjects':development,'reserved_extension_subjects':sorted(extension),
        'validation_count':4,'validation_rule':'first four indices from NumPy default_rng(seed) permutation of sorted historical people',
        'stratification':'identity/history only; no stage labels, age/sex, waveform values or scores',
        'endpoint':'prospectively reserved within-SC subject generalization; not independent external transfer',
        'test_access':'no development waveforms, annotations, scaling, anchors or model selection'},
        'subject_roles':roles,'records':assigned}


def validate_role_records(records):
    seen={k:set() for k in ['recording_id','path','source_sha256','signal_sha256','shape_sha256']}
    nights=set()
    for r in records:
        if r['study']!='SC' or r['subject_id'] in EXCLUDED:raise ValueError('ineligible source role')
        parsed=identity(Path(r['path']).name,'psg')
        if any(r[k]!=parsed[k] for k in ['study','subject_id','night','recording_id','recording_variant']):raise ValueError('canonical subject/night identity disagreement')
        night=(r['subject_id'],r['night'])
        if night in nights:raise ValueError('duplicate subject/night source')
        nights.add(night)
        for key,values in seen.items():
            if not r.get(key) or r[key] in values:raise ValueError('duplicate/missing waveform source identity: '+key)
            values.add(r[key])
    validate_splits(records)


def freeze_source_split(canonical=DEFAULT_CANONICAL, output=DEFAULT_SPLIT, seed=123, resume=False):
    canonical=Path(canonical).resolve();output=Path(output)
    if sha256_file(canonical/'COMPLETE.json')!=P2_COMPLETION:raise ValueError('reviewed P2 completion changed')
    done=json.loads((canonical/'COMPLETE.json').read_text())
    if sha256_file(canonical/'manifest.json')!=done['outputs']['manifest.json']:raise ValueError('P2 manifest digest changed')
    manifest=json.loads((canonical/'manifest.json').read_text())
    if manifest['schema_version']!=SCHEMA or manifest['mode']!='complete':raise ValueError('complete canonical P2 required')
    gates=Path('reports/data_audit/p1_5/gate_decisions.json')
    if sha256_file(gates)!=GATE_SHA256:raise ValueError('reviewed gate changed')
    gate=json.loads(gates.read_text())['sc_within_cohort'];records=[]
    for ref in manifest['records']:
        if done['outputs'].get(ref['descriptor'])!=ref['descriptor_sha256']:raise ValueError('descriptor completion proof disagreement')
        stat=(canonical/ref['descriptor']).stat();completed=(canonical/'COMPLETE.json').stat()
        if max(stat.st_mtime_ns,stat.st_ctime_ns)>max(completed.st_mtime_ns,completed.st_ctime_ns):
            raise ValueError(ref['recording_id']+': descriptor changed since trusted local P2 completion')
        row=descriptor_identity(canonical/ref['descriptor'])
        if row['recording_id']!=ref['recording_id'] or row['subject_id']!=ref['subject_id']:
            raise ValueError('descriptor/P2 identity disagreement')
        records.append({**row,'descriptor':str(canonical/ref['descriptor']),
                        'descriptor_sha256':ref['descriptor_sha256']})
    if {r['recording_id'] for r in records}!=set(gate['eligible_recordings']):raise ValueError('frozen eligibility differs')
    frozen=split_policy(records,gate['eligible_extension_subjects'],seed)
    frozen.update(canonical_signature=manifest['signature'],p2_completion_sha256=P2_COMPLETION,
                  canonical_manifest_sha256=sha256_file(canonical/'manifest.json'))
    frozen['split_fingerprint']=fingerprint(frozen)
    if output.exists():
        if not resume:raise FileExistsError('frozen split exists; use --resume for verification only')
        previous=read_frozen_split(output)
        if previous!=frozen:raise ValueError('frozen split modification refused on resume')
    else:
        output.parent.mkdir(parents=True,exist_ok=True);atomic_json(frozen,output)
        atomic_json({'split_fingerprint':frozen['split_fingerprint'],'file_sha256':sha256_file(output),
            'git':git_info(),'frozen_before_scores':True,'reserved_labels_read':False,
            'role_subject_counts':dict(Counter(frozen['subject_roles'].values())),
            'role_recording_counts':dict(Counter(r['split'] for r in records_for(frozen)))},output.with_suffix('.receipt.json'))
    print('Frozen split: '+frozen['split_fingerprint'],flush=True)
    print('Subjects: '+str(dict(Counter(frozen['subject_roles'].values()))),flush=True)
    print('Recordings: '+str(dict(Counter(r['split'] for r in frozen['records']))),flush=True)
    return frozen


def records_for(frozen):return frozen['records']


def read_frozen_split(path=DEFAULT_SPLIT):
    path=Path(path);frozen=json.loads(path.read_text());receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    payload={k:v for k,v in frozen.items() if k!='split_fingerprint'}
    if fingerprint(payload)!=frozen['split_fingerprint'] or receipt['split_fingerprint']!=frozen['split_fingerprint'] or sha256_file(path)!=receipt['file_sha256']:
        raise ValueError('frozen split fingerprint/digest changed')
    if frozen['schema_version']!=SPLIT_VERSION:raise ValueError('unsupported split schema')
    validate_role_records(frozen['records'])
    if any(r['split']!=frozen['subject_roles'][r['subject_id']] for r in frozen['records']):
        raise ValueError('frozen person/record role disagreement')
    return frozen


def load_development(frozen,role):
    """Role rejection precedes any file opening, including descriptor annotations."""
    if role not in {'train','val'}:raise ValueError('reserved test data/annotation access prohibited')
    validate_role_records(frozen['records']);result=[]
    for ref in frozen['records']:
        if ref['split']!=role:continue
        path=Path(ref['descriptor'])
        if sha256_file(path)!=ref['descriptor_sha256']:raise ValueError('development descriptor digest changed')
        saved=json.loads(path.read_text());record=saved['record']
        if fingerprint(record)!=saved['record_sha256'] or any(record[k]!=ref[k] for k in IDENTITY_KEYS):
            raise ValueError('development source identity/provenance mismatch')
        result.append({**record,'split':role})
    if not result:raise ValueError('empty source development role')
    return result

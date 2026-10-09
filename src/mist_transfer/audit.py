"""Read-only Sleep-EDF P1 inventory. No preprocessing output or model code."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pyedflib
import yaml
from .integrity import (STAGES, identity, annotation_grid, calibration, align_epochs,
                        index_runs, validate_indices, validate_splits, shape_features, shape_key)
from .provenance import (AuditRun, atomic_json, atomic_csv, atomic_text, sha256_file, git_info, environment, fingerprint)

PATH_KEYS = ['sleep_edf_20_npz', 'sleep_edf_78_npz', 'sleep_edfx_original_root',
             'sleep_edfx_sc_raw', 'sleep_edfx_st_raw', 'shhs_nsrr_root_if_approved']


def load_config(path, overrides=()):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('config must be a mapping')
    for key in PATH_KEYS:
        if os.environ.get('MIST_'+key.upper()):
            value[key] = os.environ['MIST_'+key.upper()]
    for override in overrides:
        key, sep, val = override.partition('=')
        if not sep or key not in PATH_KEYS:
            raise ValueError('path override must be a recognized KEY=PATH')
        value[key] = val
    missing = set(PATH_KEYS)-value.keys()
    if missing:
        raise ValueError(f'missing path configuration: {sorted(missing)}')
    return value


def discover(config):
    files, roots = {}, {}
    for key in PATH_KEYS:
        if key == 'shhs_nsrr_root_if_approved':
            # P1 implements no SHHS adapter or access; do not enumerate restricted files.
            roots[key] = {'path': config[key], 'status': 'disabled; authorization and usable data unconfirmed'}
            continue
        p = Path(config[key]).expanduser().resolve()
        roots[key] = {'path': str(p), 'exists': p.is_dir(), 'readable': os.access(p, os.R_OK)}
        if not p.is_dir():
            continue
        if 'npz' in key:
            selected = list(p.glob('*.npz'))+list(p.glob('*PSG.edf'))+list(p.glob('*Hypnogram.edf'))
            if (p/'MANIFEST.TXT').is_file():
                selected.append(p/'MANIFEST.TXT')
        elif key.endswith('_raw'):
            selected = list(p.glob('*.edf'))
        else:
            selected = [p/name for name in ['RECORDS','SHA256SUMS.txt','SC-subjects.xls','ST-subjects.xls'] if (p/name).is_file()]
        for f in selected:
            files[str(f)] = f
    return roots, files


def npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as z:
        members = z.namelist()
        if len(members) != len(set(members)):
            raise ValueError('duplicate NPZ keys')
        for name in members:
            if not name.endswith('.npy') or '/' in name:
                raise ValueError('unexpected NPZ member')
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1,0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2,0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f'unsupported NPY header version {version}')
            result[name[:-4]] = {'shape': list(shape), 'dtype': str(dtype),
                                'object_metadata_not_opened': bool(dtype.hasobject)}
    return result


def read_npz(path):
    headers = npz_headers(path)
    with np.load(path, allow_pickle=False) as z:
        if not {'x','y','fs','ch_label'}.issubset(z.files):
            raise ValueError('NPZ missing x/y/fs/channel metadata')
        x, y = z['x'], z['y']
        if x.ndim == 3 and x.shape[1:] == (3000,1):
            x = x[:,:,0]
        elif x.ndim == 3 and x.shape[1:] == (1,3000):
            x = x[:,0,:]
        if x.ndim != 2 or x.shape[1] != 3000 or y.ndim != 1 or len(x) != len(y) or not len(x):
            raise ValueError('NPZ signal/annotation length or shape disagreement')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite NPZ signal/labels')
        if not np.equal(y, y.astype(np.int64)).all() or not np.isin(y,range(5)).all():
            raise ValueError('NPZ stage coding outside integer 0..4')
        fs = np.asarray(z['fs'])
        if fs.size != 1 or float(fs.item()) != 100:
            raise ValueError('NPZ sampling-rate mismatch: output contract requires explicit 100 Hz')
        ch = np.asarray(z['ch_label'])
        if ch.size != 1 or ch.dtype.kind not in {'U','S'}:
            raise ValueError('ambiguous channel label')
        channel = str(ch.item()) if ch.dtype.kind == 'U' else ch.item().decode()
        present = [k for k in ('epoch_indices','epoch_index','original_epoch_index') if k in z.files]
        if len(present) > 1:
            raise ValueError('conflicting epoch index metadata')
        idx = validate_indices(z[present[0]],len(x)) if present else None
        units = None
        for k in ('units','raw_units'):
            if k in z.files:
                a = z[k]
                if a.size != 1 or a.dtype.kind != 'U':
                    raise ValueError('invalid NPZ units metadata')
                if units is not None and units != str(a.item()):
                    raise ValueError('conflicting NPZ unit metadata')
                units = str(a.item())
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()
    features = shape_features(x)
    sketch = [shape_key(f) for f in features]
    return x, y.astype(np.int64), idx, {'npz_keys': list(headers), 'npz_headers': headers,
                'fs': float(fs.item()), 'channel': channel, 'declared_units': units,
                'signal_sha256': digest, 'shape_sha256': hashlib.sha256(''.join(sketch).encode()).hexdigest(),
                'epoch_shape_hashes': sketch, 'n_epochs': len(y)}


def treatment_metadata(root):
    p = root/'ST-subjects.xls'
    if not p.is_file():
        return {}, 'ST treatment metadata missing'
    table = pd.read_excel(p,header=None)
    if table.shape[1] != 7 or str(table.iloc[0,3]) != 'Placebo night' or str(table.iloc[0,5]) != 'Temazepam night':
        raise ValueError('unrecognized ST treatment metadata layout')
    treatment = {}
    for row in table.iloc[2:].itertuples(index=False, name=None):
        sid, placebo, drug = int(row[0]), int(row[3]), int(row[5])
        if {placebo,drug} != {1,2}:
            raise ValueError('invalid ST treatment night mapping')
        for night, condition in [(placebo,'placebo'),(drug,'temazepam')]:
            key = (f'ST:{sid:02d}',night)
            if key in treatment:
                raise ValueError('duplicate ST participant/night metadata')
            treatment[key] = condition
    return treatment, None


def sc_metadata(root):
    p = root/'SC-subjects.xls'
    if not p.is_file():
        return set(), 'SC participant metadata missing'
    table = pd.read_excel(p)
    if not {'subject','night','age','sex (F=1)'}.issubset(table.columns):
        raise ValueError('unrecognized SC subject metadata layout')
    keys = [(f'SC:{int(row.subject):02d}',int(row.night)) for row in table.itertuples(index=False)]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate SC participant/night metadata')
    return set(keys), None


def public_checks(root, files, hashes):
    results, errors = {}, []
    p = root/'SHA256SUMS.txt'
    expected = {}
    if p.is_file():
        for line in p.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip('*')
            if name in expected:
                raise ValueError('duplicate external checksum identity')
            expected[name] = digest
    p = root/'RECORDS'
    listed = set(p.read_text().splitlines()) if p.is_file() else set()
    checked, listed_psg, missing = 0, 0, []
    for path in files.values():
        if not path.is_relative_to(root):
            continue
        rel = str(path.relative_to(root))
        if rel in expected:
            checked += 1
            if hashes[str(path)] != expected[rel]:
                errors.append({'file': rel, 'error': 'external SHA256 checksum mismatch'})
        elif path.suffix == '.edf':
            missing.append(rel)
        if path.name.endswith('-PSG.edf') and (rel in listed or rel.removesuffix('-PSG.edf') in listed):
            listed_psg += 1
    results = {'checksum_inputs_present': bool(expected), 'n_verified_file_checksums': checked,
               'checksum_entries': len(expected), 'raw_files_without_published_checksum': missing,
               'records_entries': len(listed), 'n_psg_in_RECORDS': listed_psg}
    # RECORDS may contain complete filenames or stems; validate both forms.
    missing_records = [str(p.relative_to(root)) for p in files.values() if p.is_relative_to(root)
                       and p.name.endswith('-PSG.edf') and str(p.relative_to(root)) not in listed
                       and str(p.relative_to(root)).removesuffix('-PSG.edf') not in listed]
    if listed and missing_records:
        errors.append({'error': 'PSG files absent from RECORDS', 'files': missing_records})
    if expected and missing:
        errors.append({'error': 'raw files missing external checksums', 'files': missing})
    return results, errors


def pair_raw(config):
    raw, nights = {}, set()
    for key, study in [('sleep_edfx_sc_raw','SC'),('sleep_edfx_st_raw','ST')]:
        root = Path(config[key]).expanduser().resolve()
        by_pair = {}
        for p in sorted(root.glob('*.edf')):
            if p.name.endswith('-PSG.edf'):
                kind = 'psg'
            elif p.name.endswith('-Hypnogram.edf'):
                kind = 'hyp'
            else:
                raise ValueError(f'unrecognized raw EDF filename: {p.name}')
            ident = identity(p.name,kind)
            if ident['study'] != study:
                raise ValueError('raw file in wrong study directory')
            entry = by_pair.setdefault(ident['pair_id'],{**ident})
            if kind in entry:
                raise ValueError(f'duplicate or ambiguous {kind} pair: {p.name}')
            entry[kind] = p
        for entry in by_pair.values():
            if not {'psg','hyp'}.issubset(entry):
                raise ValueError(f'unpaired raw recording: {entry["recording_id"]}')
            if entry['recording_id'] in raw:
                raise ValueError('duplicate raw recording ID')
            night_key = (entry['subject_id'],entry['night'])
            if night_key in nights:
                raise ValueError('conflicting raw recording variants for one participant/night')
            nights.add(night_key)
            raw[entry['recording_id']] = entry
    return raw


def summary_rows(rows):
    result = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset']==dataset]
        result[dataset] = {'recordings': len(group), 'subjects': len({r['subject_id'] for r in group}),
                           'epochs': sum(r.get('n_epochs',0) for r in group),
                           'fully_aligned_recordings': sum(r.get('indices_status') == 'all_epochs_waveform_verified' for r in group),
                           'annotation_verified_recordings': sum(r.get('indices_status') in {'original_raw_grid','all_epochs_waveform_verified'} for r in group)}
    return result


def overlap(rows):
    matrix = []
    datasets = sorted({r['dataset'] for r in rows})
    for a, b in itertools.combinations_with_replacement(datasets,2):
        ar, br = [r for r in rows if r['dataset']==a], [r for r in rows if r['dataset']==b]
        subj = {r['subject_id'] for r in ar} & {r['subject_id'] for r in br}
        rec = {r['recording_id'] for r in ar} & {r['recording_id'] for r in br}
        sig_a = {r.get('signal_sha256') for r in ar}-{None}
        sig_b = {r.get('signal_sha256') for r in br}-{None}
        verified = {r['recording_id'] for r in ar if r.get('indices_status') in {'all_epochs_waveform_verified','original_raw_grid'}} & {r['recording_id'] for r in br if r.get('indices_status') in {'all_epochs_waveform_verified','original_raw_grid'}}
        matrix.append({'dataset_a': a, 'dataset_b': b, 'overlap_subjects': len(subj),
                       'overlap_recordings': len(rec), 'equal_full_signal_hashes': len(sig_a & sig_b),
                       'raw_waveform_verified_common_recordings': len(verified),
                       'independent_cohort_comparison': False if a.startswith('edf') and b.startswith('edf') else None})
    edf20 = {r['subject_id'] for r in rows if r['dataset']=='edf20_npz'}
    edf78 = {r['subject_id'] for r in rows if r['dataset']=='edf78_npz'}
    partitions = {'purpose': 'eligibility only; no train/val/test role assignment',
                  'edf20_subjects': sorted(edf20), 'shared_sc_subjects': sorted(edf20 & edf78),
                  'edf78_sc_extension_subjects': sorted(edf78-edf20),
                  'edf78_sc_extension_recordings': sorted(r['recording_id'] for r in rows if r['dataset']=='edf78_npz' and r['subject_id'] not in edf20),
                  'st_proxy_subjects': sorted({r['subject_id'] for r in rows if r['study']=='ST'})}
    return matrix, partitions


def audit(config, output, seed=123, resume=False, dry_run=False, split_manifest=None):
    roots, files = discover(config)
    if dry_run:
        result = {'paths': roots, 'files_to_hash': len(files), 'training': False,
                  'git': git_info(), 'environment': environment(), 'config_hash': fingerprint(config),
                  'seed': seed, 'split_ids': {}, 'output_directory': str(Path(output).resolve()),
                  'source_recording_hashes': {}, 'leakage_flags': {'test_metrics': False, 'shhs_access': False},
                  'note': 'discovery only; files not opened; no scientific acceptance'}
        print(json.dumps(result,indent=2))
        return 0
    print(f'Hashing {len(files)} inventory inputs (read-only)',flush=True)
    hashes = {name: sha256_file(p) for name, p in sorted(files.items())}
    split_rows = json.loads(Path(split_manifest).read_text())['records'] if split_manifest else []
    if split_manifest:
        hashes[str(Path(split_manifest).resolve())] = sha256_file(Path(split_manifest))
    run = AuditRun(output, config, hashes, seed, resume, split_ids={r['recording_id']: r['split'] for r in split_rows})
    errors, blockers, warnings, rows, channels, distributions = [], [], [], [], [], []
    missing = [key for key in PATH_KEYS[:-1] if not roots[key]['exists']]
    if missing:
        blockers.append('Required data paths missing: '+', '.join(missing))
    # An explicit split is only checked for leakage; this inventory must not open
    # already reserved test annotations or signals. Fail closed before enumeration.
    if split_rows:
        try:
            validate_splits(split_rows)
            if any(r['split']=='test' for r in split_rows):
                raise ValueError('reserved test opening prohibited in development inventory; audit before reserving test roles')
        except ValueError as exc:
            errors.append({'error': str(exc), 'hard_stop': True})
            return write_report(run, roots, rows, channels, distributions, errors, blockers, warnings, {}, [], {})
    original = Path(config['sleep_edfx_original_root']).expanduser().resolve()
    checks, check_errors = public_checks(original, files, hashes)
    errors.extend(check_errors)
    for name in ['RECORDS', 'SHA256SUMS.txt', 'SC-subjects.xls', 'ST-subjects.xls']:
        if not (original/name).is_file():
            blockers.append('External provenance input missing: '+name)
    try:
        treatments, issue = treatment_metadata(original)
        if issue:
            blockers.append(issue)
        sc_keys, issue = sc_metadata(original)
        if issue:
            blockers.append(issue)
        raw = pair_raw(config)
        for study in ['SC', 'ST']:
            if not any(r['study'] == study for r in raw.values()):
                blockers.append(study+': no raw PSG/hypnogram pairs found')
    except (ValueError,OSError) as exc:
        errors.append({'error': str(exc), 'hard_stop': True})
        return write_report(run, roots, rows, channels, distributions, errors, blockers, warnings, checks, [], {})
    npzs = {}
    for key, dataset in [('sleep_edf_20_npz','edf20_npz'),('sleep_edf_78_npz','edf78_npz')]:
        root = Path(config[key]).expanduser().resolve()
        seen = set()
        for path in sorted(root.glob('*.npz')):
            try:
                ident = identity(path.name)
                if ident['recording_id'] in seen:
                    raise ValueError('duplicate NPZ canonical identity')
                seen.add(ident['recording_id'])
                npzs.setdefault(ident['recording_id'],[]).append((dataset,path,ident))
            except ValueError as exc:
                errors.append({'file': path.name, 'error': str(exc)})
        if not seen:
            blockers.append(dataset+': no NPZ recordings found')
        manifest = root/'MANIFEST.TXT'
        if manifest.is_file():
            listed = {}
            for line in manifest.read_text().splitlines():
                match = __import__('re').fullmatch(r'([^ ]+\.npz) \(application/octet-stream\) (\d+) bytes\.',line)
                if not match:
                    errors.append({'file': str(manifest),'error': 'unrecognized NPZ manifest line'})
                    continue
                name,size = match.groups()
                if name in listed:
                    errors.append({'file': name, 'error': 'duplicate NPZ manifest entry'})
                listed[name]=int(size)
            actual = {p.name: p.stat().st_size for p in root.glob('*.npz')}
            missing_listed = sorted(set(listed)-set(actual))
            unlisted = sorted(set(actual)-set(listed))
            size_conflicts = [{'file':name,'manifest_bytes':listed[name],'actual_bytes':actual[name]}
                             for name in sorted(set(listed)&set(actual)) if listed[name] != actual[name]]
            if missing_listed or size_conflicts:
                errors.append({'file': str(manifest), 'error': 'NPZ manifest listed files/sizes disagree',
                               'missing_listed_files':missing_listed,'size_conflicts':size_conflicts})
            if unlisted:
                blockers.append(f'{dataset}: incomplete optional MANIFEST.TXT ({len(listed)}/{len(actual)} files listed; {len(unlisted)} unlisted); provenance review required')
            checks[dataset+'_manifest'] = {'entries':len(listed), 'actual_files':len(actual),
                                          'matches_filenames_and_sizes':actual==listed,
                                          'missing_listed_files':missing_listed, 'unlisted_files':unlisted,
                                          'size_conflicts':size_conflicts, 'sha256':hashes[str(manifest)]}
    # Stop data interpretation on proven identity/checksum/manifest failure.
    if errors:
        return write_report(run, roots, rows, channels, distributions, errors, blockers, warnings, checks, [], {})
    shape_owners, signal_owners = {}, {}
    for rid in sorted(set(raw)|set(npzs)):
        entry = raw.get(rid)
        labels, raw_wave, raw_channel = None, None, None
        source = {}
        if entry:
            source = {k:v for k,v in entry.items() if k not in {'psg','hyp'}}
            source.update(dataset='raw_'+entry['study'].lower(),path=str(entry['psg']),
                          source_sha256=hashes[str(entry['psg'])],annotation_source=str(entry['hyp']),
                          annotation_sha256=hashes[str(entry['hyp'])], treatment=treatments.get((entry['subject_id'],entry['night'])))
            try:
                if entry['study']=='SC' and sc_keys and (entry['subject_id'],entry['night']) not in sc_keys:
                    raise ValueError('SC participant/night absent from source spreadsheet')
                if entry['study']=='ST' and treatments and source['treatment'] is None:
                    raise ValueError('ST participant/night absent from treatment spreadsheet')
                # PSG headers are inventoried independently so a rejected
                # hypnogram never erases known channel/rate/duration evidence.
                with pyedflib.EdfReader(str(entry['psg'])) as header_reader:
                    headers = header_reader.getSignalHeaders()
                    eeg_headers = [h for h in headers if h['label'].startswith('EEG')]
                    if not eeg_headers:
                        raise ValueError('no calibrated EEG channel')
                    for h in eeg_headers:
                        calibration(h)
                    source.update(channels=headers, duration_seconds=float(header_reader.file_duration),
                                  start_datetime=header_reader.getStartdatetime().isoformat(),
                                  n_epochs=int(header_reader.file_duration//30),
                                  indices_status='raw_grid_annotation_unverified')
                    for h in headers:
                        channels.append({'dataset':source['dataset'],'recording_id':rid,'channel':h['label'],
                                         'fs':h['sample_frequency'],'units':h['dimension'],
                                         'physical_min':h['physical_min'],'physical_max':h['physical_max'],
                                         'digital_min':h['digital_min'],'digital_max':h['digital_max'],
                                         'prefilter':h['prefilter'],'transducer':h['transducer']})
                with pyedflib.EdfReader(str(entry['psg'])) as psg, pyedflib.EdfReader(str(entry['hyp'])) as hyp:
                    if psg.getStartdatetime() != hyp.getStartdatetime():
                        raise ValueError('PSG/hypnogram start-time disagreement')
                    onsets,durations,texts = hyp.readAnnotations()
                    labels, ann = annotation_grid(onsets,durations,texts,psg.file_duration)
                    source.update(ann, n_epochs=len(labels),duration_seconds=float(psg.file_duration),
                                  start_datetime=psg.getStartdatetime().isoformat(),indices_status='original_raw_grid')
                    source['mapped_stage_counts'] = {name:int(np.sum(labels==i)) for i,name in enumerate(STAGES)}
                    source['excluded_epochs'] = int(np.sum(labels<0))
                    if rid in npzs:
                        # Read exactly the NPZ-declared EEG; no assumed montage.
                        requested = []
                        for _,path,_ in npzs[rid]:
                            with np.load(path,allow_pickle=False) as z:
                                requested.append(str(z['ch_label'].item()))
                        if len(set(requested)) != 1:
                            raise ValueError('preprocessed sets disagree in channel')
                        names = [h['label'] for h in headers]
                        if names.count(requested[0]) != 1:
                            raise ValueError('NPZ montage unavailable or ambiguous in raw EDF')
                        c = names.index(requested[0]); raw_channel = headers[c]
                        if raw_channel['sample_frequency'] != 100:
                            raise ValueError('NPZ/native sample-rate mismatch; preprocessing transform unverified')
                        raw_signal = psg.readSignal(c)
                        if len(raw_signal) < len(labels)*3000:
                            raise ValueError('raw EEG length disagrees with annotation grid')
                        raw_wave = raw_signal[:len(labels)*3000].reshape(-1,3000)
                        source['signal_sha256'] = hashlib.sha256(np.ascontiguousarray(raw_wave,dtype='<f4').tobytes()).hexdigest()
                        source['shape_sha256'] = hashlib.sha256(''.join(shape_key(f) for f in shape_features(raw_wave)).encode()).hexdigest()
            except (ValueError,OSError,KeyError) as exc:
                errors.append({'recording_id':rid,'error':str(exc),'hard_stop':True})
                # Inventory remaining records is read-only; downstream gates stay closed.
                source['indices_status']='invalid'
                source['annotation_status']='unverified; strict reader/validation failure recorded'
                labels,raw_wave = None,None
            rows.append(source)
            for name,count in source.get('mapped_stage_counts',{}).items():
                distributions.append({'dataset':source['dataset'],'recording_id':rid,'stage':name,'epochs':count})
            for name,count in source.get('raw_stage_counts',{}).items():
                distributions.append({'dataset':source['dataset'],'recording_id':rid,'stage':'raw:'+name,'epochs':count})
        for dataset,path,ident in npzs.get(rid,[]):
            row = {**ident,'dataset':dataset,'path':str(path),'source_sha256':hashes[str(path)],
                   'indices_status':'unknown','raw_units':None,'preprocessing_provenance':'unknown upstream script/policy'}
            try:
                x,y,declared_idx,meta = read_npz(path)
                sketches = meta.pop('epoch_shape_hashes')
                row.update(meta)
                row['mapped_stage_counts'] = {name:int(np.sum(y==i)) for i,name in enumerate(STAGES)}
                for name,count in row['mapped_stage_counts'].items():
                    distributions.append({'dataset':dataset,'recording_id':rid,'stage':'npz_code:'+name,'epochs':count})
                for digest in [row['signal_sha256']]:
                    previous = signal_owners.get(digest)
                    if previous and previous != rid:
                        raise ValueError(f'identical signal under conflicting recording IDs: {previous}, {rid}')
                    signal_owners[digest] = rid
                # Detect shared normalized sparse waveform fingerprints across
                # different people/recordings, even if wrappers/units differ.
                owners = {}
                for sketch in set(sketches):
                    for previous in shape_owners.get(sketch,set())-{rid}:
                        owners[previous] = owners.get(previous,0)+1
                suspicious = [previous for previous,count in owners.items() if count >= 3]
                if suspicious:
                    raise ValueError(f'potential transformed signal overlap under conflicting IDs: {suspicious}')
                for sketch in set(sketches):
                    shape_owners.setdefault(sketch,set()).add(rid)
                if raw_wave is None or labels is None:
                    blockers.append(f'{dataset}/{rid}: no usable raw pair; original epoch indices/units/stage semantics unverified')
                else:
                    idx,fit = align_epochs(x,raw_wave)
                    if declared_idx is not None and not np.array_equal(idx,declared_idx):
                        raise ValueError('declared original indices disagree with waveform alignment')
                    mismatch = int(np.sum(y != labels[idx]))
                    if mismatch:
                        raise ValueError(f'NPZ/raw stage mapping disagreement at {mismatch} aligned epochs')
                    unit_identity = abs(fit['gain_min']-1)<1e-5 and abs(fit['gain_max']-1)<1e-5 and max(abs(fit['offset_min']),abs(fit['offset_max']))<1e-5
                    if row['declared_units'] is not None and row['declared_units'] != raw_channel['dimension'] and unit_identity:
                        raise ValueError('declared NPZ physical units conflict with raw calibration')
                    row.update(indices_status='all_epochs_waveform_verified', original_epoch_runs=index_runs(idx),
                               original_epoch_index_sha256=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),
                               start_seconds_first=int(idx[0])*30,start_seconds_last=int(idx[-1])*30,
                               waveform_verification=fit,stage_semantics='all aligned labels verified against R&K mapping',
                               raw_source_sha256=source['source_sha256'],raw_annotation_sha256=source['annotation_sha256'],
                               raw_units=raw_channel['dimension'] if unit_identity else None,
                               units_status='verified identity to calibrated raw EEG' if unit_identity else 'unknown; affine relation measured',
                               selection_status='retained subset; upstream trimming/exclusion policy unknown',
                               omitted_original_full_epochs=len(labels)-len(idx))
                    if not unit_identity:
                        blockers.append(f'{dataset}/{rid}: NPZ amplitude conversion must be documented before use')
                    # Never reuse label-dependent evaluation trimming. Detect
                    # the retained subset without prescribing how it was chosen.
                    if len(idx) != len(labels):
                        warnings.append(f'{dataset}/{rid}: NPZ omits {len(labels)-len(idx)} original full epochs; unsuitable as untrimmed zero-shot evaluation input')
                channels.append({'dataset':dataset,'recording_id':rid,'channel':row['channel'],'fs':row['fs'],
                                 'units':row['raw_units'] or 'unknown', 'physical_min':'','physical_max':'',
                                 'digital_min':'','digital_max':'','prefilter':'unknown upstream process','transducer':'raw linked, upstream unknown'})
            except (ValueError,OSError,KeyError) as exc:
                errors.append({'dataset':dataset,'recording_id':rid,'error':str(exc),'hard_stop':True})
                row['indices_status']='invalid'
            rows.append(row)
        print(f'Audited {rid}: raw={bool(entry)} npz_sets={len(npzs.get(rid,[]))}',flush=True)
    # Extra EDF copies in NPZ folders: compare hashes to authoritative source
    # pairs, never count them as independent recordings.
    extras = []
    for key in ['sleep_edf_20_npz','sleep_edf_78_npz']:
        for p in sorted(Path(config[key]).glob('*.edf')):
            kind = 'psg' if p.name.endswith('-PSG.edf') else 'hyp'
            try:
                ident = identity(p.name,kind); original_entry = raw.get(ident['recording_id'])
                matched = original_entry is not None and hashes[str(p.resolve())] == hashes[str(original_entry[kind])]
                extras.append({'path':str(p),'sha256':hashes[str(p.resolve())],'matches_raw_copy':matched})
                if not matched:
                    errors.append({'file':str(p),'error':'extra EDF copy differs from authoritative raw source'})
            except (ValueError,KeyError) as exc:
                errors.append({'file':str(p),'error':str(exc)})
    matrix,partitions = overlap(rows)
    if not config.get('shhs_access_authorized',False):
        warnings.append('SHHS not opened: authorization and usable paired data unconfirmed; no external-cohort claim possible')
    return write_report(run, roots, rows, channels, distributions, errors, blockers, warnings, checks, extras, partitions, matrix)


def write_report(run, roots, rows, channels, distributions, errors, blockers, warnings, checks, extras, partitions, matrix=None):
    matrix = matrix or []
    gate = 'FAIL' if errors or blockers else 'PASS_FOR_P2_CANONICAL_PREPROCESSING_ONLY'
    summary = summary_rows(rows)
    scientific = {'gate':gate,'errors':errors,'blockers':sorted(set(blockers)), 'warnings':sorted(set(warnings)),
                  'paths':roots,'inventory':summary,'records':rows,'overlap_matrix':matrix,
                  'subject_disjoint_partitions':partitions,'public_provenance_checks':checks,'extra_edf_copies':extras,
                  'leakage_flags':{'edf20_edf78_overlap_expected': bool(partitions.get('shared_sc_subjects')),
                      'independent_edf20_edf78_transfer_allowed':False,'split_assignments_created':False,
                      'split_leakage_check_required_before_experiments':True,'test_metrics_computed':False,
                      'test_subjects_used_for_model_selection':False,'shhs_opened':False},
                  'scientific_limits':['SC vs ST is a confounded cross-study proxy, not isolated acquisition shift',
                      'NPZ upstream preprocessing script/selection policy unknown; use new raw pipeline for evaluation',
                      'Waveform affine agreement establishes source positions, not clinical morphology detection',
                      'Study-scoped IDs do not establish person-level disjointness between SC and ST; cross-study linkage unavailable',
                      'Alignment tolerance 2e-6 and shape sketches can reject ambiguous data; never assume continuity'],
                  'training_performed':False,'p2_can_begin':not bool(errors or blockers)}
    result = {**scientific,'provenance':run.identity,'signature':run.signature}
    atomic_json(result,run.output/'audit.json')
    inventory_fields = ['dataset','study','subject_id','night','recording_id','path','source_sha256','annotation_source',
                        'annotation_sha256','n_epochs','indices_status','fs','channel','raw_units','treatment',
                        'original_epoch_runs','original_epoch_index_sha256','omitted_original_full_epochs','preprocessing_provenance']
    flat = [{k: json.dumps(r[k]) if isinstance(r.get(k),(dict,list)) else r.get(k,'') for k in inventory_fields} for r in rows]
    atomic_csv(flat, inventory_fields, run.output/'edf_inventory.csv')
    atomic_csv(matrix,['dataset_a','dataset_b','overlap_subjects','overlap_recordings','equal_full_signal_hashes',
                       'raw_waveform_verified_common_recordings','independent_cohort_comparison'],run.output/'overlap_matrix.csv')
    atomic_csv(distributions,['dataset','recording_id','stage','epochs'],run.output/'stage_distributions.csv')
    atomic_csv(channels,['dataset','recording_id','channel','fs','units','physical_min','physical_max','digital_min',
                         'digital_max','prefilter','transducer'],run.output/'channel_units.csv')
    atomic_json(partitions,run.output/'subject_disjoint_partitions.json')
    lines = ['MIST-Transfer v3 P1 DATA AUDIT ONLY',f'Git input SHA: {run.identity["git"]["sha"]}',
             f'Gate: {gate}',f'P2 can begin: {scientific["p2_can_begin"]}',
             'No preprocessing, model training, target metrics or SHHS access.']
    lines += [f'{dataset}: {json.dumps(counts,sort_keys=True)}' for dataset,counts in summary.items()]
    lines += [f'Overlap: {json.dumps(row,sort_keys=True)}' for row in matrix if row['dataset_a']!=row['dataset_b']]
    lines += [f'Errors: {len(errors)}; blockers: {len(set(blockers))}; warnings: {len(set(warnings))}',
              'EDF20 vs EDF78 is same SC cohort, never independent cross-dataset transfer.',
              'Unknown upstream NPZ selection policy: rebuild from raw; do not trim target evaluation using labels.',
              'Test results: see session test logs; no test-stage performance measured.']
    lines += [f'ERROR: {json.dumps(e,sort_keys=True)}' for e in errors]
    lines += [f'BLOCKER: {b}' for b in sorted(set(blockers))]
    lines += ['NEXT SINGLE SAFE COMMAND: git status --short --branch (STOP after P0/P1 for review).']
    atomic_text('\n'.join(lines)+'\n',run.output/'FINAL_COPY_PASTE.txt')
    outputs=['audit.json','edf_inventory.csv','overlap_matrix.csv','stage_distributions.csv','channel_units.csv',
             'subject_disjoint_partitions.json','FINAL_COPY_PASTE.txt']
    run.finish(outputs,gate)
    print('\n'.join(lines),flush=True)
    return 2 if errors or blockers else 0

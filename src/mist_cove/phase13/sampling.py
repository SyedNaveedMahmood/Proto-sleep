"""Outcome-free primary ranks and separately identified prior-error enrichment."""
import json
import secrets
import time
import numpy as np
from mist_cove.phase11b.source import select_epochs
from mist_cove.phase11b.historical import canonical
from mist_cove.phase11b.security import sha256, IntegrityError
from .provenance import (PUBLIC, PRIVATE, OLD_PUBLIC, OLD_PRIVATE, setup_source,
                         authenticated_json_identity, write_new)


def primary_panel(records):
    first_twelve, _ = select_epochs(records, 12)
    counts = {}
    selected = []
    for coordinate in first_twelve:
        key = coordinate['subject'], coordinate['stage']
        counts[key] = counts.get(key, 0)+1
        if counts[key] in (11, 12):
            selected.append(coordinate)
    missing = 150-len(selected)
    return selected, missing


def secondary_panel(rows):
    def compact(row):
        return any(c['negative_bits'] == 0 and 0 < c['retained_seconds'] <= 8
                   and c['signed_literals'] <= 8 for c in row['summary']['signed']['clauses'])
    criteria = [
        ('N1_error', lambda r:r['coordinate']['stage'] == 1 and not r['correct']),
        ('N2_N3_error', lambda r:{r['coordinate']['stage'],r['native']['prediction']} == {2,3}),
        ('REM_N2_error', lambda r:{r['coordinate']['stage'],r['native']['prediction']} == {2,4}),
        ('high_confidence_error', lambda r:not r['correct'] and r['native']['confidence'] >= .9),
        ('correct_compact', lambda r:r['correct'] and compact(r)),
        ('global_dependence', lambda r:r['feature_ablations']['global_only']['prediction'] == r['native']['prediction']
         and r['feature_ablations']['local_only']['prediction'] != r['native']['prediction'])]
    selected, seen, coverage = [], set(), {}
    for name, condition in criteria:
        count = 0
        for row in rows:
            c = row['coordinate'];key = c['recording_id'],c['epoch']
            if key not in seen and condition(row):
                selected.append({**c, 'selection_category':name});seen.add(key);count += 1
                if count == 5:break
        coverage[name] = {'selected':count,'missing':5-count}
    return selected, coverage


def freeze():
    started = time.perf_counter()
    _, access = setup_source()
    records = access.records('train')
    primary, missing = primary_panel(records)
    old_selection = json.loads((OLD_PUBLIC/'sample_selection.json').read_text())
    previous = authenticated_json_identity(OLD_PRIVATE/'selection.json',old_selection['selection_sha256'])
    used = {(c['recording_id'],c['epoch']) for c in previous['extended']}
    if any((c['recording_id'],c['epoch']) in used for c in primary):
        raise IntegrityError('new primary panel overlaps old extended audit')
    rows = [json.loads(line) for line in (OLD_PRIVATE/'audit_rows.jsonl').read_text().splitlines()]
    secondary, coverage = secondary_panel(rows)
    partition_receipt = json.loads((OLD_PUBLIC/'participant_partition.json').read_text())
    allocation = authenticated_json_identity(OLD_PRIVATE/'participant_allocation.json',partition_receipt['allocation_sha256'])
    if set(allocation['fit']) & set(allocation['internal_dev']) or len(allocation['fit']) != 10 or len(allocation['internal_dev']) != 5:
        raise IntegrityError('prior10/5 allocation changed')
    subjects = {r['subject_id'] for r in records}
    if set(allocation['fit']) | set(allocation['internal_dev']) != subjects:
        raise IntegrityError('partition not exactly original TRAIN')
    fit, fit_missing = select_epochs([r for r in records if r['subject_id'] in allocation['fit']],100)
    dev, dev_missing = select_epochs([r for r in records if r['subject_id'] in allocation['internal_dev']],40)
    training = {'fit':fit,'internal_dev':dev,'missing_fit':fit_missing,'missing_dev':dev_missing}
    old_data_receipt = json.loads((OLD_PUBLIC/'new_model_data_receipt.json').read_text())
    if sha256(canonical(training)) != old_data_receipt['selection_sha256']:
        raise IntegrityError('controlled Phase XII input list differs')
    norm = authenticated_json_identity(OLD_PRIVATE/'new_model_normalization.json',old_data_receipt['normalization_sha256'])
    if norm['fit_role'] != 'train_fit' or norm['fit_participants'] != 10 or norm['std_uV'] <= 0:
        raise IntegrityError('fit-group normalization boundary')
    mechanistic = []
    for subject in sorted(allocation['internal_dev']):
        for stage in range(5):
            candidates = [c for c in primary if c['subject'] == subject and c['stage'] == stage]
            if candidates:mechanistic.append(candidates[0])
    manifest = {'primary':primary,'secondary':secondary,'training':training,
                'allocation':allocation,'normalization':norm,'mechanistic':mechanistic}
    write_new(PRIVATE/'selection.json',manifest)
    receipt = {'status':'FROZEN_BEFORE_PHASE_XIII_PREDICTIONS','canonical_sha256':sha256(canonical(manifest)),
        'primary':len(primary),'primary_missing':missing,'primary_old_extended_overlap':0,
        'secondary':len(secondary),'secondary_strata':coverage,'mechanistic':len(mechanistic),
        'fit':len(fit),'internal_dev':len(dev),'fit_missing':sum(v['missing'] for v in fit_missing),
        'dev_missing':sum(v['missing'] for v in dev_missing),'fit_group_normalization_reused':True,
        'normalization_sha256':old_data_receipt['normalization_sha256'],'allocation_sha256':partition_receipt['allocation_sha256'],
        'source_access':access.counts,'wall_seconds':time.perf_counter()-started,'raw_waveforms_read':False,'raw_VAL_read':False}
    write_new(PUBLIC/'sample_selection.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('primary','secondary','mechanistic','fit','internal_dev','primary_old_extended_overlap')}))


def frozen_selection():
    receipt = json.loads((PUBLIC/'sample_selection.json').read_text())
    return authenticated_json_identity(PRIVATE/'selection.json',receipt['canonical_sha256'])

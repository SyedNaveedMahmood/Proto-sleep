"""Aggregate release checks and accounting; never opens an EDF or checkpoint."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from mist_cove.phase11b.evaluation import clustered_mean
from .common import ROOT, PUBLIC, PRIVATE, read_json, write_new, atomic_json, measured


def sensitivity(correction=False):
    """Summarize already sealed 0/.1/.25-second event overlap results."""
    result = {}
    stage = 'postflight_fixed_tolerance_status_filter_correction' if correction else 'postflight_fixed_tolerance_aggregation'
    with measured(stage, cached_only=True, head_queries=0):
        for role in ('SC_TRAIN', 'ST_EVAL'):
            rows = [json.loads(line) for line in (PRIVATE / (role + '_witness_rows.jsonl')).read_text().splitlines()]
            if role == 'SC_TRAIN':
                fit = set(read_json(PRIVATE / 'selection.json')['allocation']['fit'])
                rows = [r for r in rows if r['primary'] and r['coordinate']['subject'] not in fit]
            else:
                rows = [r for r in rows if r['primary']]
            result[role] = {}
            for key in ('EEG Pz-Oz|spindle', 'EEG Pz-Oz|spindle_strict',
                        'EEG Pz-Oz|slow_wave', 'EEG Pz-Oz|slow_wave_relaxed'):
                by_tol = {}
                for tol in ('0', '0.1', '0.25'):
                    values = defaultdict(list); units = defaultdict(list); eligible = 0
                    for r in rows:
                        endpoint = r['proxy_alignment']['proxies'].get(key)
                        if not endpoint or not endpoint['event_eligible'] or not endpoint['support_atoms']:
                            continue
                        statuses = r['detector_record_statuses'].get(key,{})
                        if 'DETECTOR_FAILED' in statuses or 'UNASSESSABLE_FLAT' in statuses:
                            continue
                        eligible += 1
                        methods = endpoint['methods']; person = r['coordinate']['subject']
                        exact = methods['exact_positive'][tol]['precision']
                        for method in ('random_matched', 'RMS_matched', 'sigma_matched',
                                       'position_matched', 'gradient_matched'):
                            other = methods[method][tol]['precision']
                            if exact is not None and other is not None:
                                name = 'exact_minus_' + method
                                values[name].append(exact - other); units[name].append(person)
                    by_tol[tol] = {'eligible_cases': eligible,
                                   'contrasts': {k: clustered_mean(v, units[k]) for k, v in values.items()}}
                result[role][key] = by_tol
        output = {
            'method': 'cached masks and detector intervals; zero new EEG reads or head queries',
            'SC_scope': 'five developmental people only', 'ST_scope': 'frozen selected panel',
            'human_event_truth': 'NOT_TESTED', 'results': result}
        path = PUBLIC/'proxy_tolerance_sensitivity.json'
        if correction:atomic_json(path,output)
        else:write_new(path,output)


def resources():
    stages = read_json(PUBLIC / 'resource_stages.json')
    total = sum(float(x['wall_seconds']) for x in stages.values())
    sc = read_json(PUBLIC / 'SC_TRAIN_witness_execution.json')['resources']
    st = read_json(PUBLIC / 'ST_EVAL_witness_execution.json')['resources']
    ctx = read_json(PUBLIC / 'context_operator_results.json')
    scientific_queries = sc['cached_head_queries'] + st['cached_head_queries'] + ctx['cached_head_queries']
    # Generated counts are independently bounded; historical regression tests are
    # engineering checks and are not recast as full-network training exposures.
    known_generated = 4 * 131072 + 65538
    own_private_bytes = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file() and not p.is_symlink())
    detector_venv_bytes = sum(p.stat().st_size for p in (PRIVATE/'detector_environment').rglob('*')
                              if p.is_file() and not p.is_symlink())
    result = {
        'scientific_fits': 6, 'fit_limit': 6, 'optimizer_updates': 18000, 'update_limit': 18000,
        'ST_fit_updates': 0, 'GPU_seconds': 0, 'CUDA_correctness': 'NOT_RUN_CPU_ONLY_PATCHED_RUNTIME',
        'conservative_summed_stage_wall_seconds': total, 'scientific_CPU_wall_ceiling_seconds': 24000,
        'stage_wall_note': 'Sum includes failed engineering attempts and postflight aggregation; may overlap elapsed time. Not CPU-core seconds.',
        'cached_head_queries_scientific': scientific_queries,
        'cached_head_queries_by_track': {'SC': sc['cached_head_queries'], 'ST': st['cached_head_queries'], 'context': ctx['cached_head_queries']},
        'known_generated_proof_queries_lower_bound': known_generated,
        'known_scientific_plus_generated_proof_queries_lower_bound': scientific_queries + known_generated,
        'head_query_ceiling': 90000000,
        'query_accounting_limit': 'Preexisting regression-suite synthetic calls were not instrumented individually; combined exact all-test query total is not certified.',
        'tables': {'SC': sc['tables'], 'ST': st['tables'], 'context': ctx['tables']},
        'native_original_epoch_forwards_unique': {'SC': 79984, 'ST': 45446},
        'ST_initial_transport_failure_original_forwards_reused': 1196,
        'ST_context_base_epoch_forwards_in_frozen_full_grid': 45446 * 18,
        'context_training_base_waveform_forwards': 1728000,
        'context_training_clean_and_paired_center_exposures': [288000, 288000],
        'source_access': {
            'metadata_reads_initial': 3, 'official_condition_spreadsheet_reads': 1,
            'initial_authenticated_source_hash_reads': 146,
            'SC_source_campaign_hash_and_waveform_opens': 29,
            'ST_failed_first_record_hash_and_waveform_opens': 1,
            'ST_continuation_hash_and_waveform_opens': 44,
            'private_atlas_hash_and_waveform_opens': 8,
            'total_source_hash_reads_or_rechecks': 228,
            'total_PSG_waveform_value_opens': 82,
            'header_reads': 73, 'hypnogram_reads': 73,
            'unique_PSG_and_hypnogram_pairs': 73,
            'SC_VAL_new_raw_reads': 0, 'SC_RESERVED_reads': 0,
            'unapproved_SC_or_external_reads': 0,
            'source_count_note': 'Counts combine stage receipts and preserved first-record transport failure; repeated approved reopens are included. Private derived-cache reads are separate.'},
        'peak_parent_RSS_KiB_observed': max(int(v.get('peak_RSS_KiB',0)) for v in stages.values()),
        'peak_memory_limit': 'Detector subprocess peak and whole pipeline high-water were not instrumented.',
        'private_phase_folder_bytes_including_environment': own_private_bytes,
        'detector_environment_bytes': detector_venv_bytes,
        'unreviewed_reruns_for_favorable_outcome': 0,
        'failed_attempts_preserved': ['generated flat-control serialization', 'isolated worker import', 'ST continuation exclusive selection entry'],
    }
    atomic_json(PUBLIC/'resource_receipts.json', result)
    return result


def uncertainty():
    a = read_json(PUBLIC/'sc_alignment_results.json'); b = read_json(PUBLIC/'st_explanation_results.json')
    rows = {}
    for role, item, name in [('SC_TRAIN', a, 'development_primary'), ('ST_EVAL', b, 'primary')]:
        endpoint = item['strata'][name]['proxy_concordance']['EEG Pz-Oz|spindle']
        rows[role] = {'cases': item['strata'][name]['cases'],
                      'eligible_cases': endpoint['event_eligible_nonempty_cases'],
                      'eligible_people': endpoint['people_with_eligible_events'],
                      'exact_minus_random': endpoint['methods']['exact_minus_random_matched'],
                      'exact_minus_sigma': endpoint['methods']['exact_minus_sigma_matched']}
    write_new(PUBLIC/'statistical_uncertainty.json', {
        'unit': 'participant resampling; nights, cases and views remain within participant',
        'draws': 1000, 'seed': 123, 'primary_proxy_contrasts': rows,
        'SC_ST_unpaired': True, 'ST_condition_within_person_paired': True,
        'exploratory_strata_multiplicity_adjusted': False,
        'human_event_reference': 'NOT_TESTED'})


def privacy_and_manifest():
    """Scan exactly the new public tree/source and issue a file-digest manifest."""
    docs = sorted((ROOT/'docs/mist_cove').glob('PHASE_XIV_*.md'))
    src = sorted((ROOT/'src/mist_cove/phase14').glob('*.py'))
    scripts = sorted((ROOT/'scripts').glob('*mist_cove_phase14*.py'))
    public = sorted(p for p in PUBLIC.rglob('*') if p.is_file() and p.name != 'publication_manifest.json')
    tested = docs + src + scripts + public + [ROOT/'configs/mist_cove_phase14_v1.json', ROOT/'tests/mist_cove/test_phase14.py']
    actual_id = re.compile(r'\b(?:SC|ST)\d{3,}[A-Z0-9-]*\b')
    forbidden_keys = {'recording_id','subject_id','wave_uV','context_uV','mean_uV','std_uV','individual_predictions','reviewer_identity'}
    for p in tested:
        if p.suffix not in ('.json','.md','.py'): continue
        data = p.read_text()
        if ('/'+'home/') in data or actual_id.search(data.replace('SC9999E0', 'GENERATED_FIXTURE')):
            raise ValueError('private source path or identity in latest public artifact: '+str(p.relative_to(ROOT)))
        if p.suffix == '.json':
            def keys(x):
                if isinstance(x,dict):
                    if set(x) & forbidden_keys: raise ValueError('private key in '+str(p.relative_to(ROOT)))
                    for v in x.values():keys(v)
                elif isinstance(x,list):
                    for v in x:keys(v)
            keys(json.loads(data))
    pictures = sorted((PUBLIC/'figures').glob('*.png'))
    if len(pictures) < 10: raise ValueError('aggregate figure coverage incomplete')
    for p in pictures:
        if p.stat().st_size < 1000: raise ValueError('empty aggregate plot')
    for p in tested:
        if p.is_symlink():raise ValueError('public symlink')
    result = {'latest_tree_safe_scan':'PASS',
              'historical_branch_root_literal_exposure':'DISCLOSED_FAILURE_IN_EARLIER_PHASE_XIV_COMMITS',
              'history_rewritten':False,'raw_waveform_or_individual_prediction_publication':False,
              'private_atlas_publication':False,
              'human_validation':'NOT_TESTED',
              'figures':len(pictures),
              'file_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in tested},
              'manifest_excludes_itself':True}
    path=PUBLIC/'publication_manifest.json'
    atomic_json(path,result)
    return result

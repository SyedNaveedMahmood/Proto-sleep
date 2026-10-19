"""Descriptive additive-versus-individual-effect analysis; no fitting or EEG I/O.

These estimands were distinguished by the original protocol. Correlations are
descriptive and undefined for constant scores. No equality is required for a
truthful additive decomposition, and donor effects do not establish physiology.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def paired_summary(additive, effect):
    a, b = np.asarray(additive, dtype=float), np.asarray(effect, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite paired one-dimensional estimands required')
    if not len(a):
        return {'n': 0, 'mean_absolute_difference': None, 'pearson_correlation': None,
                'nonzero_sign_agreement': None, 'n_both_nonzero': 0}
    nonzero = (np.abs(a) > 1e-5) & (np.abs(b) > 1e-5)
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else None
    return {'n': len(a), 'mean_absolute_difference': float(np.abs(a-b).mean()),
            'pearson_correlation': correlation,
            'nonzero_sign_agreement': float((np.sign(a[nonzero]) == np.sign(b[nonzero])).mean()) if nonzero.any() else None,
            'n_both_nonzero': int(nonzero.sum())}


def summarize():
    source = BASE/'phase3_waveform_safe_results_v1.json'
    report = json.loads(source.read_text())
    assert report['status'] == 'POSTFIT_WAVEFORM_AUDIT_COMPLETE'
    models = {}
    for entry in report['private_detail_sha256']:
        seed, name = entry['seed'], entry['variant']
        path = BASE/'waveform_audit_v1'/f'{seed}_{name}_private.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['private_details_sha256']
        values = {key: [[], []] for key in ['true_stage', 'predicted_stage', 'all_five_stages']}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'EVENTLESS_NO_INTERVAL_ATTRIBUTION':
                continue
            intervals = row['groups_declared_before_effects']['all_selected']
            # The report preserves caller order for individual interventions.
            # Never align these vectors to the sorted, unioned joint intervals.
            effects = row['bridge']['groups']['all_selected']['donors'][0]['B_individual_counterfactual_effects']
            lookup = {(v['start_sample'], v['end_sample']): v['contributions'] for v in row['ledger']}
            assert len(effects) == len(intervals) == len(lookup)
            for interval, effect in zip(intervals, effects):
                additive = lookup[tuple(interval)]
                for key, stage in [('true_stage', row['truth']), ('predicted_stage', row['prediction'])]:
                    values[key][0].append(additive[stage]); values[key][1].append(effect[stage])
                values['all_five_stages'][0].extend(additive); values['all_five_stages'][1].extend(effect)
        models[f'{seed}_{name}'] = {key: paired_summary(*pair) for key, pair in values.items()}
    result = {'status': 'DESCRIPTIVE_INDIVIDUAL_ATTRIBUTION_SUMMARY_COMPLETE', 'models': models,
              'waveform_aggregate_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'donor': 'bridge', 'interpretation': 'Original additive event score versus full-model individual waveform effect; distinct estimands, not causal necessity. Correlation treats repeated witnesses descriptively, not as independent participants.',
              'training_executed': False, 'reserved_access': False, 'st_access': False, 'shhs_access': False}
    with (BASE/'phase3_individual_attribution_safe_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    summarize()

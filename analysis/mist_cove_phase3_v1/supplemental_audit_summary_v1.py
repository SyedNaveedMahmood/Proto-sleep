"""Exploratory postfit diagnostics of frozen interventions; no EEG I/O or fit.

No new donor, threshold-based model choice or waveform intervention is added.
The 0.9 confidence count is a descriptive supplement, not a success criterion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path.cwd()/'mist_transfer_runs/mist_cove/phase3_source_v1'


def mean(values):
    return float(np.mean(values)) if values else None


def summarize():
    path = BASE/'phase3_waveform_safe_results_v1.json'
    audit = json.loads(path.read_text())
    assert audit['status'] == 'POSTFIT_WAVEFORM_AUDIT_COMPLETE'
    methods = {}
    for method in audit['methods']:
        rows = []
        minimum_rms = []
        for seed in audit['seeds']:
            entry = next(e for e in audit['private_detail_sha256'] if e['seed'] == seed and e['variant'] == method)
            source = BASE/'waveform_audit_v1'/f'{seed}_{method}_private.jsonl'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['private_details_sha256']
            for line in source.read_text().splitlines():
                v = json.loads(line)
                if v.get('status') == 'EVENTLESS_NO_INTERVAL_ATTRIBUTION':
                    continue
                donor = v['bridge']['groups']['all_selected']['donors'][0]
                before = np.asarray(v['probabilities']); after = np.asarray(donor['diagnostics']['probabilities_after'])
                individual = np.asarray(donor['B_individual_counterfactual_effects']); joint = np.asarray(donor['C_group_counterfactual_effect'])
                record = {'original_correct': int(before.argmax()) == v['truth'], 'after_correct': int(after.argmax()) == v['truth'],
                          'original_confidence': float(before.max()), 'after_confidence': float(after.max()),
                          'after_wrong_confidence_at_least_0_9': int(after.argmax()) != v['truth'] and after.max() >= .9,
                          'group_absolute_true_stage_effect': float(abs(joint[v['truth']])),
                          'largest_individual_absolute_true_stage_effect': float(np.abs(individual[:,v['truth']]).max()),
                          'joint_exceeds_largest_individual_true_stage_by_0_25': abs(joint[v['truth']]) > np.abs(individual[:,v['truth']]).max()+.25,
                          'selected_groups_individual_necessity_not_established_flag': donor['flags']['individual_necessity_not_established'],
                          'original_true_stage_probability_increased_after_replacement': after[v['truth']] > before[v['truth']],
                          'maximum_logit_probability_increased_after_replacement': after.max() > before.max()}
                rows.append(record)
                if 'minimum_rms_uV' in v['background_donor_status']:
                    minimum_rms.append(v['background_donor_status']['minimum_rms_uV'])
        fields = {k: mean([r[k] for r in rows]) for k in rows[0]} if rows else {}
        fields.update(n_repeated_model_examples=len(rows), n_distinct_original_epochs=audit['examples_per_model'],
                      original_wrong_confidence_mean=mean([r['original_confidence'] for r in rows if not r['original_correct']]),
                      after_wrong_confidence_mean=mean([r['after_confidence'] for r in rows if not r['after_correct']]),
                      rejected_best_donor_centered_rms_uV_min=float(min(minimum_rms)) if minimum_rms else None,
                      rejected_best_donor_centered_rms_uV_median=float(np.median(minimum_rms)) if minimum_rms else None)
        methods[method] = fields
    report = {'status':'EXPLORATORY_FROZEN_INTERVENTION_SUMMARY_COMPLETE', 'methods':methods,
              'waveform_aggregate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'interpretation':'Postfit descriptive supplement, repeated optimization seeds on the same 200 epochs. Bridge-only effects do not establish clinical validity, necessity, or independently identified redundancy.',
              'new_interventions_executed':False, 'training_executed':False, 'reserved_access':False, 'st_access':False, 'shhs_access':False}
    with (BASE/'phase3_supplemental_audit_safe_v1.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__ == '__main__':
    summarize()

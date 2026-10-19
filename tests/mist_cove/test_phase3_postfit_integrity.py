"""Generated prediction grids and estimands; no real data access."""
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mist_transfer.metrics import classification_metrics


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2]/'analysis/mist_cove_phase3_v1'/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


collector = load('collect_phase3_results_v1')
summary = load('individual_attribution_summary_v1')


@pytest.fixture
def generated_grid(tmp_path):
    records = []
    rows = []
    truth, prediction, subjects, masks = [], [], [], []
    for r in range(4):
        y = np.arange(5449) % 5
        mask = np.ones(5449, dtype=bool)
        if r == 0:
            mask[:7] = False; y[:7] = -1
        record = SimpleNamespace(recording_id=f'generated_record_{r}', subject=f'generated_person_{r}', y=y, scoring_mask=mask)
        records.append(record)
        for i in range(5449):
            p = i % 5
            rows.append({'recording_id': record.recording_id, 'subject_id': record.subject, 'original_epoch_index': str(i),
                         'start_seconds': str(i*30), 'scored': str(bool(mask[i])), 'truth': str(y[i]) if mask[i] else '', 'prediction': str(p)})
            truth.append(y[i]); prediction.append(p); subjects.append(record.subject); masks.append(mask[i])
    path = tmp_path/'generated_grid.csv'
    def write(values):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(values)
    write(rows)
    return path, records, classification_metrics(truth, prediction, subjects, masks), rows, write


def test_complete_physical_grid_before_metric_mask(generated_grid):
    path, records, metrics, _, _ = generated_grid
    result = collector.verify_prediction_grid(path, records, metrics)
    assert result['physical_predictions'] == 21796 and result['scored_predictions'] == 21789


@pytest.mark.parametrize('field,value', [('subject_id','wrong_person'), ('original_epoch_index','1'), ('scored','True'), ('truth','0')])
def test_rejects_changed_source_grid_or_scoring_contract(generated_grid, field, value):
    path, records, metrics, rows, write = generated_grid
    rows[0][field] = value; write(rows)
    with pytest.raises(AssertionError):
        collector.verify_prediction_grid(path, records, metrics)


def test_rejects_unscored_grid_compression(generated_grid):
    path, records, metrics, rows, write = generated_grid
    write(rows[7:])
    with pytest.raises(AssertionError):
        collector.verify_prediction_grid(path, records, metrics)


def test_nonidentical_estimands_remain_visible():
    result = summary.paired_summary([1,2,3], [2,4,6])
    assert result['pearson_correlation'] == 1 and result['mean_absolute_difference'] == 2


def test_constant_or_empty_attribution_has_no_fabricated_correlation():
    assert summary.paired_summary([], [])['pearson_correlation'] is None
    assert summary.paired_summary([0,0], [1,2])['pearson_correlation'] is None
    assert summary.paired_summary([0,0], [1,2])['nonzero_sign_agreement'] is None


def test_nonfinite_attribution_rejected():
    with pytest.raises(ValueError):
        summary.paired_summary([1], [float('nan')])

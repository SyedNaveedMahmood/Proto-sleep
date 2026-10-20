"""Physical supports derived from the historical unfold geometry."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mist_evidence.model import ModelConfig, SUMMARY_NAMES, STAGES

ROOT = Path(__file__).resolve().parents[3]
BASE = 'ac611572c2efea1acfed48465fd482e6715ec57e'
ATOM_SAMPLES = 200
ATOMS = 15


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_config(cfg):
    if not isinstance(cfg, ModelConfig) or not cfg.raw_control or cfg.summary_only:
        raise ValueError('only historical raw_context supported')
    if cfg.radius != 0 or cfg.use_crf or cfg.samples != 3000 or cfg.fs != 100:
        raise ValueError('epoch-local 3000-sample 100Hz, no CRF required')


def interval_atoms(start, stop):
    if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= 3000:
        raise ValueError('bounded half-open physical support required')
    return list(range(start // ATOM_SAMPLES, (stop - 1) // ATOM_SAMPLES + 1))


def support_map(cfg):
    validate_config(cfg)
    scales = []
    offset = 0
    for length in cfg.scales:
        windows = [{'index': i, 'start': start, 'stop': start + length,
                    'atoms': interval_atoms(start, start + length)}
                   for i, start in enumerate(range(0, cfg.samples - length + 1, cfg.stride))]
        scales.append({'samples': length, 'stride': cfg.stride, 'windows': windows,
                       'feature_start': offset, 'feature_stop': offset + 2 * cfg.embedding_dim,
                       'aggregation_order': ['mean', 'max'], 'embedding_dim': cfg.embedding_dim})
        offset += 2 * cfg.embedding_dim
    globals_ = []
    if cfg.use_summary:
        for name in SUMMARY_NAMES:
            globals_.append({'name': name, 'feature_index': offset, 'atoms': list(range(ATOMS)),
                             'support': [0, cfg.samples], 'reason': 'native whole-epoch summary'})
            offset += 1
        if cfg.use_amplitude:
            globals_.append({'name': 'log_rms_train_scaled_sigmoid', 'feature_index': offset,
                             'atoms': list(range(ATOMS)), 'support': [0, cfg.samples],
                             'reason': 'whole-epoch centered RMS and fixed TRAIN amplitude buffer'})
            offset += 1
    graph = {'version': 'source-physical-support-v1', 'config': asdict(cfg), 'atoms': ATOMS,
             'atom_samples': ATOM_SAMPLES, 'scales': scales, 'global_features': globals_,
             'feature_count': offset, 'stage_order': list(STAGES),
             'normalization': 'fixed TRAIN affine before input; local centering/RMS inside each window',
             'local_buffer_dependency': 'encoder parameters and GroupNorm affine; bank waveform shape only',
             'bank_values_used_by_raw_path': False,
             'head': 'RawTemporal radius0: project, residual pointwise Conv1d/GELU/LayerNorm twice, classifier; no CRF'}
    graph['identity'] = digest(graph)
    return graph

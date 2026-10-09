"""Study-aware identities, annotation grids, waveform alignment and split guards."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import numpy as np

STAGE_MAP = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
             'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
             'Sleep stage ?': -1, 'Movement time': -1, 'Sleep stage M': -1}
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']


def identity(name: str, kind='npz'):
    suffix = {'npz': '.npz', 'psg': '-PSG.edf', 'hyp': '-Hypnogram.edf'}[kind]
    if not name.endswith(suffix):
        raise ValueError(f'unrecognized recording filename: {name}')
    stem = name[:-len(suffix)]
    match = re.fullmatch(r'(SC4|ST7)(\d{2})([12])([EFGJ])([A-Z0-9])', stem)
    if not match:
        raise ValueError(f'ambiguous subject/night identity: {name}')
    prefix, sid, night, equipment, scorer = match.groups()
    study = 'SC' if prefix == 'SC4' else 'ST'
    if equipment not in ({'E','F','G'} if study == 'SC' else {'J'}) or (kind != 'hyp' and scorer != '0'):
        raise ValueError(f'conflicting acquisition/recording identity: {name}')
    return {'study': study, 'subject_id': f'{study}:{sid}', 'night': int(night),
            'pair_id': stem[:7], 'recording_id': stem[:7]+'0', 'recording_variant': equipment}


def validate_splits(records):
    """Before opening any file, reject subject/recording/hash leakage across roles."""
    seen = {key: {} for key in ('subject_id', 'recording_id', 'source_sha256', 'signal_sha256', 'shape_sha256')}
    paths = set()
    for row in records:
        if row.get('split') not in {'train','val','test'}:
            raise ValueError('explicit train/val/test role required')
        if not row.get('subject_id') or not row.get('recording_id'):
            raise ValueError('canonical subject and recording identities required')
        if row.get('path'):
            path = str(Path(row['path']).expanduser().resolve())
            if path in paths:
                raise ValueError('duplicate recording path')
            paths.add(path)
        for key, values in seen.items():
            value = row.get(key)
            if value:
                if value in values and values[value] != row['split']:
                    raise ValueError(f'leakage: {key} {value} crosses splits')
                values[value] = row['split']


def open_development_npz(row):
    if row.get('split') not in {'train','val'}:
        raise ValueError('reserved test opening is prohibited')
    return np.load(row['path'], allow_pickle=False)


def validate_indices(indices, n):
    a = np.asarray(indices)
    if a.ndim != 1 or len(a) != n or not np.isfinite(a).all() or not np.equal(a, a.astype(np.int64)).all():
        raise ValueError('invalid original epoch indices')
    if np.any(a < 0) or np.any(np.diff(a) <= 0):
        raise ValueError('non-monotone original epoch indices')
    return a.astype(np.int64)


def index_runs(indices):
    """Lossless array -> original-index mapping, preserving every discontinuity."""
    a = validate_indices(indices, len(indices))
    if not len(a):
        return []
    bounds = np.r_[0, np.flatnonzero(np.diff(a) != 1)+1, len(a)]
    return [{'array_start': int(lo), 'array_stop': int(hi), 'original_start': int(a[lo]),
             'original_stop': int(a[hi-1]+1)} for lo, hi in zip(bounds[:-1], bounds[1:])]


def annotation_grid(onsets, durations, descriptions, duration_seconds):
    """Original PSG-relative 30-s grid, with unscored time and excluded tails."""
    if not (len(onsets) == len(durations) == len(descriptions)):
        raise ValueError('annotation lengths disagree')
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('invalid recording duration')
    n = int(np.floor(duration_seconds/30 + 1e-8))
    labels = np.full(n, -1, dtype=np.int64)
    reasons = np.full(n, 'unannotated', dtype=object)
    previous_end = 0.0
    raw_counts, tails, nonstage = {}, [], []
    for onset, duration, text in zip(onsets, durations, descriptions):
        onset, duration, text = float(onset), float(duration), str(text)
        if text not in STAGE_MAP:
            if text.startswith('Sleep stage') or 'Movement' in text:
                raise ValueError(f'unknown stage mapping: {text}')
            nonstage.append(text)
            continue
        if not np.isfinite([onset,duration]).all() or onset < 0 or duration <= 0:
            raise ValueError('invalid annotation interval')
        if onset < previous_end - 1e-6:
            raise ValueError('non-monotone or overlapping stage annotations')
        previous_end = onset + duration
        if abs(onset/30-round(onset/30)) > 1e-6 or abs(duration/30-round(duration/30)) > 1e-6:
            raise ValueError('annotation not on original 30-s grid')
        lo, hi = int(round(onset/30)), int(round((onset+duration)/30))
        raw_counts[text] = raw_counts.get(text, 0)+max(0, min(hi,n)-min(lo,n))
        if hi > n:
            tails.append({'description': text, 'start_seconds': max(onset, n*30),
                          'stop_seconds': onset+duration, 'reason': 'outside_full_signal_epochs'})
        lo, hi = min(lo,n), min(hi,n)
        labels[lo:hi] = STAGE_MAP[text]
        reasons[lo:hi] = '' if STAGE_MAP[text] >= 0 else ('movement' if 'M' in text else 'unknown')
    exclusions = {reason: index_runs(np.flatnonzero(reasons == reason)) for reason in sorted(set(reasons)-{''})}
    return labels, {'n_original_full_epochs': n, 'raw_stage_counts': raw_counts,
                    'excluded_epoch_ranges': exclusions, 'annotation_tails': tails,
                    'partial_signal_tail_seconds': float(duration_seconds-n*30),
                    'nonstage_annotations': sorted(set(nonstage))}


def calibration(header):
    if not header.get('dimension', '').strip() or header['dimension'] not in {'uV','µV','μV','mV','V'}:
        raise ValueError('missing or unsupported physical EEG units')
    values = [header[k] for k in ['physical_min','physical_max','digital_min','digital_max','sample_frequency']]
    if not np.isfinite(values).all() or values[1] <= values[0] or values[3] <= values[2] or values[4] <= 0:
        raise ValueError('invalid physical calibration')


def affine_match(query, source, tolerance=2e-6):
    q, s = np.asarray(query, dtype=np.float64), np.asarray(source, dtype=np.float64)
    if q.shape != s.shape or not np.isfinite(q).all() or not np.isfinite(s).all():
        return None
    q0, s0 = q-q.mean(), s-s.mean()
    qnorm, snorm = np.linalg.norm(q0), np.linalg.norm(s0)
    if min(qnorm, snorm) < 1e-10:
        return None
    gain = float(np.dot(q0, s0)/snorm**2)
    offset = float(q.mean()-gain*s.mean())
    residual = float(np.linalg.norm(q-(gain*s+offset))/qnorm)
    if gain <= 0 or residual > tolerance:
        return None
    return {'gain_npz_per_raw_unit': gain, 'offset_npz_units': offset,
            'relative_residual': residual}


def shape_features(epochs):
    f = np.asarray(epochs[:, ::100], dtype=np.float64)
    f = f-f.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return f / np.maximum(norm, 1e-12)


def shape_key(features):
    return hashlib.sha256(np.round(features, 4).astype('<f4').tobytes()).hexdigest()


def align_epochs(query, raw_epochs):
    """Align by waveform, never by labels, then validate every 3000-sample match.

    Quantized shape sketches propose candidates; positive affine full-waveform
    residual <=2e-6 certifies them. Hash-bin boundary misses get a nearest-shape
    search. More than one valid source epoch is ambiguous and is rejected.
    """
    if query.ndim != 2 or raw_epochs.ndim != 2 or query.shape[1] != raw_epochs.shape[1]:
        raise ValueError('waveform length disagreement')
    rf, qf = shape_features(raw_epochs), shape_features(query)
    lookup = {}
    for i, f in enumerate(rf):
        lookup.setdefault(shape_key(f), []).append(i)
    indices, fits = [], []
    for q, f in zip(query, qf):
        candidates = lookup.get(shape_key(f), [])
        # Also include near-identical signatures across quantization bins, to
        # detect ambiguous duplicate waveforms rather than only the first hash.
        distances = None
        if not candidates:
            distances = np.square(rf-f).sum(axis=1)
            candidates = np.flatnonzero(distances <= max(1e-8, float(distances.min())+1e-12)).tolist()
        else:
            distances = np.square(rf-f).sum(axis=1)
            candidates = sorted(set(candidates) | set(np.flatnonzero(distances < 1e-8).tolist()))
        matches = [(i, fit) for i in candidates if (fit := affine_match(q, raw_epochs[i])) is not None]
        if len(matches) != 1:
            raise ValueError(f'waveform alignment {"ambiguous" if matches else "unverified"}: {len(matches)} matches')
        indices.append(matches[0][0]); fits.append(matches[0][1])
    idx = validate_indices(indices, len(query))
    return idx, {'n_verified_epochs': len(idx),
                 'max_relative_residual': max(f['relative_residual'] for f in fits),
                 'gain_min': min(f['gain_npz_per_raw_unit'] for f in fits),
                 'gain_max': max(f['gain_npz_per_raw_unit'] for f in fits),
                 'offset_min': min(f['offset_npz_units'] for f in fits),
                 'offset_max': max(f['offset_npz_units'] for f in fits),
                 'operator': 'positive affine (gain + offset), no filtering/resampling/warping',
                 'tolerance': 2e-6}

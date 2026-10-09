"""Strict read-only Sleep-EDF annotation-only EDF+ compatibility validation.

Narrow profile: one zero-duration data record, one EDF Annotations signal.
An ST previous-day Recordingfield is accepted only when the paired PSG has
identical fixed clock, patient/recording identification fields and canonical
night ID, and independently decoded TALs agree. Original bytes are never fixed.
Spec: https://www.edfplus.info/specs/edfplus.html sections 2.1.3, 2.2.
"""
from __future__ import annotations
import datetime as dt
import hashlib
from pathlib import Path
import re
import warnings
import numpy as np
import mne
import pyedflib
from .integrity import STAGE_MAP, annotation_grid, identity

MONTHS = dict(zip('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'.split(), range(1,13)))
NUMBER = re.compile(rb'[+-]\d+(?:\.\d+)?')
DURATION = re.compile(rb'\d+(?:\.\d+)?')


def fixed_header(path):
    with Path(path).open('rb') as f:
        b = f.read(256)
    if len(b) != 256 or b[:8] != b'0       ' or any(x < 32 or x > 126 for x in b):
        raise ValueError('invalid printable EDF fixed header')
    if not re.fullmatch(rb'\d{2}\.\d{2}\.\d{2}', b[168:176]) or not re.fullmatch(rb'\d{2}\.\d{2}\.\d{2}', b[176:184]):
        raise ValueError('invalid fixed date/time')
    day, month, year = map(int, b[168:176].split(b'.'))
    hour, minute, second = map(int, b[176:184].split(b'.'))
    start = dt.datetime(1900+year if year >= 85 else 2000+year, month, day, hour, minute, second)
    return b, start


def recording_date(header):
    fields = header[88:168].decode('ascii').strip().split()
    if len(fields) < 5 or fields[0] != 'Startdate':
        raise ValueError('unsupported EDF+ Recordingfield layout')
    if fields[1] == 'X':
        return None
    if not re.fullmatch(r'\d{2}-[A-Z]{3}-\d{4}', fields[1]):
        raise ValueError('invalid Recordingfield date')
    day, month, year = fields[1].split('-')
    if month not in MONTHS:
        raise ValueError('invalid Recordingfield month')
    return dt.date(int(year), MONTHS[month], int(day))


def parse_tals(payload):
    """Consume every byte; reject regex-skipped garbage, truncation or hidden TALs."""
    offset, entries, first = 0, [], True
    while offset < len(payload):
        if payload[offset] == 0:
            if any(payload[offset:]):
                raise ValueError('nonzero bytes after annotation padding')
            break
        end = payload.find(b'\x00', offset)
        if end < 0:
            raise ValueError('unterminated TAL')
        tal = payload[offset:end]
        offset = end+1
        if not tal.endswith(b'\x14') or b'\x14' not in tal:
            raise ValueError('invalid TAL separators')
        timing, descriptions = tal.split(b'\x14',1)
        parts = timing.split(b'\x15')
        if len(parts) not in {1,2} or not NUMBER.fullmatch(parts[0]):
            raise ValueError('invalid TAL onset')
        onset = float(parts[0])
        if not np.isfinite(onset):
            raise ValueError('nonfinite TAL onset')
        if first:
            if onset != 0 or not parts[0].startswith(b'+') or len(parts) != 1 or descriptions != b'\x14':
                raise ValueError('first TAL must be empty timekeeping at +0')
            first = False
            continue
        if len(parts) != 2 or not DURATION.fullmatch(parts[1]):
            raise ValueError('stage TAL requires unsigned duration')
        duration = float(parts[1])
        if not np.isfinite(duration) or duration <= 0 or onset < 0:
            raise ValueError('invalid stage TAL interval')
        texts = descriptions[:-1].split(b'\x14')
        if len(texts) != 1:
            raise ValueError('Sleep-EDF profile requires one stage per TAL')
        text = texts[0].decode('utf-8', errors='strict')
        if text not in STAGE_MAP:
            raise ValueError('unknown or nonstage annotation in Sleep-EDF hypnogram')
        entries.append((onset,duration,text))
    if first or not entries:
        raise ValueError('missing timekeeping or sleep annotations')
    return entries


def read_annotation_profile(path):
    """Decode bounded annotation-only bytes; date conflicts remain unresolved here."""
    path = Path(path)
    if path.stat().st_size > (1 << 20):
        raise ValueError('outside bounded Sleep-EDF annotation profile')
    header, start = fixed_header(path)
    b = path.read_bytes()
    if int(header[184:192]) != 512 or int(header[236:244]) != 1 or int(header[252:256]) != 1:
        raise ValueError('requires one annotation signal and one data record')
    if float(header[244:252]) != 0 or header[192:236].strip() != b'EDF+C':
        raise ValueError('requires contiguous zero-duration annotation-only record')
    if len(b) < 512 or any(x < 32 or x > 126 for x in b[256:512]):
        raise ValueError('invalid annotation signal header')
    sh = b[256:512]
    if sh[:16].strip() != b'EDF Annotations':
        raise ValueError('not an EDF Annotations signal')
    if int(sh[120:128]) != -32768 or int(sh[128:136]) != 32767:
        raise ValueError('invalid annotation digital bounds')
    pmin, pmax = float(sh[104:112]), float(sh[112:120])
    if not np.isfinite([pmin,pmax]).all() or pmin == pmax:
        raise ValueError('invalid annotation physical bounds')
    if any(sh[a:z].strip() for a,z in [(16,104),(136,216),(224,256)]):
        raise ValueError('unsupported annotation transducer/units/prefilter/reserved fields')
    samples = int(sh[216:224])
    if samples <= 0 or len(b) != 512+2*samples:
        raise ValueError('annotation payload size disagrees with header')
    entries = parse_tals(b[512:])
    onsets = np.array([x[0] for x in entries]); durations = np.array([x[1] for x in entries])
    texts = np.array([x[2] for x in entries])
    meta = {'fixed_start_datetime':start.isoformat(),
            'recording_date':recording_date(header).isoformat() if recording_date(header) else None,
            'payload_sha256':hashlib.sha256(b[512:]).hexdigest(),
            'profile':'one EDF+C annotation signal / one zero-duration record / full byte consumption',
            'n_intervals':len(entries),'original_bytes_modified':False}
    return header, start, (onsets,durations,texts), meta


def paired_annotations(hyp_path, psg_path, duration_seconds):
    """Validate profile, clock/identity pairing and exact independent-reader agreement.

    Any exception must lead to explicit quarantine in the calling audit. Reader
    agreement alone never excuses malformed bytes or an ambiguous 24-hour origin.
    """
    h_id, p_id = identity(Path(hyp_path).name,'hyp'), identity(Path(psg_path).name,'psg')
    if h_id['recording_id'] != p_id['recording_id']:
        raise ValueError('canonical PSG/hypnogram pairing disagreement')
    h, start, arrays, meta = read_annotation_profile(hyp_path)
    p, pstart = fixed_header(psg_path)
    meta['original_header_evidence'] = {
        kind:{'patient_field':header[8:88].decode().strip(),
              'recording_field':header[88:168].decode().strip(),
              'fixed_date':header[168:176].decode(),'fixed_time':header[176:184].decode()}
        for kind,header in [('hypnogram',h),('psg',p)]}
    if start != pstart or h[8:88] != p[8:88]:
        raise ValueError('paired fixed clock or patient/recording fields disagree')
    hrec,prec = h[88:168].decode().strip().split(),p[88:168].decode().strip().split()
    if hrec[:4] != prec[:4]:
        raise ValueError('paired core recording identification fields disagree')
    meta['recording_equipment_field_difference'] = hrec[4:] != prec[4:]
    meta['recording_equipment_note'] = 'derived annotation and PSG equipment/program fields differ; canonical ID, date, investigation/technician fields and patient field agree' if hrec[4:] != prec[4:] else None
    p_duration = int(p[236:244])*float(p[244:252])
    if not np.isfinite([p_duration,duration_seconds]).all() or p_duration <= 0 or abs(p_duration-duration_seconds) > 1e-8:
        raise ValueError('paired PSG duration disagrees with audited recording')
    date = recording_date(h)
    delta = (date-start.date()).days if date else None
    if delta not in {None,0} and not (h_id['study']=='ST' and delta==-1):
        raise ValueError('unresolved Recordingfield/fixed-date discrepancy')
    if delta==-1 and h[88:168]!=p[88:168]:
        raise ValueError('midnight compatibility requires identical recording fields')
    meta.update(recording_date_delta_days=delta, pairing='canonical ID + equal fixed clock + equal patient/recording fields',
                timing_origin='paired fixed EDF date/time + TAL onset; no 24-hour adjustment',
                header_exception='ST previous-day Recordingfield corroborated by paired PSG' if delta==-1 else None)
    onsets,durations,texts = arrays
    with warnings.catch_warnings(record=True) as caught:
        ann = mne.read_annotations(hyp_path)
    if caught:
        raise ValueError('independent-reader warnings require review: '+str([str(w.message) for w in caught]))
    if not (np.array_equal(onsets,ann.onset) and np.array_equal(durations,ann.duration) and np.array_equal(texts,ann.description)):
        raise ValueError('strict TAL/MNE annotation disagreement')
    meta['mne_comparison'] = {'status':'VERIFIED','orig_time':str(ann.orig_time),
                              'warnings':[str(w.message) for w in caught]}
    strict_error = None
    try:
        with pyedflib.EdfReader(str(hyp_path)) as reader:
            po,pd,pt = reader.readAnnotations()
            if not (np.array_equal(onsets,po) and np.array_equal(durations,pd) and np.array_equal(texts,pt)):
                raise ValueError('strict TAL/pyEDFlib annotation disagreement')
            if reader.getStartdatetime() != start:
                raise ValueError('pyEDFlib/fixed-clock disagreement')
    except OSError as exc:
        strict_error = str(exc)
        if delta != -1 or 'EDF+ Recordingfield' not in strict_error:
            raise ValueError('unexpected strict-reader rejection: '+strict_error) from exc
    meta['pyedflib_comparison'] = {'status':'COMPATIBILITY_REQUIRED' if strict_error else 'VERIFIED',
                                   'error':strict_error}
    labels, grid = annotation_grid(onsets,durations,texts,duration_seconds)
    # Only the single physically incomplete final epoch may have a scored TAL
    # overhang. Preserve its original interval as diagnostics; never create a
    # complete signal epoch from it. Longer scored overhang is quarantined.
    scored_tails=[t for t in grid['annotation_tails'] if t['description'] not in {'Sleep stage ?','Movement time','Sleep stage M'}]
    for tail in scored_tails:
        if grid['partial_signal_tail_seconds'] <= 0 or tail['stop_seconds'] > (len(labels)+1)*30:
            raise ValueError('scored annotation extends beyond full physical signal grid')
    meta['scored_tail_status'] = 'ONE_INCOMPLETE_SIGNAL_EPOCH; diagnostic only, no inference/scoring epoch' if scored_tails else 'NONE'
    meta['annotation_signal_end_overhang_seconds'] = max(0,float(max(onsets+durations))-duration_seconds)
    return labels, {**meta, **grid}

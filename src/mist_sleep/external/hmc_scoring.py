"""Strict scoring geometry, available only after a sealed unblind claim."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pyedflib
from .protocol import specification
from .hmc_manifest import verify_file

LABELS = specification()["label_mapping"]
MARKERS = {"Lights off", "Lights on", "Recording starts", "Recording ends"}


def epoch_labels(events, epoch_count):
    if epoch_count < 1:
        raise ValueError("positive complete-epoch count required")
    labels = np.full(epoch_count, -1, dtype=np.int8)
    occupied = np.zeros(epoch_count, dtype=bool)
    for onset, duration, text in events:
        onset, duration = float(onset), float(duration)
        if not np.isfinite([onset, duration]).all() or onset < 0 or duration < 0:
            raise ValueError("invalid scoring time")
        if text in MARKERS:
            if onset > epoch_count * 30:
                raise ValueError("marker outside physical epoch coverage")
            continue
        if text not in LABELS:
            raise ValueError("unsupported scoring text")
        start, length = onset / 30, duration / 30
        if duration <= 0 or abs(start - round(start)) > 1e-8 or abs(length - round(length)) > 1e-8:
            raise ValueError("stage interval is not on the fixed 30-second grid")
        start, length = round(start), round(length)
        if start + length > epoch_count or occupied[start:start + length].any():
            raise ValueError("out-of-range, repeated or contradictory interval")
        labels[start:start + length] = LABELS[text]
        occupied[start:start + length] = True
    return labels


def csv_events(path):
    # Deliberately strict. Real HMC schema must be independently verified before use.
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["onset_seconds", "duration_seconds", "annotation"]:
            raise ValueError("unverified HMC text scoring schema; do not guess")
        return [(float(row["onset_seconds"]), float(row["duration_seconds"]), row["annotation"]) for row in reader]


def read_scoring(root, recording, files, *, epoch_count, waveform_start, claim_root):
    root = Path(root)
    claim_root = Path(claim_root)
    claim = claim_root / "UNBLIND_STARTED.json"
    if not claim.is_file() or not (claim_root / "prediction_seal.private.json").is_file():
        raise RuntimeError("sealed one-time scoring claim required before any annotation access")
    edf_name = f"recordings/{recording}_sleepscoring.edf"
    txt_name = f"recordings/{recording}_sleepscoring.txt"
    edf = verify_file(root, edf_name, files[edf_name])
    txt = verify_file(root, txt_name, files[txt_name])
    with pyedflib.EdfReader(str(edf)) as reader:
        if reader.getStartdatetime().isoformat() != waveform_start:
            raise ValueError("waveform/scoring absolute start-time mismatch")
        onsets, durations, texts = reader.readAnnotations()
        edf_events = list(zip(onsets, durations, texts))
    text_events = csv_events(txt)
    canonical = lambda rows: sorted((float(a), float(b), str(c)) for a, b, c in rows)
    if canonical(edf_events) != canonical(text_events):
        raise ValueError("EDF+/text annotation disagreement")
    return epoch_labels(edf_events, epoch_count)

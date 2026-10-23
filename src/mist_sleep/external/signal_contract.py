"""Fixed physical calibration and full-record deterministic rational resampling."""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, resample_poly
from .protocol import SOURCE_MEAN_UV, SOURCE_STD_UV, specification

FIR = firwin(1281, 1 / 64, window=("kaiser", 5.0))
FIR.setflags(write=False)
UNIT_TO_UV = {"uV": 1., "mV": 1000., "V": 1000000.}


def channel_index(labels, requested):
    if requested not in ("C4/M1", "F4/M1"):
        raise ValueError("unregistered channel")
    aliases = specification()["channel_label_aliases"][requested]
    hits = [i for i, label in enumerate(labels) if label.strip() in aliases]
    if len(hits) != 1:
        raise ValueError("missing or ambiguous exact channel")
    return hits[0]


def calibrated_microvolts(digital, physical_min, physical_max, digital_min, digital_max, unit):
    if unit not in UNIT_TO_UV or not np.isfinite([physical_min, physical_max, digital_min, digital_max]).all():
        raise ValueError("unsupported or nonfinite calibration")
    if physical_max <= physical_min or digital_max <= digital_min:
        raise ValueError("degenerate calibration")
    digital = np.asarray(digital)
    if not np.isfinite(digital).all() or np.any((digital < digital_min) | (digital > digital_max)):
        raise ValueError("invalid digital samples")
    return (physical_min + (digital.astype(np.float64) - digital_min) *
            (physical_max - physical_min) / (digital_max - digital_min)) * UNIT_TO_UV[unit]


def physical_epochs(values, *, unit, sample_rate):
    if sample_rate != 256 or unit not in UNIT_TO_UV:
        raise ValueError("rate or physical units incompatible")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 7680:
        raise ValueError("invalid physical recording")
    count = len(x) // 7680
    # Filter once across the physical record, never concatenate different records.
    y = resample_poly(x * UNIT_TO_UV[unit], 25, 64, window=FIR, padtype="constant", cval=0.)
    epochs = y[:count * 3000].reshape(count, 1, 3000)
    normalized = ((epochs - SOURCE_MEAN_UV) / SOURCE_STD_UV).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite normalized signal")
    return normalized, np.arange(count, dtype=np.int64) * 30

"""Parse only fixed and signal EDF headers; never waveform or EDF+ TAL data."""
import numpy as np


def parse_header(data):
    if len(data) < 256:
        raise ValueError("truncated EDF fixed header")
    if data[:8] != b"0       ":
        raise ValueError("unsupported EDF version")
    length, signals = int(data[184:192]), int(data[252:256])
    if not 1 <= signals <= 256 or length != 256 + 256 * signals or len(data) != length:
        raise ValueError("invalid EDF header geometry")
    if b"EDF+D" in data[192:236]:
        raise ValueError("discontinuous EDF unsupported")
    count, duration = int(data[236:244]), float(data[244:252])
    if count < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("unknown or invalid physical recording duration")
    widths = [("label", 16), ("transducer", 80), ("unit", 8), ("physical_min", 8),
              ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
              ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]
    result, offset = [{} for _ in range(signals)], 256
    for field, width in widths:
        for i in range(signals):
            result[i][field] = data[offset + i*width:offset + (i+1)*width].decode("ascii").strip()
        offset += signals * width
    for row in result:
        for field in ("physical_min", "physical_max", "digital_min", "digital_max"):
            row[field] = float(row[field])
        row["sample_rate"] = int(row["samples_per_record"]) / duration
        if not np.isfinite([row[k] for k in ("physical_min", "physical_max", "digital_min", "digital_max", "sample_rate")]).all():
            raise ValueError("nonfinite EDF signal header")
        if row["physical_max"] <= row["physical_min"] or row["digital_max"] <= row["digital_min"]:
            raise ValueError("invalid EDF physical calibration")
    # Do not return patient, recording or absolute start identifiers to public code.
    return {"signals": result, "duration_seconds": count*duration, "header_bytes": length}

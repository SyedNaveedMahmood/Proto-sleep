"""Header-only geometry and channel contract for prospective DREAMS files."""

from dataclasses import dataclass
import math


CENTRAL_EEG = {"C3-A1", "Cz-A1"}


@dataclass(frozen=True)
class ChannelHeader:
    label: str
    sample_rate: float
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int


def validate_event_channel(header: ChannelHeader, *, event_type: str,
                           expected_rate: int, duration_seconds: float,
                           sample_count: int) -> dict:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("invalid physical duration")
    if event_type in {"spindle", "k_complex"} and header.label not in CENTRAL_EEG:
        raise ValueError("central EEG derivation mismatch")
    if event_type == "rapid_eye_movement" and "EOG" not in header.label.upper():
        raise ValueError("REM event benchmark requires original EOG")
    if header.sample_rate != expected_rate or expected_rate not in (50, 100, 200):
        raise ValueError("native sample rate mismatch")
    if header.physical_dimension not in ("uV", "µV"):
        raise ValueError("unsupported physical unit")
    if (not all(map(math.isfinite, (header.physical_min, header.physical_max))) or
            header.physical_max <= header.physical_min or
            header.digital_max <= header.digital_min):
        raise ValueError("invalid EDF calibration")
    expected_samples = round(expected_rate * duration_seconds)
    if sample_count != expected_samples or abs(expected_samples / expected_rate - duration_seconds) > 1e-9:
        raise ValueError("physical sample extent mismatch")
    return {"channel": header.label, "sample_rate": expected_rate,
            "duration_seconds": duration_seconds, "sample_count": sample_count,
            "unit": "uV", "status": "HEADER_CONTRACT_ONLY"}

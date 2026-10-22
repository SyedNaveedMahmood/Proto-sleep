"""Typed DREAMS annotation parsing with explicit scorer coverage.

The four adapters follow the owner PDF's documented text layouts. They have
only been exercised on generated fixtures: no DREAMS archive was acquired.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
import re


EVENT_TYPES = {"spindle", "k_complex", "rapid_eye_movement", "artifact"}
MAX_TEXT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Coverage:
    recording: str
    scorer: str
    start_seconds: Decimal
    end_seconds: Decimal
    channel: str
    event_type: str

    def __post_init__(self):
        if not 0 <= self.start_seconds < self.end_seconds:
            raise ValueError("invalid annotation coverage")
        if self.event_type not in EVENT_TYPES or not self.scorer or not self.channel:
            raise ValueError("incomplete coverage identity")


@dataclass(frozen=True)
class Event:
    dataset: str
    recording: str
    participant: str
    channel: str
    event_type: str
    start_seconds: Decimal
    end_seconds: Decimal
    start_sample: int
    end_sample: int
    sample_rate: int
    scorer: str
    coverage: Coverage
    source_sha256: str
    original_category: str | None = None
    confidence: float | None = None
    uncertain: bool = False
    provenance: str = "owner_documented_visual_scoring"


def _time(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("non-numeric annotation time") from exc
    if not result.is_finite():
        raise ValueError("non-finite annotation time")
    return result


def _sample(second: Decimal, sample_rate: int) -> int:
    return int((second * sample_rate).to_integral_value(rounding=ROUND_HALF_UP))


def _parse(text: bytes, *, kind: str, recording: str, participant: str,
           channel: str, scorer: str, sample_rate: int,
           duration_seconds: Decimal, coverages: tuple[Coverage, ...]) -> list[Event]:
    if kind not in EVENT_TYPES or not recording or not participant or not scorer:
        raise ValueError("incomplete event source identity")
    if not 1 <= sample_rate <= 4096 or duration_seconds <= 0:
        raise ValueError("invalid recording geometry")
    if len(text) > MAX_TEXT_BYTES or b"\x00" in text:
        raise ValueError("annotation text invalid or oversize")
    try:
        lines = text.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("unsupported annotation encoding") from exc
    digest = sha256(text).hexdigest()
    result: list[Event] = []
    category = None
    header_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            if kind == "artifact":
                category = stripped[1:-1]
                if (not category.startswith("vis_") or "/" not in category or
                        category.split("/", 1)[1].casefold() != channel.casefold()):
                    raise ValueError("artifact category/channel header invalid")
            elif header_seen:
                raise ValueError("multiple event headers")
            header_seen = True
            continue
        fields = re.split(r"\s+", stripped)
        if len(fields) != 2 or not all(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", x) for x in fields):
            if header_seen or kind == "artifact":
                raise ValueError("malformed annotation row")
            # Non-artifact owner files have one free-text detection header.
            header_seen = True
            continue
        if not header_seen:
            raise ValueError("annotation detection header missing")
        if kind == "artifact" and category is None:
            raise ValueError("artifact row has no category")
        start, length = map(_time, fields)
        end = start + length
        if start < 0 or length <= 0 or end > duration_seconds:
            raise ValueError("event outside physical excerpt")
        applicable = [c for c in coverages if c.recording == recording and c.scorer == scorer
                      and c.event_type == kind and c.channel == channel
                      and c.start_seconds <= start and end <= c.end_seconds]
        if len(applicable) != 1:
            raise ValueError("event lacks unique verified scorer coverage")
        first, last = _sample(start, sample_rate), _sample(end, sample_rate)
        if last <= first or last > _sample(duration_seconds, sample_rate):
            raise ValueError("event collapses or exceeds sample grid")
        result.append(Event("DREAMS", recording, participant, channel, kind,
                            start, end, first, last, sample_rate, scorer,
                            applicable[0], digest, category))
    if not header_seen:
        raise ValueError("annotation header missing")
    return result


def parse_spindles(text: bytes, **kwargs) -> list[Event]:
    return _parse(text, kind="spindle", **kwargs)


def parse_kcomplexes(text: bytes, **kwargs) -> list[Event]:
    return _parse(text, kind="k_complex", **kwargs)


def parse_rems(text: bytes, **kwargs) -> list[Event]:
    return _parse(text, kind="rapid_eye_movement", **kwargs)


def parse_artifacts(text: bytes, **kwargs) -> list[Event]:
    return _parse(text, kind="artifact", **kwargs)

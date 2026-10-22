"""Coverage-aware interval decisions and exact one-to-one event matching."""

from decimal import Decimal
from scipy.optimize import linear_sum_assignment
import numpy as np

from .annotations import Coverage, Event


UNKNOWN = "UNKNOWN_ANNOTATION_COVERAGE"
COVERED_NEGATIVE = "COVERED_NO_EVENT"
POSITIVE = "COVERED_EVENT"


def joint_coverage(first: list[Coverage], second: list[Coverage]) -> list[tuple[Decimal, Decimal]]:
    """Intersect coverage from two genuine scorers of the same task/channel."""
    if not first or not second:
        return []
    result = []
    for a in first:
        for b in second:
            if (a.scorer == b.scorer or (a.recording, a.channel, a.event_type) !=
                    (b.recording, b.channel, b.event_type)):
                raise ValueError("scorer/task mismatch")
            left, right = max(a.start_seconds, b.start_seconds), min(a.end_seconds, b.end_seconds)
            if left < right:
                result.append((left, right))
    return _merge(result)


def _merge(intervals):
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def label_interval(start: Decimal, end: Decimal, events: list[Event],
                   coverage: list[tuple[Decimal, Decimal]]) -> str:
    if not 0 <= start < end:
        raise ValueError("invalid interval")
    if not any(left <= start and end <= right for left, right in _merge(coverage)):
        return UNKNOWN
    if any(e.start_seconds < end and e.end_seconds > start for e in events):
        return POSITIVE
    return COVERED_NEGATIVE


def _iou(a, b):
    overlap = max(Decimal(0), min(a.end_seconds, b.end_seconds) - max(a.start_seconds, b.start_seconds))
    union = max(a.end_seconds, b.end_seconds) - min(a.start_seconds, b.start_seconds)
    return float(overlap / union)


def match_events(first: list[Event], second: list[Event], *, min_iou: float = 0.2) -> dict:
    """Maximize matched count, then IoU, with one-to-one assignments.

    Caller must restrict both lists to the same verified joint coverage first.
    The function rejects unmatched scorer/task/recording identities.
    """
    if not 0 < min_iou <= 1:
        raise ValueError("invalid IoU threshold")
    if first and second:
        a, b = first[0], second[0]
        if ((a.dataset, a.recording, a.channel, a.event_type) !=
                (b.dataset, b.recording, b.channel, b.event_type) or a.scorer == b.scorer):
            raise ValueError("different recording, event task, or same scorer")
    for side in (first, second):
        if side and any((e.dataset, e.recording, e.channel, e.event_type, e.scorer) !=
                        (side[0].dataset, side[0].recording, side[0].channel,
                         side[0].event_type, side[0].scorer) for e in side):
            raise ValueError("mixed event identities")
    n, m = len(first), len(second)
    if n > 2000 or m > 2000:
        raise ValueError("event matching exceeds bounded matrix size")
    if not n or not m:
        return {"first": n, "second": m, "matched": 0,
                "precision": 0.0 if m else None, "recall": 0.0 if n else None,
                "f1": 0.0 if n + m else None, "mean_iou": None,
                "min_iou": min_iou, "matching": "maximum-cardinality-then-IoU",
                "pairs": []}
    values = np.full((n, m + n), -1e9, dtype=float)
    values[:, m:] = 0
    ious = np.zeros((n, m), dtype=float)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            iou = _iou(a, b)
            ious[i, j] = iou
            if iou >= min_iou:
                values[i, j] = 1_000_000 + iou
    rows, cols = linear_sum_assignment(values, maximize=True)
    pairs = [(i, j) for i, j in zip(rows, cols) if j < m and values[i, j] > 0]
    k = len(pairs)
    return {"first": n, "second": m, "matched": k, "precision": k / m,
            "recall": k / n, "f1": 2 * k / (n + m),
            "mean_iou": float(np.mean([ious[i, j] for i, j in pairs])) if pairs else None,
            "min_iou": min_iou, "matching": "maximum-cardinality-then-IoU",
            "pairs": [[int(i), int(j)] for i, j in pairs]}


def enforce_participant_split(train: dict[str, str], test: dict[str, str]) -> None:
    """Maps original recording IDs to verified participant IDs."""
    if not train or not test or set(train) & set(test):
        raise ValueError("empty or overlapping recordings")
    if set(train.values()) & set(test.values()):
        raise ValueError("participant leakage between development and test")
    if not all(train.values()) or not all(test.values()):
        raise ValueError("unknown participant identity")

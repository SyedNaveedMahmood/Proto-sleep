"""Reviewer-specific and operational union/intersection event references."""

from dataclasses import dataclass
from decimal import Decimal

from .annotations import Event
from .benchmark import match_events


@dataclass(frozen=True)
class ReferenceInterval:
    start: Decimal
    end: Decimal
    definition: str
    source_scorers: tuple[str, ...]


def reviewer_events(events: list[Event], scorer: str) -> list[ReferenceInterval]:
    return [ReferenceInterval(e.start_seconds, e.end_seconds, f"reviewer:{scorer}", (scorer,))
            for e in events if e.scorer == scorer]


def _require_same_task(first: list[Event], second: list[Event]) -> None:
    rows = first + second
    if not rows:
        return
    identity = (rows[0].dataset, rows[0].recording, rows[0].channel, rows[0].event_type)
    if any((e.dataset, e.recording, e.channel, e.event_type) != identity for e in rows):
        raise ValueError("mixed reference dataset/recording/channel/task")
    if first and second and first[0].scorer == second[0].scorer:
        raise ValueError("reference comparison requires independent scorers")


def union_events(first: list[Event], second: list[Event]) -> list[ReferenceInterval]:
    """Merge overlapping reviewer intervals; not an adjudicated truth."""
    _require_same_task(first, second)
    rows = sorted([(e.start_seconds, e.end_seconds, e.scorer) for e in first + second])
    result: list[ReferenceInterval] = []
    for start, end, scorer in rows:
        if result and start <= result[-1].end:
            old = result.pop()
            result.append(ReferenceInterval(old.start, max(end, old.end), "union",
                                            tuple(sorted(set(old.source_scorers + (scorer,))))))
        else:
            result.append(ReferenceInterval(start, end, "union", (scorer,)))
    return result


def intersection_events(first: list[Event], second: list[Event],
                        *, min_iou: float = 0.2) -> list[ReferenceInterval]:
    """Overlapping extent for one-to-one reviewer pairs meeting fixed IoU.

    The interval is an operational overlap, not an adjudicated event duration.
    """
    if not 0 < min_iou <= 1:
        raise ValueError("invalid IoU")
    _require_same_task(first, second)
    result = []
    for i, j in match_events(first, second, min_iou=min_iou)["pairs"]:
        a, b = first[i], second[j]
        result.append(ReferenceInterval(max(a.start_seconds, b.start_seconds),
                                        min(a.end_seconds, b.end_seconds),
                                        "matched_intersection", (a.scorer, b.scorer)))
    return sorted(result, key=lambda row: row.start)

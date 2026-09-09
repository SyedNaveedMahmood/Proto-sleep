"""Bank diagnostics and event scoring. Neither assigns clinical event labels."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import write_json
from .train import model_from_payload, safe_load


@torch.no_grad()
def audit_bank(checkpoint: Path, output: Path):
    payload = safe_load(checkpoint)
    if payload["arm"] != "evidence":
        raise ValueError("Bank audit requires the evidence arm")
    model = model_from_payload(payload).eval()
    scales = []
    for si, length in enumerate(model.cfg.scales_samples):
        z = model.encoder(getattr(model, f"anchors_{si}"))
        cosine = z @ z.T
        mask = ~torch.eye(len(z), dtype=torch.bool)
        pairs = cosine[torch.triu(mask, diagonal=1)]
        refs = model.refs[si]
        scales.append({"scale_samples": length, "anchor_count": len(refs),
                       "distinct_source_subjects": len({r["subject"] for r in refs}),
                       "anchors_by_subject": dict(Counter(r["subject"] for r in refs)),
                       "max_distinct_embedding_cosine": float(cosine[mask].max()),
                       "pairs_ge_099": int((pairs >= .99).sum()),
                       "clinical_events_verified": 0})
    result = {"scales": scales, "test_arrays_opened": False,
              "warning": "Real waveform anchors do not guarantee distinct embeddings or verified clinical identities. High cosine is a diagnostic, not an automatic proof of failure."}
    write_json(output, result)
    return result


def event_scores(predictions: list[dict], reference: list[dict], iou_threshold: float):
    """One-to-one interval matching, same recording and event category only.

    Inputs: dicts with recording, event, start_s, end_s. Event categories must be
    independently established before evaluation. Choose IoU thresholds on development
    data, not after seeing event-test performance. Returns micro event precision/recall/F1.
    Maximizes valid match cardinality first, then total IoU; prevents greedy double credit.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    for rows in (predictions, reference):
        for row in rows:
            values = np.asarray([row["start_s"], row["end_s"]], dtype=float)
            if not np.isfinite(values).all() or not 0 <= values[0] < values[1]:
                raise ValueError("Invalid interval")
            if not row["recording"] or not row["event"] or row["event"] == "unreviewed":
                raise ValueError("Cannot score unreviewed or missing event identities")
    groups = {(r["recording"], r["event"]) for r in predictions + reference}
    tp = 0
    for recording, event in groups:
        p = [r for r in predictions if (r["recording"], r["event"]) == (recording, event)]
        g = [r for r in reference if (r["recording"], r["event"]) == (recording, event)]
        if not p or not g:
            continue
        a = np.asarray([[r["start_s"], r["end_s"]] for r in p])
        b = np.asarray([[r["start_s"], r["end_s"]] for r in g])
        intersection = np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]))
        union = (a[:, 1]-a[:, 0])[:, None] + (b[:, 1]-b[:, 0])[None, :] - intersection
        iou = intersection / union
        valid = iou >= iou_threshold
        utility = valid.astype(float) + np.where(valid, iou / (min(len(p), len(g))+1), 0)
        ri, ci = linear_sum_assignment(-utility)
        tp += int(valid[ri, ci].sum())
    fp, fn = len(predictions)-tp, len(reference)-tp
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp/max(1, tp+fp),
            "recall": tp/max(1, tp+fn), "f1": 2*tp/max(1, 2*tp+fp+fn),
            "iou_threshold": iou_threshold, "clinical_validity_inferred": False}

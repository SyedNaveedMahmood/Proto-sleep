"""Fixed-five recording metrics and evaluation-only confidence diagnostics."""
from __future__ import annotations

import numpy as np


def stage_metrics(y, logits):
    y, logits = np.asarray(y), np.asarray(logits, dtype=np.float64)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not len(y) or logits.shape != (len(y), 5):
        raise ValueError("stage metric geometry")
    if not np.isfinite(logits).all() or ((y < 0) | (y > 4)).any():
        raise ValueError("invalid stages/logits")
    pred = logits.argmax(1)
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
    support, predicted = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * cm.diagonal(), support + predicted, out=np.zeros(5), where=support + predicted > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(5), where=support > 0)
    probability = np.exp(logits - logits.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    confidence = probability.max(1)
    correct = pred == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(float(np.abs(correct[bins == i].mean() - confidence[bins == i].mean()) * (bins == i).mean())
              for i in range(10) if (bins == i).any())
    sorted_p = np.sort(probability, axis=1)
    quantiles = lambda a: np.quantile(a, [0, .25, .5, .75, 1]).tolist() if len(a) else None
    return {"epochs": len(y), "macro_f1": float(f1.mean()), "f1": f1.tolist(), "recall": recall.tolist(),
            "confusion": cm.tolist(), "truth_counts": support.tolist(), "prediction_counts": predicted.tolist(),
            "stage_present": (support > 0).tolist(), "accuracy": float(correct.mean()),
            "balanced_accuracy_present_classes": float(recall[support > 0].mean()),
            "balanced_accuracy_fixed_five": float(recall.mean()),
            "nll": float(-np.log(probability[np.arange(len(y)), y].clip(1e-12)).mean()),
            "brier": float(np.square(probability - np.eye(5)[y]).sum(1).mean()), "ece_10": ece,
            "high_confidence_wrong_count": int(((confidence >= .9) & ~correct).sum()),
            "wrong_confidence_quantiles": quantiles(confidence[~correct]),
            "entropy_mean": float(-(probability * np.log(probability.clip(1e-12))).sum(1).mean()),
            "margin_quantiles": quantiles(sorted_p[:, -1] - sorted_p[:, -2]),
            "per_stage_wrong_confidence_mean": [float(confidence[(y == k) & ~correct].mean())
                                                 if ((y == k) & ~correct).any() else None for k in range(5)],
            "severe_collapse": bool(predicted.max() / len(y) >= .95)}

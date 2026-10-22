"""Fixed-five, participant-aware stage metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def fixed_five(y: np.ndarray, logits: np.ndarray, participant: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    participant = np.asarray(participant)
    if logits.shape != (len(y), 5) or len(participant) != len(y) or len(y) == 0:
        raise ValueError("fixed-five evaluation shape")
    if not np.isfinite(logits).all() or np.any((y < 0) | (y > 4)):
        raise ValueError("invalid stage or logits")
    prediction = logits.argmax(1)
    cm = confusion_matrix(y, prediction, labels=np.arange(5))
    f1 = f1_score(y, prediction, labels=np.arange(5), average=None, zero_division=0)
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(5), where=cm.sum(1) > 0)
    people = [float(f1_score(y[participant == p], prediction[participant == p],
                              labels=np.arange(5), average="macro", zero_division=0))
              for p in np.unique(participant)]
    centered = logits - logits.max(1, keepdims=True)
    prob = np.exp(centered)
    prob /= prob.sum(1, keepdims=True)
    nll = -np.log(prob[np.arange(len(y)), y].clip(1e-12)).mean()
    brier = np.square(prob - np.eye(5)[y]).sum(1).mean()
    confidence = prob.max(1)
    correct = prediction == y
    ece = sum(np.abs(correct[(confidence >= a) & (confidence < b)].mean() -
                     confidence[(confidence >= a) & (confidence < b)].mean()) *
              ((confidence >= a) & (confidence < b)).mean()
              for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])
              if ((confidence >= a) & (confidence < b)).any())
    return {"n": len(y), "participant_count": len(people),
            "mean_participant_macro_f1": float(np.mean(people)),
            "pooled_macro_f1": float(np.mean(f1)), "accuracy": float(accuracy_score(y, prediction)),
            "kappa": float(cohen_kappa_score(y, prediction, labels=np.arange(5))),
            "per_stage_f1": f1.tolist(), "per_stage_recall": recall.tolist(),
            "prediction_counts": np.bincount(prediction, minlength=5).tolist(),
            "truth_counts": np.bincount(y, minlength=5).tolist(), "confusion": cm.tolist(),
            "nll": float(nll), "brier": float(brier), "ece_10": float(ece),
            "participant_macro_f1_sd": float(np.std(people, ddof=1)) if len(people)>1 else 0.0}

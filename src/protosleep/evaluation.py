from __future__ import annotations

import os
import gc
import json
import math
import time
import copy
import random
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

from .config import *
from .utils import *
from .data import *
from .cache import *
from .macro import *
from .prototypes import ProtoAttnSleep
from .masking import js_divergence_np
from .micro import classification_metrics, _micro_logits

# =============================================================================
# 12. FINAL TEST PREDICTION -- RECORDING BOUNDARIES RETAINED
# =============================================================================
def predict_micro_recordings(model: nn.Module, recordings: List[Recording], batch_size=MICRO_BATCH_SIZE):
    model.eval(); out = {}
    with torch.no_grad():
        for r in recordings:
            preds = []
            for s in range(0, r.n_epochs, batch_size):
                x = torch.from_numpy(r.x[s:s+batch_size]).to(DEVICE)
                with amp_context(DEVICE):
                    logits = _micro_logits(model, x)
                preds.append(logits.argmax(-1).cpu().numpy())
            out[r.recording_id] = dict(y=r.y.copy(), pred=np.concatenate(preds), subject_id=r.subject_id)
    return out


def predict_macro_recordings(model: MacroStageClassifier, nights: List[CachedNight], feature: str):
    model.eval(); out = {}
    with torch.no_grad():
        for n in nights:
            arr = getattr(n, feature)
            x = torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE)
            valid = torch.ones((1, n.n_epochs), dtype=torch.bool, device=DEVICE)
            with amp_context(DEVICE):
                logits = model(x, valid)[0]
            out[n.recording_id] = dict(y=n.labels.copy(), pred=logits.argmax(-1).cpu().numpy(), subject_id=n.subject_id)
    return out


def pooled_metrics_from_prediction_dict(preds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    y = np.concatenate([preds[k]["y"] for k in sorted(preds)])
    p = np.concatenate([preds[k]["pred"] for k in sorted(preds)])
    return classification_metrics(y, p)


def transition_window_mask(labels: np.ndarray, radius=TRANSITION_EVAL_RADIUS) -> np.ndarray:
    y = np.asarray(labels)
    m = np.zeros(len(y), dtype=bool)
    if len(y) < 2: return m
    boundaries = np.where(y[1:] != y[:-1])[0] + 1
    for b in boundaries:
        lo = max(0, b - radius)
        hi = min(len(y), b + radius + 1)
        m[lo:hi] = True
    return m


def transition_metrics_from_prediction_dict(preds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ys, ps = [], []
    for k in sorted(preds):
        y = preds[k]["y"]; p = preds[k]["pred"]
        m = transition_window_mask(y)
        if np.any(m):
            ys.append(y[m]); ps.append(p[m])
    if not ys:
        return dict(accuracy=np.nan, macro_f1=np.nan, n1_f1=np.nan, n=0)
    y = np.concatenate(ys); p = np.concatenate(ps)
    met = classification_metrics(y, p)
    return dict(accuracy=met["accuracy"], macro_f1=met["macro_f1"],
                n1_f1=float(met["per_class_f1"][1]), n=int(len(y)))


# =============================================================================
# 13. INTERPRETABILITY / DIAGNOSTICS
# =============================================================================
def prototype_diagnostics(model: ProtoAttnSleep, train_nights: List[CachedNight]) -> Dict[str, Any]:
    usage = np.concatenate([n.prototype for n in train_nights], axis=0)
    mean_usage = usage.mean(axis=0)
    mean_usage = mean_usage / mean_usage.sum()
    entropy = -np.sum(mean_usage * np.log(np.clip(mean_usage, 1e-12, None)))
    effective = float(np.exp(entropy))
    with torch.no_grad():
        p = F.normalize(model.prototype_bank.prototypes.detach().cpu(), dim=-1).numpy()
    gram = p @ p.T
    off = gram[~np.eye(len(gram), dtype=bool)]
    return dict(
        mean_usage=mean_usage,
        effective_prototypes=effective,
        max_pairwise_cosine=float(off.max()),
        mean_pairwise_cosine=float(off.mean()),
        assignment_entropy_mean=float(np.mean(-np.sum(usage*np.log(np.clip(usage,1e-12,None)), axis=1))),
    )


def prototype_stage_association(nights: List[CachedNight]) -> np.ndarray:
    numer = np.zeros((NUM_PROTOTYPES, NUM_CLASSES), dtype=np.float64)
    for n in nights:
        for c in range(NUM_CLASSES):
            m = n.labels == c
            if np.any(m): numer[:, c] += n.prototype[m].sum(axis=0)
    denom = numer.sum(axis=1, keepdims=True)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)


def grouped_context_reconstruction(mae: MaskedSequenceAutoencoder, usage: np.ndarray, stride=8) -> np.ndarray:
    """Efficient context-only reconstruction: each epoch is masked once; co-masked epochs are >=stride apart."""
    mae.eval()
    t = len(usage)
    x = torch.from_numpy(usage).float().unsqueeze(0).to(DEVICE)
    valid = torch.ones((1, t), dtype=torch.bool, device=DEVICE)
    recon = np.zeros_like(usage, dtype=np.float32)
    with torch.no_grad():
        for offset in range(stride):
            idx = np.arange(offset, t, stride)
            if len(idx) == 0: continue
            mask = torch.zeros((1, t), dtype=torch.bool, device=DEVICE)
            mask[0, torch.from_numpy(idx).to(DEVICE)] = True
            with amp_context(DEVICE):
                logits = mae(x, valid, mask)
            probs = F.softmax(logits[0, idx], dim=-1).float().cpu().numpy()
            recon[idx] = probs
    return recon


def plot_confusion(met: Dict[str, Any], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(met["confusion_matrix"], display_labels=STAGE_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_proto_trajectory(night: CachedNight, pred: np.ndarray,
                          recon: Optional[np.ndarray], out_path: Path):
    t = np.arange(night.n_epochs)
    dom = night.prototype.argmax(axis=1)
    entropy = -np.sum(night.prototype * np.log(np.clip(night.prototype, 1e-12, None)), axis=1)
    surprise = js_divergence_np(night.prototype, recon) if recon is not None else np.zeros(night.n_epochs)

    fig, axes = plt.subplots(5, 1, figsize=(14, 9), sharex=True)
    axes[0].step(t, night.labels, where="mid"); axes[0].set_ylabel("True")
    axes[1].step(t, pred, where="mid"); axes[1].set_ylabel("Pred")
    axes[2].plot(t, dom, linewidth=0.8); axes[2].set_ylabel("Proto")
    axes[3].plot(t, entropy, linewidth=0.8); axes[3].set_ylabel("Entropy")
    axes[4].plot(t, surprise, linewidth=0.8); axes[4].set_ylabel("Context JS"); axes[4].set_xlabel("30-s epoch")
    for ax in axes[:2]:
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(STAGE_NAMES)
    fig.suptitle(f"{night.recording_id}: prototype trajectory and context surprise")
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)

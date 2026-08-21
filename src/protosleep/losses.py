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
from .cache import *
from .macro import *

# =============================================================================
# 10. SSL LOSSES + TRAIN-ONLY PROTOTYPE TRANSITION GRAPH
# =============================================================================
def fit_transition_graph(train_nights: List[CachedNight], smoothing=TRANSITION_GRAPH_SMOOTHING):
    c = np.full((NUM_PROTOTYPES, NUM_PROTOTYPES), float(smoothing), dtype=np.float64)
    for n in train_nights:
        u = n.prototype.astype(np.float64)
        if len(u) > 1:
            c += np.einsum("ti,tj->ij", u[:-1], u[1:])
    p_fwd = c / c.sum(axis=1, keepdims=True)
    c_rev = c.T.copy()
    p_rev = c_rev / c_rev.sum(axis=1, keepdims=True)
    return p_fwd.astype(np.float32), p_rev.astype(np.float32), c.astype(np.float64)


def masked_distribution_kl(logits: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    use = mask & valid
    if not torch.any(use):
        raise RuntimeError("No masked valid positions")
    log_q = F.log_softmax(logits, dim=-1)
    per = F.kl_div(log_q, target, reduction="none").sum(-1)
    return per[use].mean()


def normalized_prototype_vectors(model: ProtoAttnSleep) -> np.ndarray:
    """Frozen fold-specific spherical prototype geometry for macro reconstruction."""
    with torch.no_grad():
        p = F.normalize(model.prototype_bank.prototypes.detach().float().cpu(), dim=-1)
    return p.numpy().astype(np.float32)


def masked_spherical_barycenter_loss(logits: torch.Tensor, target: torch.Tensor,
                                      mask: torch.Tensor, valid: torch.Tensor,
                                      prototype_vectors: torch.Tensor) -> torch.Tensor:
    """
    Geometry-aware semantic reconstruction. KL treats prototype IDs as unrelated bins.
    This auxiliary term also matches the expected point on the learned unit-sphere
    prototype vocabulary:

        mu(p) = sum_k p_k P_k,   L_geo = ||mu(target)-mu(pred)||_2^2.

    Under CUDA autocast the MAE logits are usually float16 while cached targets and
    frozen prototype vectors are float32. The barycenter matmuls are intentionally
    evaluated in float32 so AMP cannot produce Half-vs-Float matmul failures and so
    this small geometry term remains numerically stable. Gradients still flow through
    the float32 cast back into the MAE logits.
    """
    use = mask & valid
    if not torch.any(use):
        raise RuntimeError("No masked valid positions")

    # Keep the geometry calculation explicitly in FP32. This is cheap (K=48) and
    # avoids AMP dtype mismatches such as Half @ Float on CUDA.
    p = target[use].to(device=logits.device, dtype=torch.float32)
    q = F.softmax(logits[use].float(), dim=-1)
    P = F.normalize(
        prototype_vectors.to(device=logits.device, dtype=torch.float32),
        dim=-1,
        eps=1e-8,
    )

    mu_p = p @ P
    mu_q = q @ P
    return (mu_p - mu_q).pow(2).sum(dim=-1).mean()


def masked_distribution_reconstruction(logits: torch.Tensor, target: torch.Tensor,
                                       mask: torch.Tensor, valid: torch.Tensor,
                                       prototype_vectors: Optional[torch.Tensor] = None,
                                       geometry_lambda: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kl = masked_distribution_kl(logits, target, mask, valid)
    geo = logits.sum() * 0.0
    if geometry_lambda > 0:
        if prototype_vectors is None:
            raise ValueError("geometry_lambda>0 requires fold-specific prototype_vectors")
        geo = masked_spherical_barycenter_loss(logits, target, mask, valid, prototype_vectors)
    return kl + float(geometry_lambda) * geo, kl, geo


def masked_continuous_mse(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    use = mask & valid
    if not torch.any(use):
        raise RuntimeError("No masked valid positions")
    per = (pred - target).pow(2).mean(dim=-1)
    return per[use].mean()


def js_torch(p: torch.Tensor, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    p = p.clamp_min(eps); p = p / p.sum(-1, keepdim=True)
    q = q.clamp_min(eps); q = q / q.sum(-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (q * (q.log() - m.log())).sum(-1)


def _masked_runs(mask_1d: torch.Tensor, length: int) -> List[Tuple[int, int]]:
    arr = mask_1d[:length].detach().cpu().numpy().astype(bool)
    runs = []
    i = 0
    while i < length:
        if not arr[i]:
            i += 1
            continue
        s = i
        while i + 1 < length and arr[i + 1]:
            i += 1
        runs.append((s, i))
        i += 1
    return runs


def transition_consistency_loss(pred_probs: torch.Tensor, target_u: torch.Tensor,
                                mask: torch.Tensor, valid: torch.Tensor,
                                p_fwd: torch.Tensor, p_rev: torch.Tensor) -> torch.Tensor:
    """
    For each contiguous masked span, propagate transition priors from the nearest visible
    left/right anchors. This avoids the invalid one-step prior when an immediate neighbor
    is itself masked.
    """
    losses = []
    b = pred_probs.size(0)
    lengths = valid.sum(1).tolist()
    for bi in range(b):
        length = int(lengths[bi])
        for s, e in _masked_runs(mask[bi], length):
            if s > 0:
                q = target_u[bi, s - 1].detach()
                for t in range(s, e + 1):
                    q = q @ p_fwd
                    losses.append(js_torch(pred_probs[bi, t], q))
            if e < length - 1:
                q = target_u[bi, e + 1].detach()
                for t in range(e, s - 1, -1):
                    q = q @ p_rev
                    losses.append(js_torch(pred_probs[bi, t], q))
    if not losses:
        return pred_probs.sum() * 0.0
    return torch.stack(losses).mean()


def fit_latent_standardizer(train_nights: List[CachedNight]) -> Tuple[np.ndarray, np.ndarray]:
    z = np.concatenate([n.latent for n in train_nights], axis=0).astype(np.float64)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardized_latent_nights(nights: List[CachedNight], mean: np.ndarray, std: np.ndarray) -> List[CachedNight]:
    out = []
    for n in nights:
        out.append(CachedNight(n.recording_id, n.subject_id, n.prototype,
                               ((n.latent - mean) / std).astype(np.float32),
                               n.micro_logits, n.labels))
    return out

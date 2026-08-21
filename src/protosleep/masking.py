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

# =============================================================================
# 8. TRANSITION INTENSITY + MASK SAMPLERS
# =============================================================================
def js_divergence_np(p: np.ndarray, q: np.ndarray, eps=1e-8) -> np.ndarray:
    p = np.clip(p, eps, None); p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q, eps, None); q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(m)), axis=-1) + 0.5 * np.sum(q * (np.log(q) - np.log(m)), axis=-1)


def transition_intensity(usage: np.ndarray) -> np.ndarray:
    t = usage.shape[0]
    r = np.zeros(t, dtype=np.float64)
    if t > 1:
        r[1:] = js_divergence_np(usage[:-1], usage[1:])
    return r


def _add_span(mask: np.ndarray, start: int, length: int, budget_left: int) -> int:
    t = len(mask)
    if budget_left <= 0 or length <= 0:
        return 0
    start = max(0, min(int(start), t - 1))
    stop = min(t, start + int(length))
    candidates = np.where(~mask[start:stop])[0] + start
    if len(candidates) == 0:
        return 0
    # Preserve contiguity as much as possible; cap at remaining token budget.
    chosen = candidates[:budget_left]
    mask[chosen] = True
    return int(len(chosen))


def make_night_mask(usage: np.ndarray, mode: str, rng: np.random.Generator,
                    ratio=MACRO_MASK_RATIO, transition_fraction=TRANSITION_MASK_FRACTION,
                    span_min=MASK_SPAN_MIN, span_max=MASK_SPAN_MAX) -> np.ndarray:
    """
    mode='random': ordinary random spans.
    mode='transition': same ratio/span distribution, but a fraction of the token budget
    is centered stochastically near large label-free JS prototype changes.
    """
    t = int(usage.shape[0])
    if t < 2:
        return np.zeros(t, dtype=bool)
    target = int(round(t * ratio))
    target = max(1, min(target, t - 2 if t >= 3 else t - 1))
    mask = np.zeros(t, dtype=bool)

    if mode not in {"random", "transition"}:
        raise ValueError(mode)

    transition_budget = int(round(target * transition_fraction)) if mode == "transition" else 0
    if transition_budget > 0 and t > 2:
        r = transition_intensity(usage)
        candidates = np.arange(1, t)
        w = r[1:].astype(np.float64)
        # Positive floor prevents deterministic leakage/top-k selection and keeps support broad.
        floor = max(float(w.mean()) * 0.05, 1e-8)
        probs = (w + floor) / np.sum(w + floor)
        attempts = 0
        while int(mask.sum()) < transition_budget and attempts < 10 * t:
            center = int(rng.choice(candidates, p=probs))
            length = int(rng.integers(span_min, span_max + 1))
            remaining = transition_budget - int(mask.sum())
            length = max(1, min(length, remaining))
            start = center - length // 2
            start = max(0, min(start, t - length))
            _add_span(mask, start, length, remaining)
            attempts += 1

    attempts = 0
    while int(mask.sum()) < target and attempts < 20 * t:
        remaining = target - int(mask.sum())
        length = int(rng.integers(span_min, span_max + 1))
        length = max(1, min(length, remaining, t - 1))
        start = int(rng.integers(0, max(1, t - length + 1)))
        _add_span(mask, start, length, remaining)
        attempts += 1

    # Guaranteed exact budget fallback.
    if int(mask.sum()) < target:
        free = np.where(~mask)[0]
        add = rng.choice(free, size=target-int(mask.sum()), replace=False)
        mask[add] = True
    if int(mask.sum()) > target:
        idx = np.where(mask)[0]
        drop = rng.choice(idx, size=int(mask.sum())-target, replace=False)
        mask[drop] = False
    if mask.all():
        mask[-1] = False
    return mask


def make_batch_masks(x: torch.Tensor, valid: torch.Tensor, mode: str,
                     rng: np.random.Generator) -> torch.Tensor:
    b, tmax, _ = x.shape
    masks = torch.zeros((b, tmax), dtype=torch.bool, device=x.device)
    x_cpu = x.detach().float().cpu().numpy()
    valid_cpu = valid.cpu().numpy()
    for bi in range(b):
        t = int(valid_cpu[bi].sum())
        # For transition mode, x must be prototype distributions. Random mode ignores values.
        usage = x_cpu[bi, :t]
        m = make_night_mask(usage, mode=mode, rng=rng)
        masks[bi, :t] = torch.from_numpy(m).to(x.device)
    return masks

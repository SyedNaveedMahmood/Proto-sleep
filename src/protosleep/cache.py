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
from .prototypes import *

# =============================================================================
# 6. CACHE FROZEN MICRO FEATURES RECORDING-BY-RECORDING
# =============================================================================
@dataclass
class CachedNight:
    recording_id: str
    subject_id: int
    prototype: np.ndarray   # [T,K]
    latent: np.ndarray      # [T,30]
    micro_logits: np.ndarray # [T,5]
    labels: np.ndarray      # [T]

    @property
    def n_epochs(self):
        return int(self.labels.shape[0])


def extract_one_recording(model: ProtoAttnSleep, rec: Recording, batch_size=MICRO_BATCH_SIZE) -> CachedNight:
    model.eval()
    usages, latents, logits_all = [], [], []
    with torch.no_grad():
        for start in range(0, rec.n_epochs, batch_size):
            xb = torch.from_numpy(rec.x[start:start+batch_size]).to(DEVICE, non_blocking=True)
            with amp_context(DEVICE):
                out = model(xb)
            usages.append(out["usage"].float().cpu().numpy())
            latents.append(out["latent_summary"].float().cpu().numpy())
            logits_all.append(out["logits"].float().cpu().numpy())
    u = np.concatenate(usages, axis=0).astype(np.float32)
    z = np.concatenate(latents, axis=0).astype(np.float32)
    l = np.concatenate(logits_all, axis=0).astype(np.float32)
    if not (len(u) == len(z) == len(l) == rec.n_epochs):
        raise RuntimeError(f"Cache length mismatch for {rec.recording_id}")
    if not np.allclose(u.sum(axis=1), 1.0, atol=2e-4):
        raise RuntimeError(f"Prototype simplex violation in {rec.recording_id}")
    return CachedNight(rec.recording_id, rec.subject_id, u, z, l, rec.y.copy())


def build_feature_cache(model: ProtoAttnSleep, recordings: List[Recording], cache_dir: Path,
                        split_name: str) -> List[CachedNight]:
    ensure_dir(cache_dir)
    nights = []
    for rec in recordings:
        night = extract_one_recording(model, rec)
        out = cache_dir / f"{split_name}__{rec.recording_id}.npz"
        np.savez_compressed(
            out,
            prototype=night.prototype, latent=night.latent, micro_logits=night.micro_logits,
            labels=night.labels, recording_id=np.array(night.recording_id),
            subject_id=np.array(night.subject_id, dtype=np.int64),
        )
        # Verify write preserves exact epoch order/labels.
        with np.load(out, allow_pickle=False) as d:
            if not np.array_equal(d["labels"], rec.y):
                raise RuntimeError(f"Cache label/order verification failed: {rec.recording_id}")
        nights.append(night)
    return nights


def load_feature_cache(recordings: List[Recording], cache_dir: Path, split_name: str) -> List[CachedNight]:
    """Reload cached nights in the exact order of `recordings`, validating labels/metadata."""
    nights: List[CachedNight] = []
    for rec in recordings:
        path = Path(cache_dir) / f"{split_name}__{rec.recording_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as d:
            rid = str(np.asarray(d["recording_id"]).item())
            sid = int(np.asarray(d["subject_id"]).item())
            labels = d["labels"].astype(np.int64)
            proto = d["prototype"].astype(np.float32)
            latent = d["latent"].astype(np.float32)
            logits = d["micro_logits"].astype(np.float32)
        if rid != rec.recording_id or sid != rec.subject_id:
            raise RuntimeError(f"Cache metadata mismatch for {rec.recording_id}")
        if not np.array_equal(labels, rec.y):
            raise RuntimeError(f"Cache order/label mismatch for {rec.recording_id}")
        if not np.allclose(proto.sum(axis=1), 1.0, atol=2e-4):
            raise RuntimeError(f"Prototype simplex violation in cached {rec.recording_id}")
        nights.append(CachedNight(rid, sid, proto, latent, logits, labels))
    return nights


def load_model_state(model: nn.Module, checkpoint: Path, device: Optional[torch.device] = None) -> nn.Module:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model


def cache_complete(recordings: List[Recording], cache_dir: Path, split_name: str) -> bool:
    return all((Path(cache_dir) / f"{split_name}__{r.recording_id}.npz").exists() for r in recordings)

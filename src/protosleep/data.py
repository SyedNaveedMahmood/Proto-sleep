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
# 2. SLEEP-EDF NPZ LOADING -- FOLLOW THE ATTACHED NOTEBOOK'S DATA CONVENTION
# =============================================================================
@dataclass
class Recording:
    path: str
    recording_id: str
    subject_id: int
    x: np.ndarray       # [T,1,3000], float32
    y: np.ndarray       # [T], int64
    fs: int

    @property
    def n_epochs(self) -> int:
        return int(self.y.shape[0])


def _extract_subject_id_like_original(filename: str) -> int:
    stem = os.path.basename(filename)
    if len(stem) >= 5 and stem[3:5].isdigit():
        return int(stem[3:5])
    raise ValueError(
        f"Cannot extract Sleep-EDF subject ID from '{stem}' using the attached notebook rule filename[3:5]."
    )


def _canonicalize_epoch_shape(x: np.ndarray, filename: str) -> np.ndarray:
    # Same shape logic as the provided one-cell AttnSleep notebook.
    if x.ndim == 3 and x.shape[1] > x.shape[2]:
        x = np.transpose(x, (0, 2, 1))
    elif x.ndim == 2:
        x = x[:, np.newaxis, :]
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"{filename}: expected [N,1,L] after canonicalization; got {x.shape}")
    return np.asarray(x, dtype=np.float32)


def load_sleep_edf_recordings(data_dir: str) -> List[Recording]:
    import glob
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Sleep-EDF NPZ directory not found: {data_dir}\n"
            "Set DATA_DIR or export SLEEP_EDF_NPZ_DIR."
        )
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    recordings: List[Recording] = []
    print(f"Loading {len(paths)} NPZ recordings from {data_dir}")
    for path in paths:
        fn = os.path.basename(path)
        with np.load(path, allow_pickle=False) as d:
            if "x" not in d.files or "y" not in d.files:
                raise KeyError(f"{fn}: expected keys 'x' and 'y'")
            x = _canonicalize_epoch_shape(d["x"], fn)
            y = np.asarray(d["y"]).reshape(-1).astype(np.int64, copy=False)
            if "fs" in d.files:
                fs = int(round(float(np.asarray(d["fs"]).reshape(-1)[0])))
            else:
                if x.shape[-1] % 30 != 0:
                    raise ValueError(f"{fn}: cannot infer fs from {x.shape[-1]} samples/epoch")
                fs = int(x.shape[-1] // 30)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{fn}: x has {x.shape[0]} epochs but y has {y.shape[0]}")
        if x.shape[-1] != EXPECTED_SAMPLES_PER_EPOCH:
            raise ValueError(
                f"{fn}: expected {EXPECTED_SAMPLES_PER_EPOCH} samples/epoch; got {x.shape[-1]}. "
                "This notebook does not silently resample."
            )
        if fs != EXPECTED_FS:
            raise ValueError(f"{fn}: expected {EXPECTED_FS} Hz; got {fs} Hz")
        if not np.isfinite(x).all():
            raise ValueError(f"{fn}: non-finite EEG values")
        bad = np.setdiff1d(np.unique(y), np.arange(NUM_CLASSES))
        if bad.size:
            raise ValueError(f"{fn}: labels outside 0..4: {bad.tolist()}")

        recordings.append(Recording(
            path=path,
            recording_id=os.path.splitext(fn)[0],
            subject_id=_extract_subject_id_like_original(fn),
            x=x,
            y=y,
            fs=fs,
        ))

    subjects = sorted({r.subject_id for r in recordings})
    print(f"Loaded {len(recordings)} recordings, {len(subjects)} subjects, "
          f"{sum(r.n_epochs for r in recordings):,} epochs")
    print("Subjects:", subjects)
    return recordings


def make_subject_order(recordings: List[Recording]) -> List[int]:
    subjects = sorted({r.subject_id for r in recordings})
    if subjects == list(range(20)):
        return [subjects[i] for i in ATTNSLEEP_R_PERMUTE_20]
    rng = np.random.default_rng(SEED)
    return [int(x) for x in rng.permutation(np.asarray(subjects))]


def split_subjects(recordings: List[Recording], fold_id: int) -> Dict[str, Any]:
    order = make_subject_order(recordings)
    if len(order) < 3:
        raise ValueError("Need >=3 subjects for disjoint train/val/test")
    if not 0 <= fold_id < len(order):
        raise ValueError(f"fold_id={fold_id} outside 0..{len(order)-1}")
    test_subject = order[fold_id]
    val_subject = order[(fold_id + 1) % len(order)]
    train_subjects = [s for s in order if s not in {test_subject, val_subject}]

    train_set = set(train_subjects)
    train = [r for r in recordings if r.subject_id in train_set]
    val = [r for r in recordings if r.subject_id == val_subject]
    test = [r for r in recordings if r.subject_id == test_subject]
    assert train and val and test
    assert {r.subject_id for r in train}.isdisjoint({r.subject_id for r in val})
    assert {r.subject_id for r in train}.isdisjoint({r.subject_id for r in test})
    assert {r.subject_id for r in val}.isdisjoint({r.subject_id for r in test})
    return dict(
        order=order,
        train_subjects=sorted({r.subject_id for r in train}),
        val_subjects=sorted({r.subject_id for r in val}),
        test_subjects=sorted({r.subject_id for r in test}),
        train=train, val=val, test=test,
    )


class EpochDataset(Dataset):
    def __init__(self, recordings: List[Recording]):
        self.recordings = recordings
        self.cum = np.cumsum([r.n_epochs for r in recordings], dtype=np.int64)
        self.total = int(self.cum[-1]) if len(self.cum) else 0

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        rec_i = int(np.searchsorted(self.cum, idx, side="right"))
        prev = 0 if rec_i == 0 else int(self.cum[rec_i - 1])
        ep_i = int(idx - prev)
        r = self.recordings[rec_i]
        return torch.from_numpy(r.x[ep_i]), torch.tensor(int(r.y[ep_i]), dtype=torch.long)


def make_epoch_loader(recordings: List[Recording], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        EpochDataset(recordings), batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )


def labels_from_recordings(recordings: List[Recording]) -> np.ndarray:
    return np.concatenate([r.y for r in recordings]).astype(np.int64, copy=False)


def balanced_class_weights_from_train(recordings: List[Recording]) -> torch.Tensor:
    y = labels_from_recordings(recordings)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split missing class(es): {counts.tolist()}")
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.mean()
    print("Train class counts:", {STAGE_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)})
    print("CE weights:", {STAGE_NAMES[i]: round(float(w[i]), 4) for i in range(NUM_CLASSES)})
    return torch.tensor(w, dtype=torch.float32)

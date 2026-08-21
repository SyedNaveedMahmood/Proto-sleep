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

# =============================================================================
# 7. NIGHT DATASET / PADDING
# =============================================================================
class NightDataset(Dataset):
    def __init__(self, nights: List[CachedNight], feature: str):
        if feature not in {"prototype", "latent"}:
            raise ValueError(feature)
        self.nights = nights
        self.feature = feature

    def __len__(self): return len(self.nights)

    def __getitem__(self, i):
        n = self.nights[i]
        x = getattr(n, self.feature)
        return dict(
            x=torch.from_numpy(x),
            y=torch.from_numpy(n.labels.astype(np.int64)),
            recording_id=n.recording_id,
            subject_id=n.subject_id,
        )


def collate_nights(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = torch.tensor([b["x"].shape[0] for b in batch], dtype=torch.long)
    max_t = int(lengths.max())
    d = int(batch[0]["x"].shape[1])
    x = torch.zeros(len(batch), max_t, d, dtype=torch.float32)
    y = torch.full((len(batch), max_t), -100, dtype=torch.long)
    valid = torch.zeros(len(batch), max_t, dtype=torch.bool)
    ids, subjects = [], []
    for i, b in enumerate(batch):
        t = b["x"].shape[0]
        x[i, :t] = b["x"].float()
        y[i, :t] = b["y"].long()
        valid[i, :t] = True
        ids.append(b["recording_id"])
        subjects.append(int(b["subject_id"]))
    return dict(x=x, y=y, valid=valid, lengths=lengths,
                recording_ids=ids, subject_ids=subjects)


def make_night_loader(nights: List[CachedNight], feature: str, batch_size=MACRO_BATCH_SIZE,
                      shuffle=True) -> DataLoader:
    return DataLoader(NightDataset(nights, feature), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=PIN_MEMORY, collate_fn=collate_nights, drop_last=False)

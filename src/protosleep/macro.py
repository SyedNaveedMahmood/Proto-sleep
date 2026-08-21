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
from .prototypes import sinusoidal_from_positions

# =============================================================================
# 9. MACRO MAE: VISIBLE-ONLY ENCODER + LIGHT DECODER
# =============================================================================
class MacroEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model=MACRO_D_MODEL, layers=MACRO_LAYERS,
                 heads=MACRO_HEADS, ff=MACRO_FF, dropout=MACRO_DROPOUT):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))

    def _embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.input_norm(x))
        return h + sinusoidal_from_positions(positions, self.d_model, h.dtype)

    def forward_full(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self._embed(x, pos)
        return self.encoder(h, src_key_padding_mask=~valid)

    def forward_visible(self, x: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor):
        b, _, _ = x.shape
        visible = valid & (~mask)
        counts = visible.sum(dim=1)
        if torch.any(counts < 1):
            raise RuntimeError("MAE mask left a sequence with no visible token")
        vmax = int(counts.max().item())
        xv = torch.zeros(b, vmax, x.size(-1), device=x.device, dtype=x.dtype)
        pv = torch.zeros(b, vmax, device=x.device, dtype=torch.long)
        vv = torch.zeros(b, vmax, device=x.device, dtype=torch.bool)
        indices: List[torch.Tensor] = []
        for bi in range(b):
            idx = torch.where(visible[bi])[0]
            n = idx.numel()
            xv[bi, :n] = x[bi, idx]
            pv[bi, :n] = idx
            vv[bi, :n] = True
            indices.append(idx)
        h = self._embed(xv, pv)
        h = self.encoder(h, src_key_padding_mask=~vv)
        return h, vv, indices


class MaskedSequenceAutoencoder(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, target_type: str,
                 d_model=MACRO_D_MODEL, decoder_dim=MACRO_DECODER_DIM):
        super().__init__()
        if target_type not in {"distribution", "continuous"}:
            raise ValueError(target_type)
        self.target_type = target_type
        self.encoder = MacroEncoder(input_dim=input_dim, d_model=d_model)
        self.enc_to_dec = nn.Linear(d_model, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=MACRO_DECODER_HEADS,
            dim_feedforward=MACRO_DECODER_FF, dropout=MACRO_DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=MACRO_DECODER_LAYERS,
                                             norm=nn.LayerNorm(decoder_dim))
        self.reconstruction_head = nn.Linear(decoder_dim, target_dim)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        enc, vis_valid, indices = self.encoder.forward_visible(x, valid, mask)
        projected = self.enc_to_dec(enc)
        # Under CUDA autocast, projected is typically FP16 while the parameter-backed
        # mask token is FP32. Indexed assignment requires exact dtype equality.
        dec = self.mask_token.to(device=projected.device, dtype=projected.dtype).expand(b, t, -1).clone()
        for bi, idx in enumerate(indices):
            n = idx.numel()
            dec[bi, idx] = projected[bi, :n]
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        dec = dec + sinusoidal_from_positions(pos, dec.size(-1), dec.dtype)
        dec = self.decoder(dec, src_key_padding_mask=~valid)
        return self.reconstruction_head(dec)


class MacroStageClassifier(nn.Module):
    def __init__(self, encoder: MacroEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, NUM_CLASSES)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.encoder.forward_full(x, valid)
        return self.head(h)

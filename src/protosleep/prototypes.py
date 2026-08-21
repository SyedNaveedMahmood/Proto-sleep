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
from .attnsleep import *

# =============================================================================
# 4. SPHERICAL PROTOTYPE TOKENIZER + PROTOTYPE-AWARE LOCAL MODEL (B)
# =============================================================================
def _slice_bounds(n: int, position_slice: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if position_slice is None:
        return 0, n
    a, b = position_slice
    if not (0 <= a < b <= n):
        raise ValueError(f"Invalid prototype position slice {position_slice} for N={n}")
    return int(a), int(b)


class SphericalPrototypeBank(nn.Module):
    def __init__(self, k=NUM_PROTOTYPES, dim=PROTO_DIM, temperature=PROTO_TEMPERATURE,
                 position_slice=PROTO_POSITION_SLICE):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be >0")
        self.k = int(k)
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.position_slice = position_slice
        p = F.normalize(torch.randn(self.k, self.dim), dim=-1)
        self.prototypes = nn.Parameter(p)

    def forward(self, afr: torch.Tensor) -> Dict[str, torch.Tensor]:
        if afr.ndim != 3 or afr.size(1) != self.dim:
            raise ValueError(f"Expected AFR [B,{self.dim},N], got {tuple(afr.shape)}")
        full_tokens = afr.transpose(1, 2)  # [B,N,30]
        a, b = _slice_bounds(full_tokens.size(1), self.position_slice)
        tokens = full_tokens[:, a:b]
        z = F.normalize(tokens, dim=-1, eps=1e-8)
        p = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        sim = torch.einsum("bnd,kd->bnk", z, p)
        assignments = torch.softmax(sim / self.temperature, dim=-1)
        usage = assignments.mean(dim=1)
        return dict(
            full_tokens=full_tokens, tokens=tokens, start=a, stop=b,
            normalized_tokens=z, normalized_prototypes=p,
            similarities=sim, assignments=assignments, usage=usage,
        )

    @torch.no_grad()
    def project_(self):
        self.prototypes.copy_(F.normalize(self.prototypes, dim=-1, eps=1e-8))


def prototype_regularizers(proto: Dict[str, torch.Tensor], sep_margin=0.30) -> Dict[str, torch.Tensor]:
    sim = proto["similarities"]
    assign = proto["assignments"]
    p = proto["normalized_prototypes"]

    # Assignment weights are detached: commitment cannot be reduced merely by changing entropy.
    commit = (assign.detach() * (1.0 - sim)).sum(dim=-1).mean()

    q = assign.mean(dim=(0, 1)).clamp_min(1e-8)
    uniform = 1.0 / q.numel()
    balance = torch.sum(q * torch.log(q / uniform))

    gram = p @ p.t()
    eye = torch.eye(gram.size(0), dtype=torch.bool, device=gram.device)
    off = gram.masked_select(~eye)
    separation = F.relu(off - sep_margin).pow(2).mean()
    return dict(commit=commit, balance=balance, separation=separation)


def sinusoidal_from_positions(positions: torch.Tensor, dim: int, dtype=None) -> torch.Tensor:
    """positions: [...], returns [...,dim]."""
    if dtype is None:
        dtype = torch.float32
    pos = positions.to(dtype=torch.float32).unsqueeze(-1)
    idx = torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / dim)
    pe = torch.zeros(*positions.shape, dim, device=positions.device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[..., 1::2] = torch.cos(pos * div[:pe[..., 1::2].shape[-1]])
    return pe.to(dtype=dtype)


class MicroMaskPredictor(nn.Module):
    """Optional local context network predicting teacher prototype distributions at hidden AFR positions."""
    def __init__(self, input_dim=PROTO_DIM, hidden=MICRO_MASK_HIDDEN, k=NUM_PROTOTYPES,
                 layers=MICRO_MASK_LAYERS, heads=MICRO_MASK_HEADS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, k)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        x = self.input_proj(tokens)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), x)
        pos = torch.arange(n, device=tokens.device).unsqueeze(0).expand(b, n)
        x = x + sinusoidal_from_positions(pos, x.size(-1), x.dtype)
        return self.head(self.encoder(x))


def make_random_local_masks(batch: int, n: int, ratio: float, rng: np.random.Generator,
                            device: torch.device) -> torch.Tensor:
    masks = np.zeros((batch, n), dtype=bool)
    target = max(1, min(n - 1, int(round(n * ratio))))
    for bi in range(batch):
        while masks[bi].sum() < target:
            length = int(rng.integers(MICRO_MASK_SPAN_MIN, MICRO_MASK_SPAN_MAX + 1))
            length = min(length, target - int(masks[bi].sum()), n - 1)
            start = int(rng.integers(0, max(1, n - length + 1)))
            masks[bi, start:start + length] = True
        if masks[bi].all():
            masks[bi, -1] = False
    return torch.from_numpy(masks).to(device=device)


class ProtoAttnSleep(nn.Module):
    """
    Model B. Prototypes are in the actual decision pathway:
      AFR token -> soft prototype mixture -> residual prototype reconstruction -> AttnSleep TCE.
    """
    is_prototype_model = True
    def __init__(self, temperature=PROTO_TEMPERATURE, position_slice=PROTO_POSITION_SLICE,
                 beta_init=PROTO_BETA_INIT, enable_micro_mask=ENABLE_MICRO_MASK):
        super().__init__()
        self.mrcnn = MRCNN(30)
        self.prototype_bank = SphericalPrototypeBank(
            k=NUM_PROTOTYPES, dim=PROTO_DIM, temperature=temperature,
            position_slice=position_slice,
        )
        self.tce = build_attnsleep_tce()
        self.fc = nn.Linear(80 * 30, NUM_CLASSES)
        beta_init = min(max(float(beta_init), 1e-4), 1 - 1e-4)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init)), dtype=torch.float32))
        self.micro_mask_predictor = MicroMaskPredictor() if enable_micro_mask else None

    def prototype_aware_afr(self, afr: torch.Tensor, proto: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = proto["tokens"]
        assign = proto["assignments"]
        p = proto["normalized_prototypes"]
        recon_direction = torch.einsum("bnk,kd->bnd", assign, p)
        token_norm = tokens.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        recon = token_norm * recon_direction
        beta = torch.sigmoid(self.beta_logit)
        blended = (1.0 - beta) * tokens + beta * recon

        full = proto["full_tokens"]
        a, b = proto["start"], proto["stop"]
        if a == 0 and b == full.size(1):
            aware_tokens = blended
        else:
            aware_tokens = full.clone()
            aware_tokens[:, a:b] = blended
        return aware_tokens.transpose(1, 2).contiguous()

    def forward(self, x, micro_mask: Optional[torch.Tensor] = None):
        afr = self.mrcnn(x)
        proto = self.prototype_bank(afr)
        aware_afr = self.prototype_aware_afr(afr, proto)
        enc = self.tce(aware_afr)
        logits = self.fc(enc.contiguous().view(enc.shape[0], -1))
        out = dict(
            logits=logits,
            afr=afr,
            prototype_aware_afr=aware_afr,
            latent_summary=afr.mean(dim=-1),  # [B,30]
            beta=torch.sigmoid(self.beta_logit),
            **proto,
        )
        if self.micro_mask_predictor is not None and micro_mask is not None:
            out["micro_mask_logits"] = self.micro_mask_predictor(proto["tokens"], micro_mask)
        return out

    @torch.no_grad()
    def project_prototypes_(self):
        self.prototype_bank.project_()

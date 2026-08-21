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
from .attnsleep import *
from .prototypes import *
from .micro import *
from .cache import *
from .night import *
from .masking import *
from .macro import *
from .losses import *
from .train_macro import *
from .evaluation import *

# =============================================================================
# 14. SELF-TESTS: SHAPES, SIMPLEX, MASKING, MAE, PADDING, LOSSES
# =============================================================================
def run_self_tests():
    print("\nRunning structural self-tests...")
    seed_everything(7)
    dev = DEVICE

    a = AttnSleepBaseline().to(dev)
    a.apply(init_attnsleep_weights)
    a.eval()
    x = torch.randn(2, 1, 3000, device=dev)
    with torch.no_grad():
        oa = a(x, return_features=True)
    assert oa["logits"].shape == (2, 5)
    assert oa["afr"].shape == (2, 30, 80), oa["afr"].shape

    b = ProtoAttnSleep(enable_micro_mask=True).to(dev)
    b.apply(init_attnsleep_weights)
    b.eval()
    local_mask = torch.zeros(2, 80, dtype=torch.bool, device=dev); local_mask[:, 10:15] = True
    with torch.no_grad():
        ob = b(x, micro_mask=local_mask)
    assert ob["usage"].shape == (2, NUM_PROTOTYPES)
    assert torch.allclose(ob["usage"].sum(-1), torch.ones(2, device=dev), atol=1e-5)
    assert ob["latent_summary"].shape == (2, 30)
    assert ob["micro_mask_logits"].shape == (2, 80, NUM_PROTOTYPES)
    regs = prototype_regularizers(ob)
    assert all(torch.isfinite(v) for v in regs.values())

    rng = np.random.default_rng(4)
    fake_u = rng.random((64, NUM_PROTOTYPES)).astype(np.float32); fake_u /= fake_u.sum(1, keepdims=True)
    mr = make_night_mask(fake_u, "random", rng)
    mt = make_night_mask(fake_u, "transition", rng)
    assert mr.any() and (~mr).any() and mt.any() and (~mt).any()
    assert abs(mr.mean() - MACRO_MASK_RATIO) < 0.03
    assert abs(mt.mean() - MACRO_MASK_RATIO) < 0.03

    mae = MaskedSequenceAutoencoder(NUM_PROTOTYPES, NUM_PROTOTYPES, "distribution").to(dev)
    xb = torch.from_numpy(np.stack([fake_u[:48], fake_u[:48]])).to(dev)
    valid = torch.ones((2, 48), dtype=torch.bool, device=dev)
    mask = torch.from_numpy(np.stack([mr[:48], mt[:48]])).to(dev)
    # Exercise the real AMP/autocast path so FP16/FP32 decoder mismatches are caught here.
    with amp_context(dev):
        pred = mae(xb, valid, mask)
    assert pred.shape == (2, 48, NUM_PROTOTYPES)
    kl = masked_distribution_kl(pred, xb, mask, valid)
    assert torch.isfinite(kl)
    proto_vec = F.normalize(torch.randn(NUM_PROTOTYPES, PROTO_DIM, device=dev), dim=-1)
    geo = masked_spherical_barycenter_loss(pred, xb, mask, valid, proto_vec)
    assert torch.isfinite(geo) and geo >= 0

    # Transition graph rows sum to one and multi-step consistency is finite.
    nights = [CachedNight("s", 0, fake_u, np.zeros((64,30),np.float32), np.zeros((64,5),np.float32), np.zeros(64,np.int64))]
    pf, pr, _ = fit_transition_graph(nights)
    assert np.allclose(pf.sum(1), 1.0, atol=1e-5)
    tc = transition_consistency_loss(F.softmax(pred,-1), xb, mask, valid,
                                     torch.from_numpy(pf).to(dev), torch.from_numpy(pr).to(dev))
    assert torch.isfinite(tc)

    # Padding must not alter loss: padded y=-100 and valid False.
    enc = MacroEncoder(NUM_PROTOTYPES).to(dev)
    clf = MacroStageClassifier(enc).to(dev)
    xp = torch.zeros(2, 12, NUM_PROTOTYPES, device=dev)
    xp[0,:10] = torch.from_numpy(fake_u[:10]).to(dev); xp[1,:7] = torch.from_numpy(fake_u[10:17]).to(dev)
    vp = torch.zeros(2,12,dtype=torch.bool,device=dev); vp[0,:10]=True; vp[1,:7]=True
    yp = torch.full((2,12),-100,dtype=torch.long,device=dev); yp[0,:10]=0; yp[1,:7]=1
    lp = clf(xp,vp)
    ce = F.cross_entropy(lp.reshape(-1,5), yp.reshape(-1), ignore_index=-100)
    assert torch.isfinite(ce)

    # One optimizer step through prototype and MAE models.
    b.train(); opt = torch.optim.Adam(b.parameters(), lr=1e-4)
    ob = b(x)
    loss = F.cross_entropy(ob["logits"], torch.tensor([0,1],device=dev)) + 0.01*prototype_regularizers(ob)["commit"]
    opt.zero_grad(); loss.backward(); opt.step(); b.project_prototypes_()
    assert torch.isfinite(loss)

    mae.train(); opt2 = torch.optim.AdamW(mae.parameters(), lr=1e-4)
    pred = mae(xb, valid, mask); loss2 = masked_distribution_kl(pred, xb, mask, valid)
    opt2.zero_grad(); loss2.backward(); opt2.step(); assert torch.isfinite(loss2)

    print("Self-tests PASSED: AttnSleep AFR/logits, prototype simplex/losses, masks, MAE, spherical-geometry loss, graph, padding, optimizer steps.")
    del a, b, mae, enc, clf, x, xb
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

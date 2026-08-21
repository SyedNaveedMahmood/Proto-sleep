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
from .night import *
from .masking import *
from .macro import *
from .losses import *
from .micro import classification_metrics

# =============================================================================
# 11. MACRO SSL PRETRAINING + SUPERVISED FINE-TUNING
# =============================================================================
def pretrain_mae(model: MaskedSequenceAutoencoder, train_loader: DataLoader,
                 mask_mode: str, out_path: Path, seed: int,
                 p_fwd_np: Optional[np.ndarray] = None,
                 p_rev_np: Optional[np.ndarray] = None,
                 prototype_vectors_np: Optional[np.ndarray] = None,
                 geometry_lambda: float = 0.0) -> Dict[str, Any]:
    seed_everything(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=MACRO_SSL_LR, weight_decay=MACRO_SSL_WEIGHT_DECAY)
    scaler = make_grad_scaler(DEVICE)
    rng = np.random.default_rng(seed + 77)
    p_fwd = torch.from_numpy(p_fwd_np).to(DEVICE) if p_fwd_np is not None else None
    p_rev = torch.from_numpy(p_rev_np).to(DEVICE) if p_rev_np is not None else None
    prototype_vectors = (torch.from_numpy(prototype_vectors_np).to(DEVICE)
                         if prototype_vectors_np is not None else None)
    history = []

    for epoch in range(1, MACRO_SSL_EPOCHS + 1):
        model.train()
        loss_sum = recon_sum = kl_sum = geo_sum = trans_sum = 0.0
        batches = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            valid = batch["valid"].to(DEVICE, non_blocking=True)
            mask = make_batch_masks(x, valid, mode=mask_mode, rng=rng)
            opt.zero_grad(set_to_none=True)
            with amp_context(DEVICE):
                pred = model(x, valid, mask)
                if model.target_type == "distribution":
                    recon, recon_kl, recon_geo = masked_distribution_reconstruction(
                        pred, x, mask, valid, prototype_vectors, geometry_lambda
                    )
                else:
                    recon = masked_continuous_mse(pred, x, mask, valid)
                    recon_kl = pred.sum() * 0.0
                    recon_geo = pred.sum() * 0.0
                trans = pred.sum() * 0.0
                if (USE_TRANSITION_GRAPH_LOSS and model.target_type == "distribution"
                        and p_fwd is not None and p_rev is not None):
                    trans = transition_consistency_loss(F.softmax(pred, -1), x, mask, valid, p_fwd, p_rev)
                loss = recon + TRANSITION_GRAPH_LAMBDA * trans

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach().cpu())
            recon_sum += float(recon.detach().cpu())
            kl_sum += float(recon_kl.detach().cpu())
            geo_sum += float(recon_geo.detach().cpu())
            trans_sum += float(trans.detach().cpu())
            batches += 1
        row = dict(epoch=epoch, loss=loss_sum/max(1,batches), recon=recon_sum/max(1,batches),
                   recon_kl=kl_sum/max(1,batches), recon_geometry=geo_sum/max(1,batches),
                   transition=trans_sum/max(1,batches))
        history.append(row)
        print(f"SSL {mask_mode:10s} e{epoch:03d} loss={row['loss']:.5f} KL={row['recon_kl']:.5f} "
              f"geo={row['recon_geometry']:.5f} trans={row['transition']:.5f}")

    state = cpu_state_dict(model)
    ensure_dir(out_path.parent)
    torch.save(dict(state_dict=state, mask_mode=mask_mode, geometry_lambda=float(geometry_lambda),
                    project_version=PROJECT_VERSION), out_path)
    return dict(model=model, history=pd.DataFrame(history))


def evaluate_macro_classifier(model: MacroStageClassifier, loader: DataLoader) -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            valid = batch["valid"].to(DEVICE, non_blocking=True)
            with amp_context(DEVICE):
                logits = model(x, valid)
            use = valid
            ys.append(y[use].cpu().numpy())
            ps.append(logits.argmax(-1)[use].cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = classification_metrics(y, p)
    m["y_true"] = y; m["y_pred"] = p
    return m


def train_macro_classifier(model: MacroStageClassifier, train_loader: DataLoader, val_loader: DataLoader,
                           class_weights: torch.Tensor, out_path: Path, seed: int) -> Dict[str, Any]:
    seed_everything(seed)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=MACRO_FT_LR, weight_decay=MACRO_FT_WEIGHT_DECAY)
    scaler = make_grad_scaler(DEVICE)
    best_f1 = -np.inf; best_state = None; best_epoch = -1; patience = 0
    history = []

    for epoch in range(1, MACRO_FT_EPOCHS + 1):
        model.train(); total = 0.0; n_valid = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            valid = batch["valid"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_context(DEVICE):
                logits = model(x, valid)
                loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            nv = int(valid.sum().item())
            total += float(loss.detach().cpu()) * nv; n_valid += nv

        val = evaluate_macro_classifier(model, val_loader)
        row = dict(epoch=epoch, train_loss=total/max(1,n_valid), val_macro_f1=val["macro_f1"], val_acc=val["accuracy"])
        history.append(row)
        print(f"macro e{epoch:03d} train={row['train_loss']:.4f} valF1={val['macro_f1']:.4f} valAcc={val['accuracy']:.4f}")
        if val["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val["macro_f1"]; best_state = cpu_state_dict(model); best_epoch = epoch; patience = 0
        else:
            patience += 1
            if patience >= MACRO_FT_PATIENCE:
                print(f"Macro early stop at epoch {epoch}; best={best_epoch}")
                break

    if best_state is None: raise RuntimeError("No macro checkpoint")
    model.load_state_dict(best_state)
    ensure_dir(out_path.parent)
    torch.save(dict(state_dict=best_state, best_val_macro_f1=float(best_f1), best_epoch=best_epoch,
                    project_version=PROJECT_VERSION), out_path)
    return dict(model=model, best_val_macro_f1=float(best_f1), best_epoch=best_epoch,
                history=pd.DataFrame(history))


def fresh_encoder(input_dim: int, seed: int) -> MacroEncoder:
    seed_everything(seed)
    return MacroEncoder(input_dim=input_dim)

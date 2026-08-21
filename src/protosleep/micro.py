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

# =============================================================================
# 5. METRICS + MICRO TRAINING
# =============================================================================
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(NUM_CLASSES)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        per_class_precision=p.astype(float),
        per_class_recall=r.astype(float),
        per_class_f1=f.astype(float),
        support=s.astype(int),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
    )


def _micro_logits(model, x):
    out = model(x)
    return out["logits"] if isinstance(out, dict) else out


def evaluate_micro_loader(model, loader, device=DEVICE) -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with amp_context(device):
                logits = _micro_logits(model, x)
            ys.append(y.numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    m = classification_metrics(y, p)
    m["y_true"] = y
    m["y_pred"] = p
    return m


def train_micro_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      class_weights: torch.Tensor, out_path: Path,
                      proto_cfg: Optional[Dict[str, float]] = None,
                      seed: int = SEED) -> Dict[str, Any]:
    seed_everything(seed)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=MICRO_LR, weight_decay=MICRO_WEIGHT_DECAY)
    scaler = make_grad_scaler(DEVICE)
    best_f1 = -np.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history = []
    rng = np.random.default_rng(seed + 991)

    for epoch in range(1, MICRO_EPOCHS + 1):
        model.train()
        total = 0.0
        n_seen = 0
        terms_acc = dict(stage=0.0, commit=0.0, balance=0.0, separation=0.0, micro_mask=0.0)
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            local_mask = None
            if isinstance(model, ProtoAttnSleep) and model.micro_mask_predictor is not None and epoch > MICRO_MASK_WARMUP_EPOCHS:
                # Need N after MRCNN; primary default keeps this OFF, so no extra forward in normal run.
                with torch.no_grad():
                    tmp_afr = model.mrcnn(x)
                    tmp_proto = model.prototype_bank(tmp_afr)
                    n_tokens = tmp_proto["tokens"].size(1)
                local_mask = make_random_local_masks(x.size(0), n_tokens, MICRO_MASK_RATIO, rng, DEVICE)

            with amp_context(DEVICE):
                out = model(x, micro_mask=local_mask) if isinstance(model, ProtoAttnSleep) else model(x)
                logits = out["logits"] if isinstance(out, dict) else out
                stage_loss = criterion(logits, y)
                loss = stage_loss
                regs = dict(commit=torch.tensor(0., device=DEVICE),
                            balance=torch.tensor(0., device=DEVICE),
                            separation=torch.tensor(0., device=DEVICE))
                micro_mask_loss = torch.tensor(0., device=DEVICE)

                if isinstance(model, ProtoAttnSleep):
                    assert proto_cfg is not None
                    regs = prototype_regularizers(out, sep_margin=proto_cfg["sep_margin"])
                    loss = (loss
                            + proto_cfg["lambda_commit"] * regs["commit"]
                            + proto_cfg["lambda_balance"] * regs["balance"]
                            + proto_cfg["lambda_sep"] * regs["separation"])
                    if local_mask is not None:
                        teacher = out["assignments"].detach()
                        log_q = F.log_softmax(out["micro_mask_logits"], dim=-1)
                        per = F.kl_div(log_q, teacher, reduction="none").sum(-1)
                        micro_mask_loss = per[local_mask].mean()
                        loss = loss + MICRO_MASK_LAMBDA * micro_mask_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if isinstance(model, ProtoAttnSleep):
                model.project_prototypes_()

            bs = y.size(0)
            n_seen += bs
            total += float(loss.detach().cpu()) * bs
            terms_acc["stage"] += float(stage_loss.detach().cpu()) * bs
            for k in ("commit", "balance", "separation"):
                terms_acc[k] += float(regs[k].detach().cpu()) * bs
            terms_acc["micro_mask"] += float(micro_mask_loss.detach().cpu()) * bs

        val = evaluate_micro_loader(model, val_loader, DEVICE)
        row = dict(epoch=epoch, train_loss=total/max(1, n_seen), val_macro_f1=val["macro_f1"], val_acc=val["accuracy"])
        for k, v in terms_acc.items():
            row[f"train_{k}"] = v/max(1, n_seen)
        history.append(row)
        print(f"micro e{epoch:03d} train={row['train_loss']:.4f} valF1={val['macro_f1']:.4f} valAcc={val['accuracy']:.4f}")

        if val["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
            if patience >= MICRO_PATIENCE:
                print(f"Micro early stop at epoch {epoch}; best={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Micro training produced no checkpoint")
    model.load_state_dict(best_state)
    ensure_dir(out_path.parent)
    torch.save(dict(state_dict=best_state, best_val_macro_f1=float(best_f1), best_epoch=best_epoch,
                    proto_cfg=proto_cfg, project_version=PROJECT_VERSION), out_path)
    return dict(model=model, best_val_macro_f1=float(best_f1), best_epoch=best_epoch, history=pd.DataFrame(history))


def train_or_select_proto_model(train_loader, val_loader, class_weights, fold_dir: Path) -> Dict[str, Any]:
    trials = PROTO_TRIALS if RUN_PROTO_HPARAM_SEARCH else PROTO_TRIALS[:1]
    results = []
    for i, cfg in enumerate(trials):
        print("\nPrototype trial", cfg)
        seed_everything(SEED + 1000 + i)
        model = ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)
        model.apply(init_attnsleep_weights)
        path = fold_dir / "checkpoints" / f"B_proto_{cfg['name']}.pt"
        result = train_micro_model(model, train_loader, val_loader, class_weights, path,
                                   proto_cfg=cfg, seed=SEED + 1000 + i)
        # Keep validation-search trials on CPU so multiple candidates do not accumulate GPU memory.
        result["model"] = result["model"].cpu()
        results.append(dict(cfg=cfg, **result))
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    best = max(results, key=lambda z: z["best_val_macro_f1"])
    print(f"Selected prototype trial {best['cfg']['name']} by validation Macro-F1={best['best_val_macro_f1']:.4f}")
    return best

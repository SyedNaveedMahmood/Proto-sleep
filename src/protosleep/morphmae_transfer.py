from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    GRAD_CLIP_NORM,
    MICRO_EPOCHS,
    MICRO_LR,
    MICRO_PATIENCE,
    MICRO_WEIGHT_DECAY,
)
from .micro import evaluate_micro_loader
from .utils import DEVICE, amp_context, cpu_state_dict, ensure_dir, make_grad_scaler, seed_everything


@dataclass(frozen=True)
class TransferRecipe:
    """Theory-driven downstream transfer recipe for a pretrained MRCNN."""

    name: str
    encoder_lr: float
    head_lr: float = MICRO_LR
    encoder_weight_decay: float = 1e-4
    head_weight_decay: float = MICRO_WEIGHT_DECAY
    warmup_epochs: int = 5
    freeze_encoder: bool = False


# These are deliberately not a broad hyperparameter grid.
# MorphMAE-v2 itself used lr=2e-4. The naive downstream runner immediately applies 1e-3
# to the pretrained MRCNN. We bracket a transfer-friendly encoder LR at 0.5x and 0.1x
# the pretraining LR while keeping the supervised TCE/classifier LR unchanged.
MAE_PROBE = TransferRecipe(
    name="mae_probe",
    encoder_lr=0.0,
    warmup_epochs=MICRO_EPOCHS,
    freeze_encoder=True,
)
MAE_STAGE_1E4 = TransferRecipe(
    name="mae_stage_1e4",
    encoder_lr=1e-4,
    warmup_epochs=5,
)
MAE_STAGE_2E5 = TransferRecipe(
    name="mae_stage_2e5",
    encoder_lr=2e-5,
    warmup_epochs=5,
)


def recipe_dict(recipe: TransferRecipe) -> Dict[str, Any]:
    return asdict(recipe)


def set_mrcnn_trainable(model: nn.Module, trainable: bool) -> None:
    if not hasattr(model, "mrcnn"):
        raise AttributeError("Expected model.mrcnn for MorphMAE transfer")
    for p in model.mrcnn.parameters():
        p.requires_grad = bool(trainable)


def mrcnn_relative_drift(initial_state: Dict[str, torch.Tensor], model: nn.Module) -> float:
    """Relative L2 drift of MRCNN parameters/buffers from the pretrained initialization."""

    final = model.mrcnn.state_dict()
    num = 0.0
    den = 0.0
    for key, before in initial_state.items():
        if key not in final:
            raise RuntimeError(f"Missing MRCNN state key after transfer: {key}")
        a = before.detach().cpu().float()
        b = final[key].detach().cpu().float()
        num += float(torch.sum((b - a) ** 2))
        den += float(torch.sum(a ** 2))
    if den <= 0:
        return float("nan")
    return float(np.sqrt(num / den))


def train_attnsleep_transfer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    out_path: Path,
    recipe: TransferRecipe,
    seed: int,
) -> Dict[str, Any]:
    """Train AttnSleep with a transfer-friendly MRCNN schedule.

    During warmup the pretrained MRCNN is frozen *and kept in eval mode* so neither
    parameters nor BatchNorm running statistics drift. For staged recipes it is then
    unfrozen with a lower encoder LR while TCE/classifier continue at the standard LR.
    The optimizer is created once, so Adam state for the supervised head is preserved
    across unfreezing.
    """

    seed_everything(int(seed))
    model = model.to(DEVICE)
    initial_mrcnn = cpu_state_dict(model.mrcnn)

    encoder_params = list(model.mrcnn.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("mrcnn.")]
    if not encoder_params or not head_params:
        raise RuntimeError("Expected non-empty MRCNN and non-MRCNN parameter groups")

    encoder_frozen = bool(recipe.freeze_encoder or recipe.warmup_epochs > 0)
    set_mrcnn_trainable(model, not encoder_frozen)

    optimizer = torch.optim.Adam(
        [
            {
                "params": encoder_params,
                "lr": 0.0 if encoder_frozen else float(recipe.encoder_lr),
                "weight_decay": float(recipe.encoder_weight_decay),
                "name": "mrcnn",
            },
            {
                "params": head_params,
                "lr": float(recipe.head_lr),
                "weight_decay": float(recipe.head_weight_decay),
                "name": "head",
            },
        ]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scaler = make_grad_scaler(DEVICE)

    best_f1 = -np.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history = []
    unfreeze_epoch = None if recipe.freeze_encoder else int(recipe.warmup_epochs) + 1

    for epoch in range(1, MICRO_EPOCHS + 1):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            set_mrcnn_trainable(model, True)
            optimizer.param_groups[0]["lr"] = float(recipe.encoder_lr)
            encoder_frozen = False

        model.train()
        if encoder_frozen:
            # Critical: model.train() would otherwise update BN running stats even when
            # gradients are disabled, silently erasing part of the pretrained state.
            model.mrcnn.eval()

        total = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(DEVICE):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.shape[0])
            total += float(loss.detach().cpu()) * bs
            n_seen += bs

        val = evaluate_micro_loader(model, val_loader, DEVICE)
        row = {
            "epoch": int(epoch),
            "train_loss": total / max(1, n_seen),
            "val_macro_f1": float(val["macro_f1"]),
            "val_acc": float(val["accuracy"]),
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "encoder_frozen": bool(encoder_frozen),
        }
        history.append(row)
        print(
            f"transfer {recipe.name} e{epoch:03d} train={row['train_loss']:.4f} "
            f"valF1={row['val_macro_f1']:.4f} valAcc={row['val_acc']:.4f} "
            f"encLR={row['encoder_lr']:.1e} frozen={int(row['encoder_frozen'])}"
        )

        if row["val_macro_f1"] > best_f1 + 1e-6:
            best_f1 = row["val_macro_f1"]
            best_epoch = int(epoch)
            best_state = cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
            if patience >= MICRO_PATIENCE:
                print(f"Transfer early stop at epoch {epoch}; best={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Transfer training produced no checkpoint")

    model.load_state_dict(best_state)
    best_metrics = evaluate_micro_loader(model, val_loader, DEVICE)
    drift = mrcnn_relative_drift(initial_mrcnn, model)

    ensure_dir(Path(out_path).parent)
    torch.save(
        {
            "state_dict": best_state,
            "best_val_macro_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "recipe": recipe_dict(recipe),
            "mrcnn_relative_drift": float(drift),
        },
        out_path,
    )

    return {
        "model": model,
        "best_val_macro_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "mrcnn_relative_drift": float(drift),
        "history": pd.DataFrame(history),
        "recipe": recipe_dict(recipe),
    }

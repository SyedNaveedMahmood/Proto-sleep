from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .config import (
    EXPECTED_FS,
    EXPECTED_SAMPLES_PER_EPOCH,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    PROJECT_VERSION,
)
from .data import EpochDataset, Recording, _canonicalize_epoch_shape
from .legacy_wavesleepnet_objective import HistoricalProtoObjective, finite_tensor
from .mist import extract_mrcnn_state_dict
from .morphmae_bridge import discover_npz_subject_files
from .utils import cpu_state_dict, seed_everything


def load_recordings_for_subjects(
    data_dir: Path | str,
    subject_ids: Sequence[int],
) -> Tuple[List[Recording], Dict[str, Any]]:
    """Load only explicitly permitted Sleep-EDF subjects.

    This helper is intentionally stricter than loading the whole dataset and selecting a
    subset afterwards. Files belonging to subjects outside ``subject_ids`` are never opened,
    which lets the historical recovery audit keep the designated test subject physically out
    of both training and validation I/O.
    """
    requested = sorted({int(x) for x in subject_ids})
    if not requested:
        raise ValueError("subject_ids must not be empty")

    by_subject = discover_npz_subject_files(data_dir)
    missing = sorted(set(requested) - set(by_subject))
    if missing:
        raise RuntimeError(f"Missing requested Sleep-EDF subjects: {missing}")

    recordings: List[Recording] = []
    opened_files: List[str] = []
    for sid in requested:
        for path in by_subject[sid]:
            with np.load(path, allow_pickle=False) as d:
                if "x" not in d.files or "y" not in d.files:
                    raise KeyError(f"{path.name}: expected x and y arrays")
                x = _canonicalize_epoch_shape(d["x"], path.name)
                y = np.asarray(d["y"]).reshape(-1).astype(np.int64, copy=False)
                if "fs" in d.files:
                    fs = int(round(float(np.asarray(d["fs"]).reshape(-1)[0])))
                else:
                    if x.shape[-1] % 30 != 0:
                        raise ValueError(f"{path.name}: cannot infer fs from {x.shape[-1]} samples/epoch")
                    fs = int(x.shape[-1] // 30)

            if x.shape[0] != y.shape[0]:
                raise ValueError(f"{path.name}: x has {x.shape[0]} epochs but y has {y.shape[0]}")
            if x.shape[-1] != EXPECTED_SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"{path.name}: expected {EXPECTED_SAMPLES_PER_EPOCH} samples/epoch; got {x.shape[-1]}"
                )
            if fs != EXPECTED_FS:
                raise ValueError(f"{path.name}: expected {EXPECTED_FS} Hz; got {fs} Hz")
            if not np.isfinite(x).all():
                raise ValueError(f"{path.name}: non-finite EEG values")
            bad = np.setdiff1d(np.unique(y), np.arange(NUM_CLASSES))
            if bad.size:
                raise ValueError(f"{path.name}: labels outside 0..{NUM_CLASSES - 1}: {bad.tolist()}")

            recordings.append(
                Recording(
                    path=str(path),
                    recording_id=path.stem,
                    subject_id=sid,
                    x=x,
                    y=y,
                    fs=fs,
                )
            )
            opened_files.append(str(path))

    actual = sorted({r.subject_id for r in recordings})
    if actual != requested:
        raise RuntimeError(f"Loaded subject mismatch: expected {requested}, got {actual}")
    return recordings, {
        "subjects": actual,
        "opened_files": opened_files,
        "n_recordings": len(recordings),
        "n_epochs": int(sum(r.n_epochs for r in recordings)),
    }


def make_seeded_epoch_loader(
    recordings: Sequence[Recording],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        EpochDataset(list(recordings)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=generator,
    )


def _model_outputs(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    out = model(x)
    if torch.is_tensor(out):
        outputs = [out]
    elif isinstance(out, (list, tuple)) and out and all(torch.is_tensor(v) for v in out):
        outputs = list(out)
    else:
        raise TypeError(f"Historical ProtoPNet returned unsupported output type: {type(out)!r}")
    for i, logits in enumerate(outputs):
        if logits.ndim != 2 or logits.shape[-1] != NUM_CLASSES:
            raise RuntimeError(
                f"Historical output {i} has shape {tuple(logits.shape)}, expected [B,{NUM_CLASSES}]"
            )
        finite_tensor(f"logits[{i}]", logits)
    return outputs


def _historical_loss_for_outputs(
    objective: HistoricalProtoObjective,
    outputs: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total: torch.Tensor | None = None
    final_terms: Dict[str, torch.Tensor] = {}
    for logits in outputs:
        loss, terms = objective(logits, labels)
        finite_tensor("historical_loss", loss)
        total = loss if total is None else total + loss
        final_terms = terms
    if total is None:
        raise RuntimeError("Historical model produced no outputs")
    return total, final_terms


def evaluate_legacy_model(
    model: torch.nn.Module,
    loader: DataLoader,
    trainer_module: Any,
    cfg: Mapping[str, Any],
    device: torch.device | str,
) -> Dict[str, Any]:
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    objective = HistoricalProtoObjective(trainer_module, model, cfg)

    total_loss = 0.0
    n_seen = 0
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            outputs = _model_outputs(model, x)
            loss, _ = _historical_loss_for_outputs(objective, outputs, y)
            bs = int(y.shape[0])
            total_loss += float(loss.detach().cpu()) * bs
            n_seen += bs
            ys.append(y.detach().cpu().numpy())
            ps.append(outputs[-1].argmax(dim=-1).detach().cpu().numpy())

    if n_seen == 0:
        raise RuntimeError("Validation loader is empty")
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "loss": float(total_loss / n_seen),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_epochs": int(n_seen),
    }


def verify_mae_checkpoint_for_fold(
    checkpoint: Path | str,
    fold: int,
    train_subjects: Sequence[int],
) -> Dict[str, Any]:
    """Fail closed unless the MorphMAE checkpoint declares this exact fold train split."""
    _, meta = extract_mrcnn_state_dict(checkpoint)
    expected = sorted(int(x) for x in train_subjects)

    declared = None
    for key in ("train_subjects", "pretrain_subjects", "subjects"):
        if key in meta:
            declared = sorted(int(x) for x in meta[key])
            break
    if declared is None:
        raise RuntimeError(f"MorphMAE checkpoint {checkpoint} has no train-subject metadata")
    if declared != expected:
        raise RuntimeError(
            f"MorphMAE checkpoint subjects {declared} do not equal fold-{fold} training subjects {expected}"
        )
    if "fold" in meta and int(meta["fold"]) != int(fold):
        raise RuntimeError(
            f"MorphMAE checkpoint declares fold={meta['fold']}, but historical recovery requested fold={fold}"
        )
    return meta


def historical_training_protocol(
    cfg: Mapping[str, Any],
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> Dict[str, Any]:
    tp = cfg["training_params"]
    es = tp["early_stopping"]
    max_epochs = int(tp["max_epochs"] if max_epochs_override is None else max_epochs_override)
    patience = int(es["patience"] if patience_override is None else patience_override)
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    return {
        "batch_size": int(tp["batch_size"]),
        "lr": float(tp["lr"]),
        "weight_decay": float(tp["weight_decay"]),
        "max_epochs": max_epochs,
        "patience": patience,
        "historical_max_epochs": int(tp["max_epochs"]),
        "historical_patience": int(es["patience"]),
        "early_stopping_mode_in_legacy_config": str(es.get("mode", "unknown")),
        "selection_metric": "validation_macro_f1",
        "precision": "fp32",
        "optimizer": "Adam",
        "gradient_clipping": False,
        "amp": False,
        "max_epochs_overridden": max_epochs_override is not None,
        "patience_overridden": patience_override is not None,
    }


def train_legacy_model(
    model: torch.nn.Module,
    train_recordings: Sequence[Recording],
    val_recordings: Sequence[Recording],
    trainer_module: Any,
    cfg: Mapping[str, Any],
    device: torch.device | str,
    seed: int,
    checkpoint_path: Path | str,
    history_path: Path | str | None = None,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> Dict[str, Any]:
    """Train recovered ProtoPNet with the exact archived prototype objective.

    Architecture, objective, Adam LR/weight decay, batch size and FP32 precision are taken
    from the audited historical implementation/config. Model selection is intentionally
    validation Macro-F1 so the recovered A3/A4 mechanism test uses the same selection target
    as the current MIST stability audit; this is recorded as a protocol adaptation rather
    than described as byte-for-byte historical trainer behavior.
    """
    protocol = historical_training_protocol(cfg, max_epochs_override, patience_override)
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        history_path = Path(history_path).expanduser().resolve()
        history_path.parent.mkdir(parents=True, exist_ok=True)

    seed_everything(int(seed))
    model = model.to(device)
    objective = HistoricalProtoObjective(trainer_module, model, cfg)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=protocol["lr"],
        weight_decay=protocol["weight_decay"],
    )
    train_loader = make_seeded_epoch_loader(
        train_recordings,
        protocol["batch_size"],
        shuffle=True,
        seed=int(seed),
    )
    val_loader = make_seeded_epoch_loader(
        val_recordings,
        protocol["batch_size"],
        shuffle=False,
        seed=int(seed),
    )

    best_f1 = -math.inf
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    patience_count = 0
    history_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for epoch in range(1, protocol["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        term_sums: Dict[str, float] = {}

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = _model_outputs(model, x)
            loss, terms = _historical_loss_for_outputs(objective, outputs, y)
            loss.backward()

            n_grad = 0
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                finite_tensor(f"grad:{name}", p.grad)
                n_grad += 1
            if n_grad == 0:
                raise RuntimeError("Historical objective produced no gradients during training")
            optimizer.step()

            bs = int(y.shape[0])
            n_seen += bs
            total_loss += float(loss.detach().cpu()) * bs
            for name, value in terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + float(value.detach().cpu()) * bs

        val = evaluate_legacy_model(model, val_loader, trainer_module, cfg, device)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(total_loss / max(1, n_seen)),
            "val_loss": float(val["loss"]),
            "val_macro_f1": float(val["macro_f1"]),
            "val_accuracy": float(val["accuracy"]),
        }
        for name, total in sorted(term_sums.items()):
            row[f"train_{name}"] = float(total / max(1, n_seen))
        history_rows.append(row)
        print(
            f"legacy e{epoch:04d} train={row['train_loss']:.4f} "
            f"valLoss={row['val_loss']:.4f} valF1={row['val_macro_f1']:.4f} "
            f"valAcc={row['val_accuracy']:.4f}"
        )

        if row["val_macro_f1"] > best_f1 + 1e-6:
            best_f1 = float(row["val_macro_f1"])
            best_epoch = int(epoch)
            best_state = cpu_state_dict(model)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= protocol["patience"]:
                print(
                    f"Historical recovery early stop at epoch {epoch}; "
                    f"best epoch={best_epoch} valF1={best_f1:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("Historical recovery training produced no best checkpoint")

    model.load_state_dict(best_state, strict=True)
    final_val = evaluate_legacy_model(model, val_loader, trainer_module, cfg, device)
    history = pd.DataFrame(history_rows)
    if history_path is not None:
        history.to_csv(history_path, index=False)

    payload = {
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "best_val_loss": float(final_val["loss"]),
        "best_val_accuracy": float(final_val["accuracy"]),
        "seed": int(seed),
        "protocol": protocol,
        "project_version": PROJECT_VERSION,
    }
    torch.save(payload, checkpoint_path)

    return {
        "model": model,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "best_val_loss": float(final_val["loss"]),
        "best_val_accuracy": float(final_val["accuracy"]),
        "history": history,
        "protocol": protocol,
        "seconds": float(time.perf_counter() - t0),
        "checkpoint": str(checkpoint_path),
    }

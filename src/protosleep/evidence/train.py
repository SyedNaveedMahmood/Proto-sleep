"""Validation-only trainers with epoch-boundary resume and atomic completion markers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .crf import EvidenceCRF
from .data import Epochs, STAGES, contiguous_runs, digest_file, load_split, validate_manifest, write_json
from .model import EvidenceConfig, EvidenceNet, build_bank


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 123
    epochs: int = 60
    patience: int = 12
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    log_every: int = 100

    def __post_init__(self):
        if min(self.epochs, self.patience, self.batch_size) < 1 or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid train config")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def atomic_torch(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def safe_load(path: Path):
    # All checkpoints contain tensors and primitive containers only.
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def source_digest():
    h = hashlib.sha256()
    # Include upstream package code too: the optional AttnSleep baseline imports it.
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        h.update(str(path.relative_to(Path(__file__).resolve().parents[1])).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def fingerprint(doc, train_nights, val_nights, model_cfg, train_cfg, arm):
    payload = {"manifest": doc, "files": {n.key: n.sha256 for n in train_nights + val_nights},
               "model": asdict(model_cfg), "training": asdict(train_cfg), "arm": arm,
               "source_sha256": source_digest(), "schema": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def model_from_payload(payload):
    if payload["arm"] == "attnsleep":
        from ..attnsleep import AttnSleepBaseline
        model = AttnSleepBaseline()
    else:
        model = EvidenceNet(EvidenceConfig(**payload["model_config"]), payload["bank"])
    model.load_state_dict(payload["model"], strict=True)
    return model


def forward(model, x):
    result = model(x)
    if isinstance(result, dict):
        return result
    return {"logits": result, "aux_loss": result.sum() * 0.0}


@torch.no_grad()
def night_logits(model, night, device, batch_size):
    model.eval()
    outs = []
    for start in range(0, len(night.x), batch_size):
        x = torch.from_numpy(night.x[start:start + batch_size]).to(device)
        outs.append(forward(model, x)["logits"].float().cpu())
    return torch.cat(outs)


def scores(y, probabilities):
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) == 0 or probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid evaluation arrays")
    predicted = probabilities.argmax(-1)
    per_stage = f1_score(y, predicted, labels=list(range(5)), average=None, zero_division=0)
    kappa = float(cohen_kappa_score(y, predicted, labels=list(range(5))))
    return {"accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(per_stage.mean()),
            "per_stage_f1": dict(zip(STAGES, map(float, per_stage))),
            "kappa": kappa if np.isfinite(kappa) else None,
            "confusion": confusion_matrix(y, predicted, labels=list(range(5))).tolist(), "n_epochs": len(y)}


def evaluate(model, nights, device, batch_size):
    model.eval()
    ys, ps = [], []
    for n in nights:
        logits = night_logits(model, n, device, batch_size)
        valid = n.y >= 0
        ys.extend(n.y[valid].tolist())
        ps.extend(logits[valid].softmax(-1).numpy().tolist())
    return scores(ys, ps)


def fit_local(manifest: Path, output: Path, model_cfg: EvidenceConfig, cfg: TrainConfig,
              device: str = "cpu", resume: bool = False, attnsleep: bool = False):
    doc, _ = validate_manifest(manifest)
    print("MIST-Evidence v1 | validation-only | precision=FP32 | device=" + str(device), flush=True)
    print("Test arrays are never opened. EDF-20 results are development results, not pristine confirmation.", flush=True)
    train_nights, val_nights = load_split(manifest, "train"), load_split(manifest, "val")
    arm = "attnsleep" if attnsleep else model_cfg.arm
    identity, provenance = fingerprint(doc, train_nights, val_nights, model_cfg, cfg, arm)
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path, complete_path = (output / n for n in ("last.pt", "best.pt", "COMPLETE.json"))
    if last_path.exists() or best_path.exists() or complete_path.exists():
        if not resume:
            raise FileExistsError("Existing run. Use --resume with the identical recipe or a NEW output directory.")
        if not last_path.exists():
            raise RuntimeError("No epoch-boundary last.pt. Do not silently treat an intermediate best.pt as complete.")
        previous = safe_load(last_path)
        if previous["identity"] != identity:
            raise ValueError("Resume rejected: code/data/config fingerprint changed")
        if complete_path.exists():
            done = json.loads(complete_path.read_text())
            if done["identity"] != identity or digest_file(best_path) != done["best_sha256"]:
                raise RuntimeError("Completion/checkpoint identity mismatch")
            print("VERIFIED COMPLETE: no training repeated", flush=True)
            return done
    else:
        previous = None
    seed_all(cfg.seed)
    data = Epochs(train_nights)
    counts = np.bincount(np.concatenate([n.y[n.y >= 0] for n in train_nights]), minlength=5)
    if np.any(counts == 0):
        raise ValueError("Every training stage must be present")
    weights = counts.sum() / counts.astype(np.float64)
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=str(device).startswith("cuda"))
    bank = previous["bank"] if previous else (None if attnsleep else build_bank(train_nights, model_cfg, cfg.seed))
    if previous:
        model = model_from_payload(previous)
    elif attnsleep:
        from ..attnsleep import AttnSleepBaseline, init_attnsleep_weights
        model = AttnSleepBaseline()
        model.apply(init_attnsleep_weights)
    else:
        model = EvidenceNet(model_cfg, bank)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best, best_epoch, stale, start = [], -1.0, 0, 0, 1
    if previous:
        optimizer.load_state_dict(previous["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu() if k == "step" else v.to(device)
        history, best, best_epoch, stale = previous["history"], previous["best"], previous["best_epoch"], previous["stale"]
        start = previous["epoch"] + 1
        torch.set_rng_state(previous["rng_cpu"])
        generator.set_state(previous["shuffle_rng"])
        if torch.cuda.is_available() and previous["rng_cuda"]:
            torch.cuda.set_rng_state_all(previous["rng_cuda"])
    print(f"arm={arm}; train_epochs={len(data)}; steps_per_epoch={len(loader)}; params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    write_json(output / "provenance.json", {"identity": identity, **provenance,
               "test_arrays_opened": False, "clinical_validation": False})
    for epoch in range(start, cfg.epochs + 1):
        if stale >= cfg.patience:
            break
        model.train()
        total, nobs = 0.0, 0
        then = time.monotonic()
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = forward(model, x)
            loss = F.cross_entropy(result["logits"].float(), y, weight=class_weights)
            if not attnsleep:
                loss = loss + model_cfg.aux_morph_weight * result["aux_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss epoch={epoch} step={step}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            nobs += len(x)
            if cfg.log_every > 0 and step % cfg.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} loss={float(loss.detach()):.5f} grad_norm={float(norm):.3f}", flush=True)
        metrics = evaluate(model, val_nights, device, cfg.batch_size)
        improved = metrics["macro_f1"] > best
        if improved:
            best, best_epoch, stale = metrics["macro_f1"], epoch, 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": total / nobs, "val_macro_f1": metrics["macro_f1"],
               "seconds": time.monotonic() - then, "steps": len(loader)}
        history.append(row)
        payload = {"schema": 1, "identity": identity, "arm": arm, "model_config": asdict(model_cfg),
                   "training_config": asdict(cfg), "bank": bank, "model": cpu_state(model),
                   "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                   "stale": stale, "history": history, "rng_cpu": torch.get_rng_state(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "shuffle_rng": generator.get_state(), "provenance": provenance}
        if improved:
            atomic_torch(best_path, payload)
        atomic_torch(last_path, payload)
        write_json(output / "history.json", history)
        print(f"epoch={epoch:03d} train={total/nobs:.5f} valF1={metrics['macro_f1']:.5f} best={best:.5f} seconds={row['seconds']:.1f}", flush=True)
    if not best_path.exists():
        raise RuntimeError("Training produced no best checkpoint")
    best_payload = safe_load(best_path)
    model.load_state_dict(best_payload["model"], strict=True)
    final = evaluate(model, val_nights, device, cfg.batch_size)
    done = {"identity": identity, "status": "complete", "arm": arm, "best_epoch": best_epoch,
            "epochs_completed": len(history), "validation": final, "best_sha256": digest_file(best_path),
            "test_arrays_opened": False, "independent_confirmation": False,
            "bank_train_subjects": bank["train_subjects"] if bank else []}
    write_json(complete_path, done)
    print(json.dumps(done, indent=2), flush=True)
    return done


def fit_context(manifest: Path, local_checkpoint: Path, output: Path, epochs: int = 20,
                patience: int = 5, lr: float = 0.02, assume_contiguous: bool = False,
                device: str = "cpu", resume: bool = False):
    """Fit only CRF transition/start/end parameters over fixed local model logits.

    Float64 CPU forward/backward; full contiguous nights, not shuffled epoch chains.
    This is an ablation, not a claim that adjacent-state transitions explain physiology.
    """
    if min(epochs, patience) < 1 or lr <= 0:
        raise ValueError("Invalid context config")
    payload = safe_load(local_checkpoint)
    train_nights, val_nights = load_split(manifest, "train"), load_split(manifest, "val")
    doc, _ = validate_manifest(manifest)
    expected_files = payload["provenance"]["files"]
    if doc != payload["provenance"]["manifest"] or {n.key: n.sha256 for n in train_nights + val_nights} != expected_files:
        raise ValueError("Context data/split differs from the local checkpoint")
    for n in train_nights + val_nights:
        contiguous_runs(n, assume_contiguous)
    output.mkdir(parents=True, exist_ok=True)
    identity_data = {"local_sha256": digest_file(local_checkpoint), "epochs": epochs,
                     "patience": patience, "lr": lr, "assume_contiguous": assume_contiguous,
                     "source": source_digest()}
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    last, best_path, done_path = (output / n for n in ("last.pt", "best.pt", "COMPLETE.json"))
    previous = None
    if any(p.exists() for p in (last, best_path, done_path)):
        if not resume or not last.exists():
            raise FileExistsError("Context run exists; requires epoch-boundary --resume")
        previous = safe_load(last)
        if previous["identity"] != identity:
            raise ValueError("Context resume fingerprint mismatch")
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done["best_sha256"] != digest_file(best_path):
                raise ValueError("Context checkpoint changed")
            return done
    model = model_from_payload(payload).to(device).eval()
    def cache(nights):
        pairs = []
        for n in nights:
            logits = night_logits(model, n, device, 32).double()
            for ix in contiguous_runs(n, assume_contiguous):
                pairs.append((logits[ix], torch.from_numpy(n.y[ix])))
        return pairs
    train, val = cache(train_nights), cache(val_nights)
    del model
    crf = EvidenceCRF()
    optimizer = torch.optim.Adam(crf.parameters(), lr=lr)
    start, best, best_epoch, stale, history = 1, -1.0, 0, 0, []
    if previous:
        crf.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start, best, best_epoch, stale, history = previous["epoch"]+1, previous["best"], previous["best_epoch"], previous["stale"], previous["history"]
    def assess():
        with torch.no_grad():
            return scores(torch.cat([y for _, y in val]).numpy(),
                          torch.cat([crf.log_marginals(e).exp() for e, _ in val]).numpy())
    for epoch in range(start, epochs + 1):
        if stale >= patience:
            break
        count, loss_sum = sum(len(e) for e, _ in train), 0.0
        optimizer.zero_grad(set_to_none=True)
        # Backpropagate per chain to keep memory bounded while exactly accumulating
        # the epoch-weighted full training-set NLL before one optimizer update.
        for e, y in train:
            loss = crf.nll(e, y) / count
            if not torch.isfinite(loss):
                raise FloatingPointError("CRF NLL is not finite")
            loss.backward()
            loss_sum += float(loss.detach())
        nn.utils.clip_grad_norm_(crf.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        metrics = assess()
        improved = metrics["macro_f1"] > best
        if improved:
            best, best_epoch, stale = metrics["macro_f1"], epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_nll_per_epoch": loss_sum, "val_macro_f1": metrics["macro_f1"]})
        cp = {"schema": 1, "identity": identity, "local_sha256": identity_data["local_sha256"],
              "model": cpu_state(crf), "optimizer": optimizer.state_dict(), "epoch": epoch,
              "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
              "assume_contiguous": assume_contiguous, "bound": crf.bound}
        if improved:
            atomic_torch(best_path, cp)
        atomic_torch(last, cp)
        print(f"context epoch={epoch} nll={loss_sum:.5f} valF1={metrics['macro_f1']:.5f}", flush=True)
    crf.load_state_dict(safe_load(best_path)["model"])
    done = {"status": "complete", "identity": identity, "validation": assess(), "best_epoch": best_epoch,
            "best_sha256": digest_file(best_path), "test_arrays_opened": False,
            "continuity_assumed": assume_contiguous, "inference": "offline_bidirectional_marginals"}
    write_json(done_path, done)
    return done

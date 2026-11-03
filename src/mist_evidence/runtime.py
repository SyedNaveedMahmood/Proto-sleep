"""Training and validation with completion markers and epoch-boundary resume."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .data import Recording, blocks, fingerprint
from .model import EvidenceModel


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    patience: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    core_epochs: int = 16
    encode_batch: int = 16
    grad_clip: float = 5.0
    seed: int = 123
    deterministic: bool = True

    def __post_init__(self):
        if min(self.epochs, self.patience, self.core_epochs, self.encode_batch) < 1 or self.lr <= 0:
            raise ValueError("invalid training settings")


def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tensor_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_trusted(path: Path):
    """Only use with this runner's local, trusted checkpoints (contains RNG objects)."""
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def evaluate(model: EvidenceModel, recordings: list[Recording], device: torch.device, batch: int):
    model.eval()
    truth, predicted, rows, sums = [], [], [], []
    subject_data = {}
    for r in recordings:
        encoded = [model.encode(torch.from_numpy(r.x[i:i+batch]).to(device)).cpu()
                   for i in range(0, len(r.y), batch)]
        features = torch.cat(encoded).to(device)
        for lo, hi in r.segments():
            emissions = model.emissions(features[lo:hi][None])[0]
            pred, probs = model.predict(emissions)
            y = torch.from_numpy(r.y[lo:hi]).to(device)
            valid = torch.ones((1, hi-lo), dtype=torch.bool, device=device)
            nll = float(model.loss(emissions[None], y[None], valid))
            sums.append((nll, hi-lo))
            pred_np, prob_np = pred.cpu().numpy(), probs.cpu().numpy()
            truth.extend(r.y[lo:hi].tolist())
            predicted.extend(pred_np.tolist())
            sy, sp = subject_data.setdefault(r.subject, ([], []))
            sy.extend(r.y[lo:hi].tolist()); sp.extend(pred_np.tolist())
            for j, (yt, yp, probability) in enumerate(zip(r.y[lo:hi], pred_np, prob_np), lo):
                rows.append({"recording": Path(r.path).name, "subject": r.subject,
                             "array_epoch": j, "original_epoch": int(r.indices[j]) if r.indices_known else None,
                             "truth": int(yt), "pred": int(yp),
                             **{f"p_{k}": float(v) for k, v in enumerate(probability)}})
    kappa = float(cohen_kappa_score(truth, predicted, labels=list(range(5))))
    result = {"accuracy": float(accuracy_score(truth, predicted)),
              "macro_f1": float(f1_score(truth, predicted, labels=list(range(5)), average="macro", zero_division=0)),
              "kappa": kappa if np.isfinite(kappa) else None,
              "per_stage_f1": f1_score(truth, predicted, labels=list(range(5)), average=None, zero_division=0).tolist(),
              "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(5))).tolist(),
              "nll": sum(loss*n for loss, n in sums) / sum(n for _, n in sums),
              "n_epochs": len(truth),
              "subject_macro_f1": {sid: float(f1_score(a, b, labels=list(range(5)), average="macro", zero_division=0))
                                   for sid, (a, b) in subject_data.items()}}
    result["mean_subject_macro_f1"] = float(np.mean(list(result["subject_macro_f1"].values())))
    return result, rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def train(model: EvidenceModel, train_records: list[Recording], val_records: list[Recording],
          cfg: TrainConfig, run: Path, bank: dict, provenance: dict, device: torch.device,
          resume: bool = False, interrupt_after_epoch: int | None = None):
    """Train core-block conditional likelihoods; evaluate whole continuous segments.

    Checkpoints are saved at epoch boundaries. Mid-epoch interruption repeats that
    epoch; never pretends a saved best-so-far model means training is complete.
    """
    run.mkdir(parents=True, exist_ok=True)
    identity = {"model": model.cfg.dictionary(), "train": asdict(cfg), "provenance": provenance,
                "bank_digest": tensor_digest({str(i): w for i, w in enumerate(bank["waveforms"])}),
                "bank_metadata_digest": fingerprint({k:v for k,v in bank.items() if k != "waveforms"})}
    signature = fingerprint(identity)
    last, best, done = run / "last.pt", run / "best.pt", run / "COMPLETE.json"
    seed_all(cfg.seed, cfg.deterministic)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_score, best_epoch, stale, start, history = -float("inf"), 0, 0, 1, []
    if any(p.exists() for p in (last, best, done)) and not resume:
        raise FileExistsError(f"{run}: output exists; use --resume or a new directory")
    if resume and done.exists():
        finished = json.loads(done.read_text())
        if finished["signature"] != signature:
            raise ValueError("completed run configuration/data signature mismatch")
        payload = load_trusted(best)
        if payload["signature"] != signature or tensor_digest(payload["state_dict"]) != finished["best_state_digest"]:
            raise ValueError("completed checkpoint does not match completion marker")
        model.load_state_dict(payload["state_dict"])
        metrics, rows = evaluate(model, val_records, device, cfg.encode_batch)
        if abs(metrics["mean_subject_macro_f1"] - finished["metrics"]["mean_subject_macro_f1"]) > 1e-5:
            raise RuntimeError("checkpoint re-evaluation changed primary validation metric")
        print(f"VERIFIED COMPLETE: {run}", flush=True)
        return model, metrics, rows
    if resume and last.exists():
        state = load_trusted(last)
        if state["signature"] != signature:
            raise ValueError("resume refused: data, anchors, configuration or provenance changed")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        start, history = state["epoch"] + 1, state["history"]
        restore_rng(state["rng"])
    elif resume and best.exists():
        raise RuntimeError("best.pt without last.pt is not a resumable completed experiment")
    atomic_json({**identity, "signature": signature}, run / "provenance.json")
    items = blocks(train_records, cfg.core_epochs, model.cfg.radius)
    if not items:
        raise ValueError("no training blocks")
    for epoch in range(start, cfg.epochs+1):
        if stale >= cfg.patience:
            break
        model.train()
        total, count = 0.0, 0
        order = torch.randperm(len(items)).tolist()
        t0 = time.perf_counter()
        for number in order:
            ri, a, b, lo, hi = items[number]
            x = torch.from_numpy(train_records[ri].x[a:b]).to(device)
            y = torch.from_numpy(train_records[ri].y[lo:hi]).to(device)[None]
            optimizer.zero_grad(set_to_none=True)
            # Each chunk is independently encoded; no cross-epoch normalization.
            features = torch.cat([model.encode(x[i:i+cfg.encode_batch])
                                  for i in range(0, len(x), cfg.encode_batch)])
            emissions = model.emissions(features[None])[:, lo-a:hi-a]
            valid = torch.ones(y.shape, dtype=torch.bool, device=device)
            loss = model.loss(emissions, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * (hi-lo); count += hi-lo
        metrics, rows = evaluate(model, val_records, device, cfg.encode_batch)
        score = metrics["mean_subject_macro_f1"]
        improved = score > best_score + 1e-8
        stale = 0 if improved else stale+1
        if improved:
            best_score, best_epoch = score, epoch
            atomic_torch_save({"state_dict": model.state_dict(), "model_config": model.cfg.dictionary(),
                               "bank": bank, "signature": signature, "best_epoch": epoch,
                               "metrics": metrics}, best)
        elapsed = time.perf_counter()-t0
        history.append({"epoch": epoch, "train_nll": total/count, "val_nll": metrics["nll"],
                        "val_subject_macro_f1": score, "val_pooled_macro_f1": metrics["macro_f1"],
                        "val_accuracy": metrics["accuracy"], "best_epoch": best_epoch,
                        "stale": stale, "seconds": elapsed, "optimizer_steps": len(items),
                        "last_grad_norm": float(norm)})
        write_csv(history, run / "history.csv")
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "signature": signature, "epoch": epoch, "best_score": best_score,
                           "best_epoch": best_epoch, "stale": stale, "history": history,
                           "rng": rng_state()}, last)
        print(f"e{epoch:03d} nll={total/count:.5f} valMF1={score:.5f} "
              f"valAcc={metrics['accuracy']:.5f} best={best_epoch} steps={len(items)} {elapsed:.1f}s", flush=True)
        if interrupt_after_epoch == epoch:
            raise InterruptedError("test-only simulated epoch-boundary interruption")
    payload = load_trusted(best)
    if payload["signature"] != signature:
        raise ValueError("best checkpoint signature mismatch")
    model.load_state_dict(payload["state_dict"])
    metrics, rows = evaluate(model, val_records, device, cfg.encode_batch)
    write_csv(rows, run / "validation_predictions.csv")
    atomic_json({"signature": signature, "best_epoch": best_epoch, "metrics": metrics,
                 "best_state_digest": tensor_digest(payload["state_dict"]),
                 "test_data_accessed": False, "development_only": True}, done)
    print(f"COMPLETE best={best_epoch} validation subject-MF1={metrics['mean_subject_macro_f1']:.5f}", flush=True)
    return model, metrics, rows

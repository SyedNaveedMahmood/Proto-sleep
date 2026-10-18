"""Digest-sealed wrapper of the unchanged trainer; synthetic or authorized inputs.

No source loader is called here. The live runner must authorize/preflight first.
All checkpoint digests and identities are checked before trusted deserialization.
"""
from dataclasses import asdict
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from mist_evidence.runtime import train, atomic_json
from mist_transfer.experiment import evaluate_source
from .provenance import digest, environment, implementation_snapshot, file_sha


def configure_precision(device):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.device(device).type == "cuda":
        import os
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG before initializing CUDA")


def validate_views(train_records, val_records):
    # Role rejection precedes every file read/hash, including implementation files.
    if (not train_records or not val_records or any(getattr(r, "split", None) != "train" for r in train_records)
            or any(getattr(r, "split", None) != "val" for r in val_records)):
        raise ValueError("only train/val views are permitted before any file access")
    if {r.subject for r in train_records} & {r.subject for r in val_records}:
        raise ValueError("participant leakage across source roles")
    identities = set()
    for r in train_records + val_records:
        if r.recording_id in identities:
            raise ValueError("recording appears more than once")
        identities.add(r.recording_id)
        if (r.x.shape != (len(r.y), 1, 3000) or r.scoring_mask.dtype != np.bool_
                or r.indices.shape != r.y.shape or np.any(np.diff(r.indices) <= 0)
                or np.any(r.indices < 0) or not np.array_equal(r.scoring_mask, r.y >= 0)
                or not np.isin(r.y, [-1, 0, 1, 2, 3, 4]).all()):
            raise ValueError("physical-grid view, labels or scoring masks invalid")


def view_identity(r):
    return {"recording_id": r.recording_id, "subject": r.subject, "role": r.split,
            "source_sha256": r.file_hash,
            "indices_sha256": digest(r.indices.tolist()),
            "labels_sha256": digest(r.y.tolist()), "scoring_mask_sha256": digest(r.scoring_mask.tolist())}


def verify_seal(run, identity):
    run = Path(run)
    marker = run / "COMPLETE_SEAL.json" if (run / "COMPLETE_SEAL.json").exists() else run / "EPOCH_SEAL.json"
    if not marker.exists():
        raise ValueError("resume requires a sealed checkpoint boundary")
    saved = json.loads(marker.read_text())
    if saved["identity"] != identity:
        raise ValueError("resume identity changed: config/code/source/environment/model")
    required = {"last.pt", "best.pt", "history.csv", "provenance.json"}
    if marker.name == "COMPLETE_SEAL.json":
        required |= {"COMPLETE.json", "validation_predictions.csv", "EPOCH_SEAL.json"}
    if set(saved["outputs"]) != required:
        raise ValueError("checkpoint seal has incomplete or unexpected outputs")
    for name, expected in saved["outputs"].items():
        if Path(name).name != name or file_sha(run / name) != expected:
            raise ValueError("checkpoint digest mismatch before deserialization: " + name)
    return saved


def _sealed_train(model, train_records, val_records, cfg, run, *, provenance, device="cpu", bank=None,
                 resume=False, interrupt_after_epoch=None, pause_after_epoch=None):
    validate_views(train_records, val_records)
    if model.cfg.radius != 0:
        raise ValueError("Phase II source adapter requires epoch-only radius zero")
    configure_precision(device)
    bank = {"waveforms": [], "type": "no_bank"} if bank is None else bank
    # The unchanged trainer independently binds bank tensors and metadata too.
    from mist_evidence.runtime import tensor_digest
    identity = digest({"version": "mist-cove-sealed-training-v1", "model": model.cfg.dictionary(),
                       "train": asdict(cfg), "sources": [view_identity(r) for r in train_records + val_records],
                       "provenance": provenance, "code": implementation_snapshot(), "environment": environment(device),
                       "bank_waveform_digest": tensor_digest({str(i): v for i, v in enumerate(bank["waveforms"])}),
                       "bank_metadata_digest": digest({k: v for k, v in bank.items() if k != "waveforms"})})
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        if not resume:
            raise FileExistsError("training output exists; never overwrite")
        verify_seal(run, identity)
    elif resume:
        raise ValueError("no sealed run to resume")
    initial_code = implementation_snapshot()

    def observer(run, signature, row, metrics, checkpoint_seconds):
        if implementation_snapshot() != initial_code:
            raise ValueError("implementation changed during active experiment")
        names = ["last.pt", "best.pt", "history.csv", "provenance.json"]
        atomic_json({"identity": identity, "epoch": row["epoch"], "signature": signature,
                     "resources": row, "checkpoint_write_seconds": checkpoint_seconds,
                     "outputs": {name: file_sha(run / name) for name in names}}, run / "EPOCH_SEAL.json")
        if interrupt_after_epoch == row["epoch"]:
            raise InterruptedError("synthetic test interruption after sealed epoch")

    result = train(model, train_records, val_records, cfg, run, bank, {**provenance, "sealed_identity": identity},
                   torch.device(device), resume=resume, evaluate_fn=evaluate_source,
                   pause_after_epoch=pause_after_epoch, epoch_callback=observer, monitor=True)
    names = ["last.pt", "best.pt", "history.csv", "provenance.json", "COMPLETE.json", "validation_predictions.csv", "EPOCH_SEAL.json"]
    atomic_json({"identity": identity, "status": "COMPLETE", "outputs": {name: file_sha(run / name) for name in names}},
                run / "COMPLETE_SEAL.json")
    return result


def sealed_train(model, train_records, val_records, cfg, run, **kwargs):
    """One writer per output, without touching files before rejecting bad roles."""
    validate_views(train_records, val_records)
    run = Path(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent / (run.name + ".writer.lock")).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("training output already has an active writer") from error
        return _sealed_train(model, train_records, val_records, cfg, run, **kwargs)

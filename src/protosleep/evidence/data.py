"""Strict NPZ contracts, explicit subject splits, and train-only exemplar selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

STAGES = ("Wake", "N1", "N2", "N3", "REM")
EDF_ORDER = [14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2, 13]


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON replacement. An interrupted write never marks a run complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def make_edf_manifest(root: Path, fold: int, dest: Path) -> dict:
    """Discover filenames only. No NPZ arrays, including held-out arrays, are opened."""
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite split manifest: {dest}")
    root = root.expanduser().resolve()
    paths = sorted(root.glob("*.npz"))
    if not 0 <= fold < 20:
        raise ValueError("EDF-20 fold must be in 0..19")
    records = []
    for p in paths:
        if not p.name.startswith("SC") or not p.name[3:5].isdigit():
            raise ValueError(f"EDF filename does not match verified SC / [3:5] rule: {p.name}")
        sid = int(p.name[3:5])
        role = "test" if sid == EDF_ORDER[fold] else "val" if sid == EDF_ORDER[(fold + 1) % 20] else "train"
        records.append({"path": p.name, "subject": str(sid), "split": role})
    if {int(r["subject"]) for r in records} != set(range(20)):
        raise ValueError("Expected exactly Sleep-EDF-20 subjects 0..19")
    doc = {"schema": 1, "dataset": "Sleep-EDF-20", "data_root": str(root),
           "development_only": True, "fold": fold, "fs": 100, "epoch_samples": 3000,
           "label_mapping": list(STAGES), "records": records,
           "note": "All EDF-20 subjects participated in prior development. Not a pristine test cohort."}
    write_json(dest, doc)
    return doc


def validate_manifest(path: Path) -> tuple[dict, Path]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != 1 or doc.get("label_mapping") != list(STAGES):
        raise ValueError("Expected schema=1 and labels [Wake,N1,N2,N3,REM] in that order")
    if doc.get("fs") != 100 or doc.get("epoch_samples") != 3000:
        raise ValueError("v1 is deliberately locked to 100 Hz, 30 s, single-channel EEG; no silent resampling")
    root = Path(doc["data_root"]).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    owners: dict[str, str] = {}
    seen: set[str] = set()
    for r in doc["records"]:
        if r["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split role")
        sid = str(r["subject"])
        if sid in owners and owners[sid] != r["split"]:
            raise ValueError(f"Subject crosses splits: {sid}")
        owners[sid] = r["split"]
        p = Path(r["path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Record paths must be relative within data_root")
        resolved = str((root / p).resolve())
        if resolved in seen:
            raise ValueError("Duplicate or aliased recording in manifest")
        seen.add(resolved)
    if not {"train", "val", "test"} <= set(owners.values()):
        raise ValueError("Manifest must declare disjoint train, val and locked test subjects")
    return doc, root


@dataclass
class Night:
    key: str
    subject: str
    split: str
    x: np.ndarray
    y: np.ndarray
    epoch_index: np.ndarray
    continuity_verified: bool
    sha256: str


def load_split(manifest: Path, role: str) -> list[Night]:
    """Open only the requested train/val role. Test access is intentionally unavailable."""
    if role not in {"train", "val"}:
        raise ValueError("This development runner cannot open test records")
    doc, root = validate_manifest(manifest)
    nights = []
    for r in doc["records"]:
        if r["split"] != role:
            continue
        path = root / r["path"]
        with np.load(path, allow_pickle=False) as d:
            if not {"x", "y"} <= set(d.files):
                raise ValueError(f"{path.name}: missing x/y")
            x, raw_y = d["x"], np.asarray(d["y"]).reshape(-1)
            if x.ndim == 2:
                x = x[:, None, :]
            elif x.ndim == 3 and x.shape[1:] == (3000, 1):
                x = x.transpose(0, 2, 1)
            if x.ndim != 3 or x.shape[1:] != (1, 3000):
                raise ValueError(f"{path.name}: expected [N,1,3000], got {x.shape}")
            if not np.isfinite(x).all() or not np.isfinite(raw_y).all():
                raise ValueError(f"{path.name}: nonfinite values")
            if not np.equal(raw_y, np.floor(raw_y)).all():
                raise ValueError(f"{path.name}: labels must be integers, not silently rounded")
            y = raw_y.astype(np.int64)
            if len(x) != len(y) or len(y) == 0 or not set(y.tolist()) <= {-1, 0, 1, 2, 3, 4}:
                raise ValueError(f"{path.name}: length/label contract failed")
            if "fs" in d.files and (d["fs"].size != 1 or float(d["fs"].reshape(-1)[0]) != 100.0):
                raise ValueError(f"{path.name}: wrong sampling frequency")
            has_index = "epoch_index" in d.files
            ix = np.asarray(d["epoch_index"]).reshape(-1) if has_index else np.arange(len(y))
            if len(ix) != len(y) or not np.isfinite(ix).all() or not np.equal(ix, np.floor(ix)).all():
                raise ValueError("Invalid epoch_index")
            ix = ix.astype(np.int64)
            if len(ix) > 1 and not (np.diff(ix) > 0).all():
                raise ValueError("epoch_index must increase strictly")
        nights.append(Night(r["path"], str(r["subject"]), role, np.asarray(x, dtype=np.float32),
                            y, ix, has_index or bool(r.get("epochs_contiguous", False)), digest_file(path)))
    if not nights:
        raise ValueError(f"Empty {role} split")
    return nights


class Epochs(Dataset):
    def __init__(self, nights: list[Night]):
        self.nights = nights
        self.items = [(i, int(j)) for i, n in enumerate(nights) for j in np.flatnonzero(n.y >= 0)]
        if not self.items:
            raise ValueError("No labeled epochs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n, j = self.items[i]
        return torch.from_numpy(self.nights[n].x[j]), int(self.nights[n].y[j])


def contiguous_runs(night: Night, assume_contiguous: bool = False) -> list[np.ndarray]:
    """Unknown labels and missing epochs break chains. Never join different nights."""
    if not night.continuity_verified and not assume_contiguous:
        raise ValueError(f"{night.key}: continuity not verified. Provide original epoch_index or explicitly acknowledge --assume-contiguous.")
    valid = np.flatnonzero(night.y >= 0)
    if not len(valid):
        return []
    cuts = np.flatnonzero((np.diff(valid) != 1) | (np.diff(night.epoch_index[valid]) != 1)) + 1
    return list(np.split(valid, cuts))


def candidates(nights: list[Night], length: int, per_subject: int, seed: int):
    """Subject-balanced random crop pool, train only. Stage labels are epoch labels, not event labels."""
    if any(n.split != "train" for n in nights):
        raise ValueError("Anchor construction must use training subjects only")
    rng = np.random.default_rng(seed + length)
    xs, ys, refs = [], [], []
    for sid in sorted({n.subject for n in nights}):
        items = [(n, int(j)) for n in nights if n.subject == sid for j in np.flatnonzero(n.y >= 0)]
        if not items:
            continue
        for k in rng.choice(len(items), size=min(len(items), per_subject), replace=False):
            n, j = items[int(k)]
            start = int(rng.integers(0, 3000 - length + 1))
            xs.append(n.x[j, :, start:start + length].copy())
            ys.append(int(n.y[j]))
            refs.append({"recording": n.key, "subject": sid, "epoch_row": j,
                         "epoch_index": int(n.epoch_index[j]), "start_sample": start,
                         "end_sample": start + length, "epoch_stage": int(n.y[j]),
                         "source_sha256": n.sha256, "clinical_event": "unreviewed"})
    return torch.from_numpy(np.stack(xs)), np.asarray(ys), refs

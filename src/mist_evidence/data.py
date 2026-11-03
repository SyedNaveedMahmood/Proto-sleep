"""Explicit split manifests, selective NPZ access, and real waveform anchors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .model import ModelConfig, observed_parts

ORDER = (14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2, 13)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def edf_manifest(root: Path, fold: int) -> dict:
    """Filename discovery only. Does not open NPZs, including reserved test files."""
    if not 0 <= fold < 20:
        raise ValueError("EDF-20 fold must be in 0..19")
    entries = []
    for path in sorted(root.expanduser().resolve().glob("*.npz")):
        name = path.name
        if len(name) < 5 or not name[3:5].isdigit():
            raise ValueError(f"cannot parse subject from filename[3:5]: {name}")
        sid = int(name[3:5])
        split = "test" if sid == ORDER[fold] else "val" if sid == ORDER[(fold+1) % 20] else "train"
        entries.append({"path": str(path), "subject": str(sid), "split": split})
    if {int(e["subject"]) for e in entries} != set(range(20)):
        raise ValueError("EDF convenience mode requires exactly subjects 0..19")
    return {"dataset": "Sleep-EDF-20", "purpose": "development_only", "fold": fold, "records": entries}


def read_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    for entry in value.get("records", []):
        p = Path(entry["path"]).expanduser()
        entry["path"] = str((path.parent / p if not p.is_absolute() else p).resolve())
        entry["subject"] = str(entry["subject"])
    return value


def validate_manifest(manifest: dict) -> None:
    entries = manifest.get("records", [])
    if not entries or not manifest.get("dataset"):
        raise ValueError("manifest requires dataset and records")
    subjects, paths = {}, set()
    for e in entries:
        if e.get("split") not in {"train", "val", "test"} or not str(e.get("subject", "")):
            raise ValueError("every record needs a subject and train/val/test split")
        path = str(Path(e["path"]).expanduser().resolve())
        if path in paths:
            raise ValueError("duplicate recording path in manifest")
        paths.add(path)
        sid = str(e["subject"])
        if sid in subjects and subjects[sid] != e["split"]:
            raise ValueError(f"subject {sid} crosses splits")
        subjects[sid] = e["split"]
    if not {"train", "val", "test"}.issubset(set(subjects.values())):
        raise ValueError("explicit nonempty train, validation, and reserved test subjects required")


@dataclass
class Recording:
    path: str
    subject: str
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    file_hash: str
    continuity_assumed: bool
    indices_known: bool = True
    independent_epochs: bool = False

    def segments(self):
        if self.independent_epochs:
            return [(i, i+1) for i in range(len(self.y))]
        cuts = np.flatnonzero(np.diff(self.indices) != 1) + 1
        bounds = np.r_[0, cuts, len(self.y)]
        return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def load_split(manifest: dict, split: str, assume_contiguous: bool = False, independent_epochs: bool = False) -> list[Recording]:
    """This development runner deliberately has no test-opening code path."""
    validate_manifest(manifest)
    if split not in {"train", "val"}:
        raise ValueError("test access is not implemented; use a separately frozen evaluation protocol")
    result = []
    for e in manifest["records"]:
        if e["split"] != split:
            continue
        path = Path(e["path"])
        with np.load(path, allow_pickle=False) as z:
            if "x" not in z.files or "y" not in z.files:
                raise ValueError(f"{path.name}: x/y arrays required")
            x, labels = np.asarray(z["x"]), np.asarray(z["y"]).reshape(-1)
            if x.ndim == 2:
                x = x[:, None]
            elif x.ndim == 3 and x.shape[1:] == (3000, 1):
                x = x.transpose(0, 2, 1)
            if x.ndim != 3 or x.shape[1:] != (1, 3000) or len(x) != len(labels) or not len(x):
                raise ValueError(f"{path.name}: expected x [N,1,3000] and y [N]")
            if not np.isfinite(x).all() or not np.isfinite(labels).all():
                raise ValueError(f"{path.name}: non-finite signal or labels")
            if not np.equal(labels, labels.astype(np.int64)).all() or not np.isin(labels, range(5)).all():
                raise ValueError(f"{path.name}: labels must already map to Wake/N1/N2/N3/REM = 0..4")
            fs = np.asarray(z["fs"]).reshape(-1) if "fs" in z.files else np.array([100.0])
            if fs.size != 1 or abs(float(fs[0]) - 100.0) > 1e-8:
                raise ValueError(f"{path.name}: 100 Hz required; no silent resampling")
            present = [k for k in ("epoch_indices", "epoch_index") if k in z.files]
            if len(present) > 1:
                raise ValueError(f"{path.name}: ambiguous epoch index arrays")
            assumed = not present and assume_contiguous and not independent_epochs
            if not present and not assume_contiguous and not independent_epochs:
                raise ValueError(f"{path.name}: no epoch_indices. Verify temporal continuity before using "
                                 "--assume-contiguous; removed epochs must retain their original indices. "
                                 "Use --epoch-only for safe non-temporal development.")
            idx = np.asarray(z[present[0]]).reshape(-1) if present else np.arange(len(x))
            if len(idx) != len(x) or not np.isfinite(idx).all() or not np.equal(idx, idx.astype(np.int64)).all():
                raise ValueError(f"{path.name}: invalid epoch indices")
            if np.any(np.diff(idx) <= 0) or np.any(idx < 0):
                raise ValueError(f"{path.name}: epoch indices must be strictly increasing and nonnegative")
        result.append(Recording(str(path), str(e["subject"]), x.astype(np.float32), labels.astype(np.int64),
                                idx.astype(np.int64), sha256_file(path), assumed, bool(present), independent_epochs))
    if not result:
        raise ValueError(f"empty {split} split")
    # Reject exact duplicate recording files even under different names/subjects.
    if len({r.file_hash for r in result}) != len(result):
        raise ValueError(f"duplicate recording bytes within {split}")
    return result


def blocks(recordings: list[Recording], core: int, radius: int):
    """Nonoverlapping scored cores with context halos. Never cross nights or gaps."""
    if core < 1 or radius < 0:
        raise ValueError("invalid block size")
    output = []
    for ri, r in enumerate(recordings):
        for a, b in r.segments():
            for lo in range(a, b, core):
                hi = min(lo + core, b)
                output.append((ri, max(a, lo-radius), min(b, hi+radius), lo, hi))
    return output


def build_anchors(recordings: list[Recording], cfg: ModelConfig, seed: int = 1337,
                  per_subject: int = 64, method: str = "medoid") -> dict:
    """Subject/stage-balanced candidates; unsupervised clustering; actual medoids.

    Stage labels are used ONLY to balance candidate sampling from training data.
    The anchor's stage label is not a clinical label for the waveform structure.
    """
    if method not in {"medoid", "random"} or per_subject < 5:
        raise ValueError("invalid bank construction settings")
    log_rms = np.concatenate([np.log(np.std(r.x, axis=(1,2)).clip(min=1e-6)) for r in recordings])
    amplitude_stats = (float(log_rms.mean()), max(0.1, float(log_rms.std())))
    rng = np.random.default_rng(seed)
    subjects = sorted({r.subject for r in recordings})
    waves, metadata = [], []
    for length in cfg.scales:
        candidates, identities, seen = [], [], set()
        for sid in subjects:
            rids = [i for i, r in enumerate(recordings) if r.subject == sid]
            by_stage = {s: [(i, int(j)) for i in rids for j in np.flatnonzero(recordings[i].y == s)]
                        for s in range(5)}
            available = [s for s in range(5) if by_stage[s]]
            for trial in range(per_subject * 4):
                s = available[trial % len(available)]
                pool = by_stage[s]
                ri, ep = pool[int(rng.integers(len(pool)))]
                start = int(rng.integers((cfg.samples-length)//cfg.stride + 1)) * cfg.stride
                key = (ri, ep, start)
                if key in seen:
                    continue
                patch = recordings[ri].x[ep, 0, start:start+length].copy()
                if float(patch.std()) <= 1e-6:
                    continue
                seen.add(key)
                candidates.append(patch)
                r = recordings[ri]
                identities.append({"source_path": r.path, "source_sha256": r.file_hash,
                                   "subject": r.subject, "array_epoch": ep,
                                   "original_epoch": int(r.indices[ep]) if r.indices_known else None, "start_sample": start,
                                   "length_samples": length, "stage_at_source_epoch": int(r.y[ep]),
                                   "clinical_event_label": None})
                if sum(m["subject"] == sid for m in identities) >= per_subject:
                    break
        k = cfg.prototypes_per_scale
        if len(candidates) < k:
            raise ValueError(f"only {len(candidates)} nonflat candidate windows for {k} anchors")
        bank = torch.tensor(np.stack(candidates))
        with torch.no_grad():
            shape, power, envelope, _ = observed_parts(bank)
            pooled = torch.nn.functional.adaptive_avg_pool1d(shape[:, None], 32)[:, 0]
            desc = torch.cat((torch.nn.functional.normalize(pooled, dim=-1), power, envelope), -1).numpy()
        if method == "random":
            chosen = rng.choice(len(bank), k, replace=False).tolist()
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                                 batch_size=min(256, len(bank)), reassignment_ratio=0.0).fit(desc)
            chosen = []
            for center in km.cluster_centers_:
                order = np.argsort(np.square(desc-center).sum(-1), kind="stable")
                chosen.append(next(int(i) for i in order if int(i) not in chosen))
        waves.append(bank[chosen])
        metadata.append([identities[i] for i in chosen])
    return {"waveforms": waves, "metadata": metadata, "seed": seed, "method": method,
            "training_subjects": subjects, "candidate_labels_used": True,
            "config": cfg.dictionary(), "amplitude_stats": amplitude_stats,
            "amplitude_statistics_fit": "training only; input file units, not inferred microvolts"}

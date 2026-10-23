"""Authenticated SC TRAIN-only full-epoch cache for the Phase M0 experiment."""
from __future__ import annotations

import json
import hashlib
import numpy as np

from mist_cove.phase11b.historical import ROOT
from mist_cove.phase11b.security import IntegrityError
from mist_cove.phase13.provenance import setup_source
from mist_cove.phase13.sampling import frozen_selection
from mist_transfer.preprocessing import epoch_targets

PRIVATE = ROOT / "mist_transfer_runs" / "mist_sleep" / "method_recovery_v1"
PUBLIC = ROOT / "reports" / "mist_sleep" / "method_recovery_v1"


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_full_train() -> dict:
    """Decode only authenticated original TRAIN recordings, preserving every epoch.

    The Phase XIII allocation is authenticated before the waveform access. Existing
    fit-only normalization is reused exactly, and private outputs stay ignored.
    """
    selection = frozen_selection()
    allocation = selection["allocation"]
    normalization = selection["normalization"]
    if normalization["fit_role"] != "train_fit" or normalization["fit_participants"] != 10 or normalization["std_uV"] <= 0:
        raise IntegrityError("fit-only normalization receipt differs")
    fit_subjects = set(allocation["fit"])
    dev_subjects = set(allocation["internal_dev"])
    if len(fit_subjects) != 10 or len(dev_subjects) != 5 or fit_subjects & dev_subjects:
        raise IntegrityError("frozen Phase XIII participant partition differs")
    _, access = setup_source()
    records = access.records("train")
    if len(records) != 29 or {r["subject_id"] for r in records} != fit_subjects | dev_subjects:
        raise IntegrityError("not the 29 original TRAIN recordings")
    count = sum(r["epoch_grid"]["n_epochs"] for r in records)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if any((PRIVATE / name).exists() for name in ("full_train_wave.npy", "full_train_index.npz", "full_train_receipt.json")):
        raise FileExistsError("full TRAIN cache already exists; do not overwrite")
    wave = np.lib.format.open_memmap(PRIVATE / "full_train_wave.npy", mode="w+", dtype="float32", shape=(count, 1, 3000))
    stage = np.full(count, -1, np.int8)
    role = np.empty(count, np.int8)
    person = np.empty(count, np.int16)
    recording = np.empty(count, np.int16)
    epoch = np.empty(count, np.int32)
    all_subjects = sorted(fit_subjects | dev_subjects)
    offset = 0
    for ri, record in enumerate(records):
        n = record["epoch_grid"]["n_epochs"]
        subject = record["subject_id"]
        if subject not in fit_subjects | dev_subjects:
            raise IntegrityError("unexpected participant")
        target, valid, *_ = epoch_targets(record)
        if len(target) != n or len(valid) != n:
            raise IntegrityError("annotation/epoch geometry mismatch")
        stage[offset:offset+n] = np.where(valid, target, -1).astype(np.int8)
        role[offset:offset+n] = 0 if subject in fit_subjects else 1
        person[offset:offset+n] = all_subjects.index(subject)
        recording[offset:offset+n] = ri
        epoch[offset:offset+n] = np.arange(n)
        with access.reader(record) as reader:
            for start in range(0, n, 256):
                indices = np.arange(start, min(start+256, n))
                _, raw = access.read(record, reader, indices)
                wave[offset+start:offset+start+len(indices)] = (raw - normalization["mean_uV"]) / normalization["std_uV"]
        offset += n
    wave.flush()
    del wave
    np.savez(PRIVATE / "full_train_index.npz", stage=stage, role=role, person=person,
             recording=recording, epoch=epoch)
    receipt = {"status": "PASS_TRAIN_ONLY", "physical_epochs": count,
               "scored_fit": int(((stage >= 0) & (role == 0)).sum()),
               "scored_development": int(((stage >= 0) & (role == 1)).sum()),
               "fit_participants": 10, "development_participants": 5,
               "recordings": len(records), "normalization_fit_role": "train_fit",
               "raw_VAL_reads": 0, "excluded_source_reads": 0,
               "waveform_read_count": access.counts["waveform_file_reads"],
               "private_cache": True,
               "wave_cache_sha256": _sha256_file(PRIVATE / "full_train_wave.npy"),
               "index_cache_sha256": _sha256_file(PRIVATE / "full_train_index.npz")}
    (PRIVATE / "full_train_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "source_scope.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_full_train() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    receipt = json.loads((PRIVATE / "full_train_receipt.json").read_text())
    if receipt["status"] != "PASS_TRAIN_ONLY" or receipt["raw_VAL_reads"] or receipt["excluded_source_reads"]:
        raise IntegrityError("full TRAIN cache role receipt differs")
    if _sha256_file(PRIVATE / "full_train_wave.npy") != receipt.get("wave_cache_sha256") or \
       _sha256_file(PRIVATE / "full_train_index.npz") != receipt.get("index_cache_sha256"):
        raise IntegrityError("private TRAIN cache bytes differ")
    wave = np.load(PRIVATE / "full_train_wave.npy", mmap_mode="r")
    with np.load(PRIVATE / "full_train_index.npz", allow_pickle=False) as archive:
        index = {k: archive[k] for k in archive.files}
    if len(wave) != receipt["physical_epochs"] or any(len(v) != len(wave) for v in index.values()):
        raise IntegrityError("cache length differs")
    if np.any((index["role"] < 0) | (index["role"] > 1)):
        raise IntegrityError("cache role differs")
    return wave, index

"""Outcome-independent scientific identities and execution gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_COMMIT = "1d6950021d43a0aa1e1c719f98dfed1b61778d10"
CHECKPOINTS = {
    "M0_123": "bc2cb6d293868d8029dde631adb2c0d4962e4166c6c060e59e61a328b21fdadf",
    "M0_456": "15dafb7c1d742fd81b014e04fc3b74f29b9819150059539d0bd879403df515f7",
    "M2_123": "c07d03a421cde428ec81a923baad459dded6c098bab064c41ee6cad8b95ae2f2",
    "M2_456": "883a743b71f8cc22d688fa5c91fc1da75f0810ae7bfb99f88a1e0b312d78f9fe",
}
STAGES = ("Wake", "N1", "N2", "N3", "REM")
SOURCE_MEAN_UV = 0.2903136955214651
SOURCE_STD_UV = 26.31764815403481
SOURCE_PROTOCOL_SHA256 = "ff032a8f1cfa76bf4adb8ae23d3436d1e3a0271e4355481695467c69b5bcb6b3"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def specification():
    return {
        "version": "mist-sleep-phase-m1-hmc-v1", "base_commit": BASE_COMMIT,
        "dataset": "hmc-sleep-staging", "dataset_version": "1.1", "doi": "10.13026/t79q-fr32",
        "checkpoints": CHECKPOINTS, "step": 1200, "arms": ["M0", "M2"], "seeds": [123, 456],
        "primary_channel": "C4/M1", "secondary_channel": "F4/M1", "stage_order": list(STAGES),
        "channel_label_aliases": {"C4/M1": ["C4/M1", "EEG C4/M1", "C4-M1", "EEG C4-M1"],
                                  "F4/M1": ["F4/M1", "EEG F4/M1", "F4-M1", "EEG F4-M1"]},
        "source_normalization": {"mean_uV": SOURCE_MEAN_UV, "std_uV": SOURCE_STD_UV,
                                 "fit_role": "train_fit", "fit_participants": 10},
        "signal": {"source_Hz": 256, "target_Hz": 100, "epoch_seconds": 30,
                   "epoch_samples": 3000, "up": 25, "down": 64,
                   "filter": "firwin(1281, 1/64, window=('kaiser',5.0)); float64",
                   "padding": "constant zero only at physical recording ends",
                   "alignment": "continuous full-record polyphase; t=0 at first source sample; floor(duration/30) complete epochs",
                   "units": {"uV": 1.0, "mV": 1000.0, "V": 1000000.0},
                   "polarity": "unchanged; no reference reconstruction", "additional_filter": None},
        "eligibility": ["official v1.1 waveform and both matching scoring files with SHA256",
                        "unique exact primary channel or EEG-prefixed equivalent; 256 Hz; supported units",
                        "finite calibrated waveform and at least one complete 30-second epoch",
                        "continuous EDF; no arbitrary repairs; scoring timestamp agreement after seal"],
        "label_mapping": {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
                          "Sleep stage N3": 3, "Sleep stage R": 4, "Sleep stage ?": -1,
                          "Movement time": -1},
        "annotations": "EDF+ exact stage text; whole 30-second intervals only; reject unknown stages, overlaps, contradictions, fractional shifts; gaps masked; require EDF+/text agreement",
        "text_schema_status": "unverified on real HMC; generated CSV onset_seconds,duration_seconds,annotation supported; other schemas block, never guess",
        "statistical_unit": "recording; participant linkage unverified; never claim 151 independent people",
        "grouping_policy": "if independently verified linkage is available before scoring: equal recording endpoint, participant-cluster bootstrap; otherwise recordings",
        "primary_metric": "equal-recording mean fixed-five Macro-F1; absent classes zero",
        "primary_contrast": "mean over seeds of paired M2-M0 recording means",
        "bootstrap": {"draws": 10000, "seed": 20261023, "interval": "percentile 2.5/97.5",
                      "resample": "complete paired recording clusters; both seeds held together"},
        "success": {"minimum_gain": 0.02, "both_seeds_positive": True, "lower_CI_above_zero": True,
                    "authenticated_sources": True, "full_input_reproduced": True,
                    "preprocessing_correct": True, "sealed_before_labels": True, "no_adaptation": True},
        "collapse_flag": "any arm/seed predicts >=95% one stage pooled on scored primary epochs; report alongside original metric",
        "secondary": ["F4/M1 same checkpoints and conversion; never replaces primary",
                      "stage F1/recall, confusion, balanced accuracy, pooled F1, stage-present/absent counts",
                      "NLL, Brier, ECE 10 bins including confidence=1, wrong confidence>=.9, entropy and margins",
                      "seed direction/SD, paired recording extremes and leave-one-record-out effect",
                      "embedding moments, prototype entropy/effective usage, bounded-view drift/KL/switching; source comparisons only frozen source-fit statistics"],
        "limits": {"download_bytes": 18000000000, "estimated_collection_bytes": 15700000000,
                   "disk_safety_bytes": 2000000000, "CPU_seconds": 14400, "GPU_seconds": 7200,
                   "optimizer_updates": 0, "target_normalizer_fits": 0, "SC_RESERVED_reads": 0},
        "stages": ["git-commit protocol", "blind primary AND secondary predictions", "seal all predictions",
                   "git-commit public seal receipt", "one-time unblind"],
        "source_access": "authenticated existing source-fit metadata only by default; no new VAL, RESERVED, ST or other cohort reads",
        "M0_decision_preserved": "MPI_FEASIBILITY_NOT_ESTABLISHED",
        "interpretation": "combined clinical, acquisition and montage shift; resampling is not montage correction; no clinical deployment or multi-target claim",
    }


def scientific_decision(effect, per_seed, interval, *, integrity_passed, collapse=False):
    if not integrity_passed:
        decision = "PRIMARY_EVALUATION_BLOCKED"
    else:
        if effect is None or interval is None or len(per_seed) != 2:
            raise ValueError("complete primary outcome required")
        import math
        if not all(math.isfinite(v) for v in [effect, *per_seed, *interval]):
            raise ValueError("nonfinite scientific result")
        if abs(effect - sum(per_seed) / 2) > 1e-10 or interval[0] > interval[1]:
            raise ValueError("inconsistent effect or interval")
        decision = ("MPI_HMC_PRIMARY_GAIN_NOT_ESTABLISHED" if effect < .02 else
                    "MPI_HMC_CROSS_MONTAGE_TRANSFER_POSITIVE" if min(per_seed) > 0 and interval[0] > 0 else
                    "MPI_HMC_RESULT_INCONCLUSIVE")
    return {"decision": decision, "flag": "SEVERE_CROSS_MONTAGE_FAILURE" if collapse else None}

"""Phase-separated entrypoints. Real execution requires every frozen gate.

The present Phase M1 run stops at disk preflight. These entrypoints are retained
for review and generated validation; they do not run automatically on import.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import numpy as np

from .protocol import CHECKPOINTS, digest, file_sha256, specification, scientific_decision
from .hmc_manifest import private_root, verified_manifest, verify_file
from .hmc_loader import load_waveform
from .source_checkpoint import load_source
from .blind_inference import predict
from .prediction_seal import seal_predictions, begin_scoring, write_once
from .stage_metrics import stage_metrics
from .paired_statistics import paired_effect

PROTOCOL_PATH = "reports/mist_sleep/phase_m1_hmc_v1/frozen_protocol.json"


def committed_identity(repo, relative):
    repo = Path(repo)
    value = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
    if (repo / relative).read_bytes() != value:
        raise ValueError("registered public artifact differs from committed bytes")
    return json.loads(value)


def require_frozen_protocol(repo):
    protocol = committed_identity(repo, PROTOCOL_PATH)
    if protocol != specification():
        raise ValueError("scientific protocol changed")
    # Historical scientific implementation is also byte frozen.
    m0 = committed_identity(repo, "reports/mist_sleep/method_recovery_v1/frozen_protocol.json")
    for name, expected in m0["source_hashes"].items():
        if file_sha256(Path(repo) / "src/mist_sleep" / name) != expected:
            raise ValueError("historical source implementation changed")
    return protocol


def require_inference_readiness(repo):
    """Engineering receipts must be committed and bind the executed implementation."""
    base = "reports/mist_sleep/phase_m1_hmc_v1/"
    verification = committed_identity(repo, base + "verification.json")
    generated = committed_identity(repo, base + "generated_validation.json")
    if not verification["CPU"]["passed"] or len(generated["models"]) != 4:
        raise ValueError("generated full-input validation is missing")
    for name, expected in verification["implementation_sha256"].items():
        if file_sha256(Path(repo) / name) != expected:
            raise ValueError("validated external implementation changed")
    rows = {row["model"]: row for row in generated["models"]}
    if set(rows) != set(CHECKPOINTS) or any(
            rows[key]["source_sha256"] != expected or not rows[key]["generated_full_input_exact_match"]
            or not rows[key]["state_unchanged"] for key, expected in CHECKPOINTS.items()):
        raise ValueError("source behavior reproduction failed")
    # Re-authenticate metadata, never a source waveform, for the scaler binding.
    from mist_cove.phase13.sampling import frozen_selection
    norm = frozen_selection()["normalization"]
    fixed = specification()["source_normalization"]
    if any(norm[key] != fixed[key] for key in fixed):
        raise ValueError("source-fit normalization lineage differs")


def blind(repo, metadata_root, output_root):
    """CPU only; every source triplet verified, no annotation interpretation."""
    import torch
    repo, out = Path(repo), Path(output_root)
    protocol = require_frozen_protocol(repo)
    require_inference_readiness(repo)
    root = private_root()
    out = out.absolute()
    if root == out or root in out.parents:
        raise ValueError("prediction destination must be separate from immutable raw source")
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    if out.resolve() != out or out.stat().st_mode & 0o077:
        raise ValueError("private nonsymlink output root with mode 0700 required")
    manifest = verified_manifest(metadata_root)
    # Hashing scoring bytes is an integrity operation, not semantic label access.
    for row in manifest:
        for name, expected in row["files"].items():
            verify_file(root, name, expected)
    models = {}
    for arm in ("M0", "M2"):
        for seed in (123, 456):
            models[f"{arm}_{seed}"] = load_source(repo / "mist_transfer_runs/mist_sleep/method_recovery_v1/models", arm, seed)[0]
    torch.set_num_threads(2)
    write_once(out / "BLIND_STARTED.json", {"protocol_sha256": digest(protocol), "manifest_sha256": digest(manifest)})
    wall, cpu = time.monotonic(), time.process_time()
    files, ledger, inferences = [], [], 0
    for row in manifest:
        for channel in ("C4/M1", "F4/M1"):
            if time.monotonic() - wall > 14400 or time.process_time() - cpu > 14400:
                raise RuntimeError("scientific CPU execution ceiling reached")
            relative = f"recordings/{row['recording']}.edf"
            try:
                epochs, onsets, header = load_waveform(root, relative, row["files"][relative], channel)
            except ValueError as error:
                # Original error is private; no patient-linked rows are published.
                ledger.append({"recording": row["recording"], "channel": channel, "eligible": False, "reason": str(error)})
                continue
            for key, model in models.items():
                result = predict(model, epochs)
                name = f"{row['recording']}_{channel.replace('/', '-')}_{key}.npz"
                with (out / name).open("xb") as stream:
                    np.savez_compressed(stream, **result, onsets=onsets, predictions=result["logits"].argmax(1))
                files.append(name)
                inferences += len(epochs)
            ledger.append({"recording": row["recording"], "channel": channel, "eligible": True, "header": header})
    if not any(r["eligible"] and r["channel"] == "C4/M1" for r in ledger):
        raise RuntimeError("no eligible primary recordings")
    write_once(out / "eligibility.private.json", ledger)
    write_once(out / "runtime.private.json", {"wall_seconds": time.monotonic() - wall,
               "CPU_seconds": time.process_time() - cpu, "inference_epochs": inferences, "optimizer_updates": 0})
    files.extend(["eligibility.private.json", "runtime.private.json"])
    return seal_predictions(out, files, protocol_sha256=digest(protocol), manifest_sha256=digest(manifest),
                            model_identities=CHECKPOINTS, preprocessing_sha256=digest(protocol["signal"]))


def score(repo, metadata_root, output_root, public_seal_relative):
    """Claim once before opening labels. Any defect preserves the exposed run."""
    from .hmc_scoring import read_scoring
    root, out = private_root(), Path(output_root)
    protocol = require_frozen_protocol(repo)
    require_inference_readiness(repo)
    public_seal = committed_identity(repo, public_seal_relative)
    manifest = verified_manifest(metadata_root)
    if (public_seal.get("protocol_sha256"), public_seal.get("manifest_sha256"),
        public_seal.get("preprocessing_sha256"), public_seal.get("model_identities")) != (
            digest(protocol), digest(manifest), digest(protocol["signal"]), CHECKPOINTS):
        raise ValueError("seal does not match frozen experiment")
    sealed = begin_scoring(out, public_seal)
    ledger = json.loads((out / "eligibility.private.json").read_text())
    by_record = {r["recording"]: r for r in manifest}
    results = {}
    for channel in ("C4/M1", "F4/M1"):
        channel_rows, paired = [], {key: [] for key in CHECKPOINTS}
        pooled = {key: {"y": [], "logits": []} for key in CHECKPOINTS}
        for entry in ledger:
            if entry["channel"] != channel or not entry["eligible"]:
                continue
            identity = entry["recording"]
            y = read_scoring(root, identity, by_record[identity]["files"],
                             epoch_count=entry["header"]["complete_epochs"],
                             waveform_start=entry["header"]["start_datetime"], claim_root=out)
            valid = y >= 0
            if not valid.any():
                continue
            metrics = {}
            for key in CHECKPOINTS:
                name = f"{identity}_{channel.replace('/', '-')}_{key}.npz"
                if name not in sealed["files"]:
                    raise ValueError("model-paired prediction absent from seal")
                with np.load(out / name, allow_pickle=False) as archive:
                    if not np.array_equal(archive["onsets"], np.arange(len(y)) * 30):
                        raise ValueError("prediction timestamp shift")
                    logits = archive["logits"]
                    if logits.shape != (len(y), 5):
                        raise ValueError("prediction/scoring epoch mismatch")
                metrics[key] = stage_metrics(y[valid], logits[valid])
                paired[key].append(metrics[key]["macro_f1"])
                pooled[key]["y"].append(y[valid])
                pooled[key]["logits"].append(logits[valid])
            channel_rows.append({"recording": identity, "models": metrics})
        write_once(out / f"{channel.replace('/', '-')}_scores.private.json", channel_rows)
        if len(channel_rows) < 2:
            raise ValueError("insufficient paired eligible recordings")
        statistics = paired_effect(np.column_stack([paired["M0_123"], paired["M0_456"]]),
                                   np.column_stack([paired["M2_123"], paired["M2_456"]]),
                                   [r["recording"] for r in channel_rows])
        aggregate = {key: stage_metrics(np.concatenate(values["y"]), np.concatenate(values["logits"]))
                     for key, values in pooled.items()}
        results[channel] = {"analysis": "primary" if channel == "C4/M1" else "secondary sensitivity",
                            "statistics": statistics, "pooled": aggregate,
                            "stage_present_recordings": [sum(row["models"]["M0_123"]["stage_present"][k] for row in channel_rows) for k in range(5)]}
    primary = results["C4/M1"]
    primary["decision"] = scientific_decision(primary["statistics"]["effect"], primary["statistics"]["per_seed_effect"],
                                              primary["statistics"]["interval_95"], integrity_passed=True,
                                              collapse=any(v["severe_collapse"] for v in primary["pooled"].values()))
    write_once(out / "aggregate_results.private.json", results)
    return results

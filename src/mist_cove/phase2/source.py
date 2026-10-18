"""Prepared source-only runner. NEVER invoked on real inputs in Phase II.

Explicit authorization and an independently frozen launch receipt are required
before split/normalization/descriptor access. No statistic or bank is fitted.
"""
import json
from pathlib import Path

import torch

from mist_evidence.runtime import TrainConfig, seed_all, atomic_json
from .adapter import build_adapter
from .provenance import ROOT, digest, file_sha, implementation_snapshot
from .training import configure_precision, sealed_train

SPLIT = "3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84"
SPLIT_SHA = "1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5"
VARIANTS = ["q1", "availability", "background", "rounded_q1", "raw_context", "evidence"]
RECIPE = {"seeds": [123, 456, 789], "optimizer": "AdamW", "lr": .0003, "weight_decay": .0001,
          "max_epochs": 60, "patience": 12, "core_epochs": 16, "encode_batch": 16, "grad_clip": 5.,
          "deterministic": True, "amp": False, "tf32": False, "fp32": True, "threads": 2,
          "temporal_radius": 0, "crf": False, "selection_improvement": 1e-8,
          "selection_metric": "mean_subject_macro_f1"}


def validate_source_config(c):
    expected = {**RECIPE, "version": "mist-cove-source-prepared-v1", "status": "PREPARED_NOT_AUTHORIZED_OR_EXECUTED",
                "primary": "q1", "variants": VARIANTS, "split_fingerprint": SPLIT, "split_file_sha256": SPLIT_SHA,
                "input": {"channel": "EEG Fpz-Cz", "units": "uV", "sample_rate_hz": 100, "samples": 3000},
                "normalization": "REUSE_VERIFIED_TRAIN_ONLY_NO_REFIT", "exemplar_fitting": False,
                "reserved_access": False, "st_access": False, "shhs_access": False}
    if c != expected:
        raise ValueError("frozen prepared source configuration violated")


def training_config(c, seed):
    validate_source_config(c)
    if seed not in RECIPE["seeds"]:
        raise ValueError("unapproved source seed")
    return TrainConfig(epochs=c["max_epochs"], patience=c["patience"], lr=c["lr"], weight_decay=c["weight_decay"],
                       core_epochs=c["core_epochs"], encode_batch=c["encode_batch"], grad_clip=c["grad_clip"], seed=seed, deterministic=True)


def validate_launch_receipt(receipt, config):
    # The independently reviewed receipt is a future gate, never emitted by the
    # Phase II synthetic harness. A command-line flag alone cannot unlock data.
    if (receipt.get("scope") != "SC_TRAIN_VAL_ONLY_AUTHORIZED"
            or not receipt.get("authorization_reference") or receipt.get("config_sha256") != digest(config)
            or receipt.get("implementation_sha256") != implementation_snapshot()["sha256"]
            or receipt.get("split_fingerprint") != SPLIT or receipt.get("split_file_sha256") != SPLIT_SHA
            or receipt.get("cuda_exact_resume_passed") is not True or receipt.get("cpu_regression_passed") is not True
            or receipt.get("gpu_feasibility_passed") is not True):
        raise ValueError("source launch BLOCKED: missing/mismatched approval or correctness receipt")
    required = ["normalization_path", "normalization_sha256", "normalization_receipt_path", "normalization_receipt_sha256", "banks", "correctness_artifacts"]
    if any(not receipt.get(k) for k in required) or set(receipt["banks"]) != {"123", "456", "789"}:
        raise ValueError("source launch BLOCKED: normalization/bank/correctness proofs missing")
    for proof in receipt["correctness_artifacts"]:
        if file_sha(proof["path"]) != proof["sha256"]:
            raise ValueError("source launch BLOCKED: correctness artifact digest changed")


def load_role(frozen, role):
    if role not in {"train", "val"}:
        raise ValueError("reserved/external roles prohibited before any file access")
    from mist_transfer.protocol import load_development
    return load_development(frozen, role)


def validate_normalization(saved, train_rows, val_rows):
    from mist_transfer.experiment import source_input_proof
    norm = saved["statistics"]
    if (saved["input_proof"] != source_input_proof(train_rows + val_rows)
            or norm.get("fit_role") != "train" or norm.get("label_access") is not False
            or norm.get("physical_epoch_count") != 79984
            or norm.get("fit_subjects") != sorted({r["subject_id"] for r in train_rows})
            or norm.get("source_hashes") != {r["recording_id"]: r["source_sha256"] for r in train_rows}
            or not torch.isfinite(torch.tensor([norm["mean_uV"], norm["std_uV"]])).all()
            or norm["std_uV"] <= 0):
        raise ValueError("normalization TRAIN-only provenance invalid; never refit")
    return norm


def run_source(config_path, split_path, output, *, authorize_source_train_val=False, launch_receipt_path=None,
               resume=False, preflight_only=True):
    if not authorize_source_train_val:
        raise ValueError("source data BLOCKED: separate explicit source authorization required")
    if launch_receipt_path is None:
        raise ValueError("source data BLOCKED: independently reviewed launch receipt required")
    config = json.loads(Path(config_path).read_text())
    validate_source_config(config)
    receipt = json.loads(Path(launch_receipt_path).read_text())
    validate_launch_receipt(receipt, config)
    configure_precision("cuda")  # fail before opening source metadata if unusable
    torch.set_num_threads(config["threads"])
    output = Path(output).resolve()
    private_root = ROOT / "mist_transfer_runs/mist_cove/phase2_source_v1"
    if not output.is_relative_to(private_root):
        raise ValueError("source outputs must use the dedicated ignored Phase II directory")
    # Exact split bytes are checked before parsing; the existing parser checks
    # its independent receipt and all identity/role overlaps. No split regeneration.
    if file_sha(split_path) != SPLIT_SHA:
        raise ValueError("frozen source split SHA256 mismatch")
    from mist_transfer.protocol import read_frozen_split
    frozen = read_frozen_split(split_path)
    if frozen["split_fingerprint"] != SPLIT:
        raise ValueError("frozen source split fingerprint mismatch")
    for path_key, hash_key in (("normalization_path", "normalization_sha256"), ("normalization_receipt_path", "normalization_receipt_sha256")):
        if file_sha(receipt[path_key]) != receipt[hash_key]:
            raise ValueError("verified normalization proof changed")
    prior = json.loads(Path(receipt["normalization_receipt_path"]).read_text())
    if prior["outputs"]["normalization.json"] != receipt["normalization_sha256"]:
        raise ValueError("normalization trusted completion binding invalid")
    train_rows, val_rows = load_role(frozen, "train"), load_role(frozen, "val")
    norm = validate_normalization(json.loads(Path(receipt["normalization_path"]).read_text()), train_rows, val_rows)
    from mist_transfer.source_data import SourceRecording, validate_bank
    from mist_transfer.full_experiment import verify_grid
    train_records = [SourceRecording(r, norm, smoke=False) for r in train_rows]
    val_records = [SourceRecording(r, norm, smoke=False) for r in val_rows]
    grid = verify_grid(train_records, val_records)
    banks = {}
    for seed, proof in receipt["banks"].items():
        if file_sha(proof["path"]) != proof["sha256"]:
            raise ValueError("trusted historical TRAIN bank changed")
        # No fallback to unsafe unpickling. Incompatible legacy serialization is
        # a blocker for a separately reviewed artifact conversion, not a bypass.
        bank = torch.load(proof["path"], map_location="cpu", weights_only=True)
        if bank.get("seed") != int(seed):
            raise ValueError("TRAIN bank seed mismatch")
        validate_bank(bank, train_rows, norm, verify_waveforms=False)
        banks[int(seed)] = bank
    provenance = {"scope": "SC_TRAIN_VAL_ONLY", "config": config, "launch_receipt_sha256": file_sha(launch_receipt_path),
                  "code": implementation_snapshot(), "grid": grid, "normalization_sha256": receipt["normalization_sha256"]}
    # Preflight is explicit source metadata/contract access, not a Phase II action.
    if preflight_only:
        return {"status": "PREFLIGHT_ONLY_NO_TRAINING", "grid": grid, "code": provenance["code"]["sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in config["seeds"]:
        for name in config["variants"]:
            seed_all(seed, True)
            model = build_adapter(name, norm, banks[seed])
            run = output / f"seed_{seed}" / name
            _, metrics, _ = sealed_train(model, train_records, val_records, training_config(config, seed), run,
                                         provenance=provenance, device="cuda", bank=banks[seed], resume=resume and run.exists())
            results.append({"seed": seed, "variant": name, "parameters": sum(p.numel() for p in model.parameters()), "metrics": metrics})
    atomic_json({"status": "SOURCE_ONLY_COMPLETE", "runs": results, "provenance": provenance}, output / "results.json")
    return results

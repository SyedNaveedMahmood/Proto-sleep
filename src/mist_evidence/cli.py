"""Development-only runner. All prior EDF-20 results remain development evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

from .data import (build_anchors, edf_manifest, fingerprint, load_split, read_manifest,
                   sha256_file, validate_manifest)
from .explain import export_explanation
from .model import EvidenceModel, ModelConfig
from .runtime import (TrainConfig, atomic_json, atomic_torch_save, load_trusted,
                      seed_all, train, write_csv)
from .transport import TransportEvidenceModel, TransportModelConfig
from .transport_explain import export_transport_explanation

VARIANTS = ("transport", "evidence", "summary_only", "observed_only", "no_summary", "no_amplitude", "no_context", "no_crf", "raw_context")


def variant_config(name: str, base: ModelConfig) -> ModelConfig:
    changes = {"transport": {}, "evidence": {}, "summary_only": {"summary_only": True},
               "observed_only": {"neural_fraction_cap": 0.0}, "no_summary": {"use_summary": False},
               "no_amplitude": {"use_amplitude": False},
               "no_context": {"radius": 0, "use_crf": False}, "no_crf": {"use_crf": False},
               "raw_context": {"raw_control": True}}
    return replace(base, **changes[name])


def implementation_hash():
    root = Path(__file__).parent
    return fingerprint({p.name: sha256_file(p) for p in sorted(root.glob("*.py"))})


def synthetic_manifest(root: Path) -> dict:
    """Small artificial signals for software testing, NOT research evidence."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    entries = []
    t = np.arange(3000, dtype=np.float32) / 100
    for subject, split in (("synthetic_train_a", "train"), ("synthetic_train_b", "train"), ("synthetic_val", "val")):
        labels = np.tile(np.arange(5), 2)
        x = np.stack([np.sin(2*np.pi*(1+int(y)*3)*t) + 0.25*rng.normal(size=3000) for y in labels])
        path = root / f"{subject}.npz"
        np.savez(path, x=x.astype(np.float32), y=labels, fs=100, epoch_indices=np.arange(len(x)))
        entries.append({"path": str(path.resolve()), "subject": subject, "split": split})
    # This nonexistent reserved path is intentional: any test-opening bug fails.
    entries.append({"path": str((root / "DO_NOT_OPEN_TEST.npz").resolve()), "subject": "synthetic_test", "split": "test"})
    return {"dataset": "synthetic_software_test", "purpose": "development_only", "records": entries}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "smoke", "train"))
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--data-dir", type=Path)
    sources.add_argument("--manifest", type=Path)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variants", default="transport,evidence,summary_only,raw_context")
    p.add_argument("--seeds", default="123")
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--core-epochs", type=int, default=16)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--anchor-method", choices=("medoid", "random"), default="medoid")
    p.add_argument("--assume-contiguous", action="store_true")
    p.add_argument("--epoch-only", action="store_true", help="No context/CRF; safe when original epoch indices are absent")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--explain", type=int, default=1, help="deterministic validation examples per interpretable model")
    args = p.parse_args(argv)
    if args.mode != "smoke" and args.data_dir is None and args.manifest is None:
        p.error("supply --data-dir or --manifest")
    args.variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not args.variant_list or any(v not in VARIANTS for v in args.variant_list) or len(set(args.variant_list)) != len(args.variant_list):
        p.error("variants must be unique members of " + ",".join(VARIANTS))
    try:
        args.seed_list = [int(v) for v in args.seeds.split(",")]
    except ValueError:
        p.error("seeds must be comma-separated integers")
    if not args.seed_list or min(args.seed_list) < 0 or len(set(args.seed_list)) != len(args.seed_list):
        p.error("seeds must be unique nonnegative integers")
    if args.threads < 1 or args.explain < 0:
        p.error("invalid threads/explain")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = (synthetic_manifest(out / "synthetic") if args.mode == "smoke" else
                read_manifest(args.manifest.expanduser().resolve()) if args.manifest else
                edf_manifest(args.data_dir, args.fold))
    validate_manifest(manifest)
    tr = load_split(manifest, "train", args.assume_contiguous, args.epoch_only)
    va = load_split(manifest, "val", args.assume_contiguous, args.epoch_only)
    if {r.file_hash for r in tr} & {r.file_hash for r in va}:
        raise ValueError("duplicate recording bytes across train/validation")
    counts = np.bincount(np.concatenate([r.y for r in tr]), minlength=5)
    if np.any(counts == 0):
        raise ValueError("training data must contain all five mapped stages")
    selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(selected_device)
    io = {"manifest": manifest, "opened_records": [{"path": r.path, "subject": r.subject,
          "sha256": r.file_hash, "continuity_assumed": r.continuity_assumed,
          "indices_known": r.indices_known, "independent_epochs": r.independent_epochs,
          "n_epochs": len(r.y), "segments": r.segments()} for r in tr+va],
          "test_files_opened": False, "development_only": True, "torch_version": str(torch.__version__),
          "numpy_version": str(np.__version__), "device": str(device), "precision": "float32",
          "implementation_sha256": implementation_hash()}
    atomic_json(io, out / "data_audit.json")
    print("MIST-EVIDENCE / MIST-MORPH v2 research suite | development only | FP32", flush=True)
    print("device:", device, "| train epochs:", sum(len(r.y) for r in tr), "| val epochs:", sum(len(r.y) for r in va), flush=True)
    print("Train stage counts:", counts.tolist(), flush=True)
    print("Reserved test files opened: NO", flush=True)
    if any(r.continuity_assumed for r in tr+va):
        print("WARNING: temporal continuity is a USER ASSUMPTION, not verified from original epoch indices.", flush=True)
    if args.mode == "preflight":
        print("PREFLIGHT COMPLETE; no model trained.", flush=True)
        return
    base = ModelConfig(radius=0, use_crf=False) if args.epoch_only else ModelConfig()
    if args.epoch_only:
        print("EPOCH-ONLY MODE: no inter-epoch context or CRF. Not a full architecture result.", flush=True)
    per_subject = 64
    if args.mode == "smoke":
        base = replace(base, scales=(100, 200), prototypes_per_scale=2, embedding_dim=8, radius=2)
        args.variant_list, args.seed_list = ["transport", "evidence"], [123]
        args.max_epochs, args.patience, args.core_epochs, args.encode_batch = 2, 2, 4, 4
        per_subject = 8
    bank_identity = {"data": io, "model": base.dictionary(), "seed": 1337, "method": args.anchor_method}
    bank_path = out / "anchor_bank.pt"
    if args.resume and bank_path.exists():
        bank = load_trusted(bank_path)
        if bank["identity"] != fingerprint(bank_identity):
            raise ValueError("anchor bank data/config/code identity changed")
    elif bank_path.exists():
        raise FileExistsError("anchor bank exists; use --resume or a new output directory")
    else:
        bank = build_anchors(tr, base, seed=1337, per_subject=per_subject, method=args.anchor_method)
        bank["identity"] = fingerprint(bank_identity)
        atomic_torch_save(bank, bank_path)
    atomic_json({k: v for k, v in bank.items() if k != "waveforms"}, out / "anchor_provenance.json")
    rows = []
    for seed in args.seed_list:
        for variant in args.variant_list:
            seed_all(seed)
            cfg = variant_config(variant, base)
            if variant == "transport":
                transport_cfg = TransportModelConfig(**cfg.dictionary())
                model = TransportEvidenceModel(transport_cfg, bank["waveforms"], bank["amplitude_stats"])
            else:
                model = EvidenceModel(cfg, bank["waveforms"], bank["amplitude_stats"])
            train_cfg = TrainConfig(epochs=args.max_epochs, patience=args.patience, seed=seed,
                                    core_epochs=args.core_epochs, encode_batch=args.encode_batch)
            run = out / f"seed_{seed}" / variant
            print(f"\nSTART seed={seed} variant={variant} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
            started = time.perf_counter()
            model, metrics, predictions = train(model, tr, va, train_cfg, run, bank, io, device, args.resume)
            row = {"seed": seed, "variant": variant, "val_subject_macro_f1": metrics["mean_subject_macro_f1"],
                   "val_pooled_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "seconds_this_invocation": time.perf_counter()-started, "test_accessed": False,
                   "transport_audit": ""}
            if variant == "transport":
                audit_x = torch.from_numpy(va[0].x[:1]).to(device)
                row["transport_audit"] = json.dumps(model.transport_audit(audit_x), sort_keys=True)
            rows.append(row)
            write_csv(rows, out / "results.csv")
            if variant == "evidence":
                for i in range(args.explain):
                    r = va[i % len(va)]
                    ep = min(len(r.y)-1, len(r.y)//2 + i//len(va))
                    export_explanation(model, r, ep, bank, run / "explanations" / f"example_{i:02d}", device, args.encode_batch)
            elif variant == "transport":
                for i in range(args.explain):
                    r = va[i % len(va)]
                    ep = min(len(r.y)-1, len(r.y)//2 + i//len(va))
                    export_transport_explanation(model, r, ep, bank, run / "explanations" / f"example_{i:02d}", device, args.encode_batch)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    atomic_json({"status": "complete", "runs": rows, "development_only": True,
                 "clinical_labels_validated": False, "sota_claim_allowed": False,
                 "test_files_opened": False}, out / "SUITE_COMPLETE.json")
    print("\nSUITE COMPLETE. Results:", out / "results.csv", flush=True)
    print("This is software/research evidence, not proof of clinical morphology, novelty, or SOTA.", flush=True)


if __name__ == "__main__":
    main()

"""Command-line orchestration; all commands are foreground and print flushed progress."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .audit import audit_bank, event_scores
from .data import STAGES, load_split, make_edf_manifest, validate_manifest, write_json
from .explain import explain_epoch
from .model import EvidenceConfig
from .train import TrainConfig, fit_context, fit_local


def synthetic_manifest(output: Path):
    """Tiny shape-correct integration fixture. Its scores are NOT research results."""
    data = output / "synthetic_data"
    data.mkdir(parents=True, exist_ok=True)
    path = output / "synthetic_manifest.json"
    rng = np.random.default_rng(83)
    records = []
    for subject in range(5):
        y = np.tile(np.arange(5), 2).astype(np.int64)
        t = np.arange(3000, dtype=np.float32) / 100
        x = np.stack([np.sin(2*np.pi*[2, 5, 9, 13, 20][int(c)]*t) + rng.normal(0, .1, len(t)) for c in y]).astype(np.float32)
        name = f"synthetic_{subject}.npz"
        role = "train" if subject < 3 else "val" if subject == 3 else "test"
        if role == "test":
            # A valid run must never attempt to load this deliberately invalid archive.
            (data / name).write_bytes(b"LOCKED TEST: NOT A VALID NPZ")
        else:
            np.savez(data / name, x=x, y=y, fs=100, epoch_index=np.arange(len(y)))
        records.append({"path": name, "subject": str(subject), "split": role})
    write_json(path, {"schema": 1, "dataset": "synthetic_integration_only", "data_root": str(data.resolve()),
                      "development_only": True, "fs": 100, "epoch_samples": 3000,
                      "label_mapping": list(STAGES), "records": records})
    return path


def load_configs(path):
    doc = json.loads(Path(path).read_text()) if path else {}
    if set(doc) - {"model", "training"}:
        raise ValueError("Config accepts only model and training sections")
    return EvidenceConfig(**doc.get("model", {})), TrainConfig(**doc.get("training", {}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental MIST-Evidence. No SOTA or clinical-event claims.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-split", help="Create EDF-20 development manifest from filenames, without reading arrays")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("inspect", help="Validate train/val NPZ contract and report continuity; never opens test")
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("audit-bank")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score-events")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("fit", "fit-context", "explain", "suite", "smoke"):
        p = sub.add_parser(command)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--threads", type=int, default=4)
        if command != "smoke":
            p.add_argument("--manifest", type=Path, required=True)
        if command in {"fit", "suite"}:
            p.add_argument("--config", type=Path)
            p.add_argument("--resume", action="store_true")
        if command == "fit":
            p.add_argument("--seed", type=int)
            p.add_argument("--epochs", type=int)
            p.add_argument("--patience", type=int)
            p.add_argument("--arm", choices=("evidence", "descriptor", "dense", "attnsleep"))
        if command == "suite":
            p.add_argument("--seeds", default="123,456,789")
            p.add_argument("--arms", default="descriptor,dense,evidence,attnsleep")
            p.add_argument("--with-context", action="store_true")
            p.add_argument("--assume-contiguous", action="store_true")
        if command in {"fit-context", "explain"}:
            p.add_argument("--checkpoint", type=Path, required=True)
        if command == "fit-context":
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--patience", type=int, default=5)
            p.add_argument("--assume-contiguous", action="store_true")
            p.add_argument("--resume", action="store_true")
        if command == "explain":
            p.add_argument("--recording")
            p.add_argument("--epoch", type=int, default=0)
            p.add_argument("--context-checkpoint", type=Path)
            p.add_argument("--perturb", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "audit-bank":
        print(json.dumps(audit_bank(args.checkpoint, args.output), indent=2), flush=True)
        return
    if args.command == "score-events":
        predictions = json.loads(args.predictions.read_text())
        reference = json.loads(args.reference.read_text())
        result = event_scores(predictions, reference, args.iou_threshold)
        write_json(args.output, result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.command == "make-split":
        result = make_edf_manifest(args.data_dir, args.fold, args.output)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.command == "inspect":
        doc, _ = validate_manifest(args.manifest)
        out = {"dataset": doc["dataset"], "test_arrays_opened": False}
        for role in ("train", "val"):
            ns = load_split(args.manifest, role)
            y = np.concatenate([n.y for n in ns])
            out[role] = {"subjects": sorted({n.subject for n in ns}), "n_recordings": len(ns),
                         "n_epochs": len(y), "unknown_labels": int((y < 0).sum()),
                         "class_counts": np.bincount(y[y >= 0], minlength=5).tolist(),
                         "continuity_verified": all(n.continuity_verified for n in ns)}
        print(json.dumps(out, indent=2), flush=True)
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. No silent fallback.")
    if args.command == "smoke":
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Use a new, empty smoke output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = synthetic_manifest(args.output_dir)
        mc = EvidenceConfig(scales_samples=(100, 200), anchors_per_class=1, embedding_dim=8, pool_per_subject=20)
        tc = TrainConfig(epochs=2, patience=2, batch_size=5, log_every=2)
        fit_local(manifest, args.output_dir / "local", mc, tc, device=device)
        fit_context(manifest, args.output_dir / "local/best.pt", args.output_dir / "context", epochs=2, patience=2, device=device)
        explain_epoch(manifest, args.output_dir / "local/best.pt", args.output_dir / "explanation",
                      context_checkpoint=args.output_dir / "context/best.pt", device=device, top_k=2, perturb=True)
        print("SYNTHETIC FULL PIPELINE PASSED. This is execution evidence, not sleep-staging performance.", flush=True)
    elif args.command == "fit":
        mc, tc = load_configs(args.config)
        change = {k: getattr(args, k) for k in ("seed", "epochs", "patience") if getattr(args, k) is not None}
        tc = TrainConfig(**{**asdict(tc), **change})
        arm = args.arm or mc.arm
        mc = EvidenceConfig(**{**asdict(mc), "arm": "evidence" if arm == "attnsleep" else arm})
        fit_local(args.manifest, args.output_dir, mc, tc, device, args.resume, arm == "attnsleep")
    elif args.command == "fit-context":
        result = fit_context(args.manifest, args.checkpoint, args.output_dir, epochs=args.epochs,
                             patience=args.patience, assume_contiguous=args.assume_contiguous,
                             device=device, resume=args.resume)
        print(json.dumps(result, indent=2), flush=True)
    elif args.command == "explain":
        explain_epoch(args.manifest, args.checkpoint, args.output_dir, args.recording, args.epoch,
                      args.context_checkpoint, device, perturb=args.perturb)
    elif args.command == "suite":
        mc, tc = load_configs(args.config)
        seeds = [int(v) for v in args.seeds.split(",")]
        arms = args.arms.split(",")
        if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms) or not set(arms) <= {"descriptor", "dense", "evidence", "attnsleep"}:
            raise ValueError("Duplicate seeds/arms or unknown arm")
        rows = []
        for seed in seeds:
            for arm in arms:
                out = args.output_dir / f"seed_{seed}" / arm
                model_cfg = EvidenceConfig(**{**asdict(mc), "arm": "evidence" if arm == "attnsleep" else arm,
                                               "aux_morph_weight": 0.0 if arm == "descriptor" else mc.aux_morph_weight})
                done = fit_local(args.manifest, out, model_cfg, TrainConfig(**{**asdict(tc), "seed": seed}),
                                 device, args.resume, arm == "attnsleep")
                row = {"seed": seed, "arm": arm, "local_val_macro_f1": done["validation"]["macro_f1"],
                       "best_epoch": done["best_epoch"], "context_val_macro_f1": None}
                if args.with_context:
                    context = fit_context(args.manifest, out / "best.pt", out / "context",
                                          assume_contiguous=args.assume_contiguous, device=device, resume=args.resume)
                    row["context_val_macro_f1"] = context["validation"]["macro_f1"]
                rows.append(row)
                write_json(args.output_dir / "suite_partial.json", rows)
                print(json.dumps(row), flush=True)
        with (args.output_dir / "suite_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        write_json(args.output_dir / "SUITE_COMPLETE.json", {"development_only": True, "rows": rows,
                                                             "test_arrays_opened": False})
        print("SUITE COMPLETE. Seeds are repetitions, not independent held-out subjects.", flush=True)


if __name__ == "__main__":
    main()

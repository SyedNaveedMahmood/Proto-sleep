"""Matched five-arm Phase M0 training with private resumable states."""
from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .augmentation import paired_view
from .core import MISTSleep, mpi_terms, prototype_diagnostics, supcon_loss
from .metrics import fixed_five
from .source import PRIVATE, PUBLIC, load_full_train

ARMS = ("A0", "M0", "M1", "M2", "M3")


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
            ("core.py", "augmentation.py", "source.py", "metrics.py", "experiment.py")}


def freeze_protocol() -> dict:
    if (PUBLIC / "frozen_protocol.json").exists():
        raise FileExistsError("protocol already frozen")
    protocol = {"version": "mist-sleep-method-recovery-v1", "arms": list(ARMS),
                "seeds": [123, 456], "steps": 1200, "milestones": [300, 750, 1200],
                "warmup_steps": 50, "batch_size": 16, "optimizer": "AdamW",
                "learning_rate": .001, "weight_decay": .0001, "gradient_clip_norm": 5.0,
                "sampling": "fit-only class-balanced; replacement; same per-seed stream across arms",
                "views": "one original and one deterministic bounded augmented view for every arm",
                "stage_loss": "mean cross-entropy of both views; no class weighting because balanced sampling",
                "mpi_embedding_coefficient": .1, "mpi_assignment_coefficient": .1,
                "supcon_coefficient": .05, "supcon_temperature": .1,
                "prototype_regularization": {"commitment": .05, "balance": .01, "separation": .01},
                "prototype_count": 64, "prototype_temperature": .1,
                "prototype_initialization": "50 shared stage-loss warmup steps, then fit-only cosine k-means, 4096 tokens, 12 iterations",
                "development_selection": "fixed final 1200 update state; milestones descriptive only",
                "primary_endpoint": "mean participant fixed-five Macro-F1 on full scored internal-development TRAIN partition",
                "primary_contrast": "M2 minus M0", "secondary_contrasts": ["M2-M1", "M3-M2", "M3-M1", "M0-A0"],
                "resource_limits": {"scientific_trajectories": 10, "updates_each": 1200,
                                    "scientific_updates": 12000, "GPU_seconds": 21600, "CPU_seconds": 43200,
                                    "engineering_updates": 200},
                "go_rule": {"nuisance_mean_participant_macro_f1_delta_min": .02,
                            "clean_mean_participant_macro_f1_delta_min": -.02,
                            "required_seeds": 2, "no_N1_REM_collapse": True,
                            "no_prototype_collapse": True},
                "source_hashes": source_hashes()}
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def verify_protocol() -> dict:
    protocol = json.loads((PUBLIC / "frozen_protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes() or protocol["arms"] != list(ARMS):
        raise RuntimeError("frozen implementation or arms changed")
    return protocol


def new_model(arm: str) -> torch.nn.Module:
    if arm == "A0":
        from protosleep.attnsleep import AttnSleepBaseline, init_attnsleep_weights
        model = AttnSleepBaseline()
        model.apply(init_attnsleep_weights)
        return model
    if arm in ARMS:
        return MISTSleep()
    raise ValueError("unknown frozen arm")


def _logits(output):
    return output if isinstance(output, torch.Tensor) else output["logits"]


def make_sampler(index: dict[str, np.ndarray], seed: int):
    fit = (index["role"] == 0) & (index["stage"] >= 0)
    pools = [np.flatnonzero(fit & (index["stage"] == stage)) for stage in range(5)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("missing stage in fit partition")
    rng = np.random.default_rng(seed)
    return pools, rng


def sample_batch(pools, rng, batch_size: int):
    stage = rng.integers(0, 5, size=batch_size)
    return np.array([pools[int(s)][rng.integers(len(pools[int(s)]))] for s in stage], dtype=np.int64)


def _regularizer(output, model):
    q = output["q"]
    token = F.normalize(output["token"], dim=-1, eps=1e-8)
    p = F.normalize(model.bank.vectors, dim=-1, eps=1e-8)
    similarity = token @ p.T
    commitment = (q.detach() * (1 - similarity)).sum(-1).mean()
    usage = q.mean((0, 1)).clamp_min(1e-8)
    balance = (usage * (usage * len(usage)).log()).sum()
    gram = p @ p.T
    off = gram[~torch.eye(len(p), dtype=torch.bool, device=p.device)]
    separation = F.relu(off - .3).square().mean()
    return .05 * commitment + .01 * balance + .01 * separation


def one_update(model, optimizer, x, y, view, arm: str, auxiliary: bool = True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    clean = model(x)
    transformed = model(view)
    stage_loss = (F.cross_entropy(_logits(clean), y) + F.cross_entropy(_logits(transformed), y)) / 2
    loss = stage_loss
    record = {"stage_loss": float(stage_loss.detach())}
    if arm != "A0":
        regularizer = (_regularizer(clean, model) + _regularizer(transformed, model)) / 2
        loss = loss + regularizer
        record["prototype_regularizer"] = float(regularizer.detach())
        if auxiliary and arm in ("M2", "M3"):
            distance, kl = mpi_terms(clean, transformed)
            loss = loss + .1 * distance + .1 * kl
            record.update(mpi_embedding=float(distance.detach()), mpi_assignment=float(kl.detach()))
        if auxiliary and arm in ("M1", "M3"):
            contrastive = supcon_loss(torch.cat((clean["embedding"], transformed["embedding"])),
                                      torch.cat((y, y)), .1)
            loss = loss + .05 * contrastive
            record["supcon"] = float(contrastive.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite scientific loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    record.update(total_loss=float(loss.detach()), gradient_norm=float(grad.detach()))
    return record


@torch.no_grad()
def initialize_bank(model: MISTSleep, wave, index, seed: int, device: torch.device) -> dict:
    fit = np.flatnonzero((index["role"] == 0) & (index["stage"] >= 0))
    rng = np.random.default_rng(seed + 9407)
    chosen = rng.choice(fit, size=256, replace=len(fit) < 256)
    model.eval()
    embeddings = []
    for batch in np.array_split(chosen, 16):
        x = torch.from_numpy(np.asarray(wave[batch]).copy()).to(device)
        tokens = model.encoder(model.tokenizer(x)).reshape(-1, model.geometry.channels)
        embeddings.append(tokens.detach().cpu())
    bank_input = torch.cat(embeddings)[:4096]
    model.bank.initialize_from_fit(bank_input, seed)
    return {"fit_only": True, "token_count": len(bank_input),
            "token_sha256": hashlib.sha256(bank_input.numpy().tobytes()).hexdigest()}


@torch.no_grad()
def evaluate(model, wave, index, device, max_examples: int | None = None) -> tuple[dict, dict]:
    positions = np.flatnonzero((index["role"] == 1) & (index["stage"] >= 0))
    if max_examples is not None:
        positions = positions[:max_examples]
    model.eval()
    logits, embedding, assignment = [], [], []
    for batch in np.array_split(positions, max(1, (len(positions)+63)//64)):
        if not len(batch):
            continue
        x = torch.from_numpy(np.asarray(wave[batch]).copy()).to(device)
        output = model(x)
        logits.append(_logits(output).float().cpu().numpy())
        if isinstance(output, dict):
            embedding.append(output["embedding"].float().cpu().numpy())
            assignment.append(output["q"].float().cpu().numpy())
    predictions = np.concatenate(logits)
    metrics = fixed_five(index["stage"][positions], predictions, index["person"][positions])
    diagnostics = {}
    if assignment:
        q = torch.from_numpy(np.concatenate(assignment))
        diagnostics = prototype_diagnostics(q, model.bank.vectors)
        diagnostics["embedding_feature_variance"] = float(np.concatenate(embedding).var(axis=0).mean())
    return metrics, diagnostics


def _checkpoint(path, model, optimizer, sampler, augment, step, arm, seed, protocol_digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sampler": sampler.bit_generator.state, "augment": augment.get_state(),
             "torch_cpu_rng": torch.get_rng_state(),
             "torch_cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
             "step": step, "arm": arm, "seed": seed, "protocol_sha256": protocol_digest}
    temp = path.with_suffix(".writing")
    torch.save(state, temp)
    os.replace(temp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_checkpoint(path, model, optimizer, sampler, augment, expected_arm, expected_seed, expected_protocol):
    """Restore a phase-owned private state at a recorded step, without custom globals."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["arm"], state["seed"], state["protocol_sha256"]) != (
            expected_arm, expected_seed, expected_protocol):
        raise RuntimeError("checkpoint identity differs")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.bit_generator.state = state["sampler"]
    augment.set_state(state["augment"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if state["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda_rng"])
    return int(state["step"])


def run_arm(arm: str, seed: int, device: torch.device, max_steps: int | None = None) -> dict:
    protocol = verify_protocol()
    if arm not in protocol["arms"] or seed not in protocol["seeds"]:
        raise ValueError("arm/seed not frozen")
    steps = protocol["steps"] if max_steps is None else min(max_steps, protocol["steps"])
    if max_steps is not None and max_steps > 200:
        raise ValueError("engineering probe exceeds 200 updates")
    path = PRIVATE / "models" / f"{arm}_{seed}"
    if path.exists():
        raise FileExistsError("run already attempted; no automatic rerun")
    wave, index = load_full_train()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = new_model(arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    pools, sampler = make_sampler(index, seed)
    augment = torch.Generator(device="cpu").manual_seed(seed + 100000)
    protocol_digest = hashlib.sha256((PUBLIC / "frozen_protocol.json").read_bytes()).hexdigest()
    path.mkdir(parents=True)
    (path / "started.json").write_text(json.dumps({"arm": arm, "seed": seed, "steps": steps,
                                                    "protocol_sha256": protocol_digest}) + "\n")
    started = time.perf_counter()
    trajectory = []
    init_receipt = None
    milestones = set(protocol["milestones"]) if max_steps is None else {steps}
    for step in range(1, steps+1):
        if time.perf_counter()-started > 21600:
            raise TimeoutError("single-trajectory GPU wall ceiling")
        indices = sample_batch(pools, sampler, protocol["batch_size"])
        x_cpu = torch.from_numpy(np.asarray(wave[indices]).copy())
        y = torch.from_numpy(index["stage"][indices].astype("int64")).to(device)
        view = paired_view(x_cpu, augment)
        record = one_update(model, optimizer, x_cpu.to(device), y, view.to(device), arm,
                            auxiliary=step > protocol["warmup_steps"])
        if step == protocol["warmup_steps"] and arm != "A0":
            init_receipt = initialize_bank(model, wave, index, seed, device)
        if step not in milestones:
            continue
        metrics, diagnostics = evaluate(model, wave, index, device, max_examples=200 if max_steps else None)
        digest = _checkpoint(path / f"step_{step:04d}.pt", model, optimizer, sampler,
                             augment, step, arm, seed, protocol_digest)
        trajectory.append({"step": step, "development": metrics, "diagnostics": diagnostics,
                           "training_last": record, "checkpoint_sha256": digest,
                           "elapsed_seconds": time.perf_counter()-started})
    public = {"arm": arm, "seed": seed, "status": "COMPLETE" if max_steps is None else "ENGINEERING_PROBE",
              "steps": steps, "physical_train_views": 2*steps*protocol["batch_size"],
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "wall_seconds": time.perf_counter()-started,
              "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "prototype_initialization": init_receipt, "trajectory": trajectory}
    # Results contain only aggregates; all model weights and private predictions stay ignored.
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / f"{arm}_{seed}.json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    return public

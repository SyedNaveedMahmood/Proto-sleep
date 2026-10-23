"""Exact-byte authentication through the inherited patched restricted loader."""
from __future__ import annotations

from pathlib import Path
import torch

from mist_cove.phase11b.security import authenticated_snapshot, restricted_load, IntegrityError
from mist_sleep.core import MISTSleep
from .protocol import CHECKPOINTS, SOURCE_PROTOCOL_SHA256


def load_source(models_root, arm, seed):
    key = f"{arm}_{seed}"
    if key not in CHECKPOINTS:
        raise IntegrityError("unregistered model identity")
    path = Path(models_root).absolute() / key / "step_1200.pt"
    with authenticated_snapshot(path, CHECKPOINTS[key], allowlist=[path]) as snapshot:
        payload = restricted_load(snapshot)
    if (payload.get("arm"), payload.get("seed"), payload.get("step"), payload.get("protocol_sha256")) != (
            arm, seed, 1200, SOURCE_PROTOCOL_SHA256):
        raise IntegrityError("frozen final checkpoint metadata mismatch")
    model = MISTSleep()
    model.load_state_dict(payload["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 41429 or model.bank.vectors.shape != (64, 64):
        raise IntegrityError("source architecture mismatch")
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise IntegrityError("nonfinite source model")
    model.eval().requires_grad_(False)
    return model, {"arm": arm, "seed": seed, "step": 1200, "sha256": CHECKPOINTS[key],
                   "parameters": 41429, "prototypes": 64, "restricted_loader": True,
                   "optimizer_state_restored": False}

"""Bounded, token-aligned acquisition-like nuisance transforms.

Time shifting and frequency deletion are intentionally excluded from training:
neither has an unambiguous one-to-one token correspondence here.
"""
from __future__ import annotations

import torch


def paired_view(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Same sample count and physical token grid; no fit from development data."""
    if x.ndim != 3 or x.size(1) != 1 or x.size(2) != 3000 or not torch.isfinite(x).all():
        raise ValueError("expected finite physical epochs")
    # Generate on CPU so seed streams are device-independent and replayable.
    batch = len(x)
    amp = 0.95 + .10 * torch.rand((batch, 1, 1), generator=generator)
    offset = (2 * torch.rand((batch, 1, 1), generator=generator) - 1) * .05
    noise = torch.randn(tuple(x.shape), generator=generator) * .02
    t = torch.linspace(0, 1, x.size(-1)).view(1, 1, -1)
    phase = 2 * torch.pi * torch.rand((batch, 1, 1), generator=generator)
    wander = .03 * torch.sin(2 * torch.pi * .2 * t + phase)
    return x * amp.to(x.device) + offset.to(x.device) + noise.to(x.device) + wander.to(x.device)


def stress_view(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Separate destructive diagnostic, never part of MPI training."""
    noise = torch.randn(tuple(x.shape), generator=generator).to(x.device)
    return .5 * x + .25 * noise

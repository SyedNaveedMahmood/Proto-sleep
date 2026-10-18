"""Known-component signals, not clinical EEG simulators or human sleep labels."""
from dataclasses import dataclass
import torch
from torch import Tensor

# Two distinct narrowband waveforms per artificial class. No physiological claim.
FREQUENCIES = ((1.5, 2.5), (5., 7.), (9., 11.), (13., 15.), (20., 26.))


@dataclass
class SyntheticBatch:
    x: Tensor
    y: Tensor
    background: Tensor
    a: Tensor
    b: Tensor
    nuisance: Tensor
    intervals: Tensor  # [B,3,2]: A, B, label-independent nuisance


def make_batch(n: int, seed: int, *, random_presence: bool = False) -> SyntheticBatch:
    if n < 1:
        raise ValueError("positive synthetic batch required")
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 5
    background = torch.randn(n, 1, 3000, generator=g) * (0.7 + torch.rand(n, 1, 1, generator=g))
    a, b, nuisance = [torch.zeros_like(background) for _ in range(3)]
    intervals = torch.empty(n, 3, 2, dtype=torch.long)
    t = torch.arange(200) / 100
    envelope = torch.hann_window(200, periodic=False)
    for i in range(n):
        slots = torch.randperm(15, generator=g)[:3] * 200
        amp = float(20 + 20 * torch.rand((), generator=g))
        for j, component in enumerate((a, b, nuisance)):
            start = int(slots[j]); intervals[i, j] = torch.tensor([start, start + 200])
            frequency = FREQUENCIES[int(y[i])][j] if j < 2 else 35.
            phase = 2 * torch.pi * torch.rand((), generator=g)
            component[i, 0, start:start + 200] = amp * envelope * torch.sin(2 * torch.pi * frequency * t + phase)
        if random_presence:
            choice = int(torch.randint(3, (), generator=g))
            if choice == 0:
                a[i].zero_()
            elif choice == 1:
                b[i].zero_()
    return SyntheticBatch(background + a + b + nuisance, y, background, a, b, nuisance, intervals)


def perturbations(batch: SyntheticBatch) -> dict[str, Tensor]:
    # Matched L2 distortion: add an off-band component without removing A.
    a_norm = batch.a.flatten(1).norm(dim=1)
    nuisance_norm = batch.nuisance.flatten(1).norm(dim=1).clamp_min(1e-8)
    matched = batch.nuisance * (a_norm / nuisance_norm)[:, None, None]
    return {"clean": batch.x, "lose_a": batch.x - batch.a,
            "lose_b": batch.x - batch.b, "both_lost": batch.x - batch.a - batch.b,
            "matched_corruption": batch.x + matched,
            "attenuate_a": batch.x - .8 * batch.a,
            # Reversible calibrated measurement change; inverse is known, not learned.
            "calibrated_gain": ((batch.x * 1.7 + 4.) - 4.) / 1.7,
            "nuisance_only": batch.background + batch.nuisance}

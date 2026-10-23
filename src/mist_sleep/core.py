"""Frozen morphology architecture and independent MPI/SupCon objectives.

Inputs are fit-normalized 30-second, 100 Hz Fpz-Cz epochs. All temporal
operations remain inside the current physical epoch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Geometry:
    sample_rate: int = 100
    epoch_samples: int = 3000
    window_samples: int = 200
    hop_samples: int = 100
    channels: int = 64
    prototypes: int = 64
    temperature: float = 0.1

    @property
    def token_count(self) -> int:
        return 1 + (self.epoch_samples - self.window_samples) // self.hop_samples


class MorphologyTokenizer(nn.Module):
    def __init__(self, geometry: Geometry = Geometry()):
        super().__init__()
        self.geometry = geometry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.geometry
        if x.ndim != 3 or x.shape[1:] != (1, g.epoch_samples):
            raise ValueError("expected [batch, 1, 3000] physical epochs")
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite EEG")
        return x[:, 0].unfold(-1, g.window_samples, g.hop_samples)

    def bounds_seconds(self) -> torch.Tensor:
        g = self.geometry
        start = torch.arange(g.token_count, dtype=torch.float64) * g.hop_samples / g.sample_rate
        return torch.stack((start, start + g.window_samples / g.sample_rate), -1)


class MultiScaleEncoder(nn.Module):
    """Parallel local filters; 0.07, 0.31 and 1.01 second kernel spans."""
    def __init__(self, width: int = 64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 16, k, stride=4, padding=k // 2),
                          nn.GroupNorm(4, 16), nn.GELU(),
                          nn.Conv1d(16, 16, 5, stride=2, padding=2),
                          nn.GroupNorm(4, 16), nn.GELU(),
                          nn.AdaptiveAvgPool1d(1))
            for k in (7, 31, 101)
        ])
        self.project = nn.Sequential(nn.Linear(48, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("windows must be [batch, tokens, samples]")
        b, n, w = windows.shape
        flat = windows.reshape(b * n, 1, w)
        features = torch.cat([branch(flat).squeeze(-1) for branch in self.branches], -1)
        return self.project(features).reshape(b, n, -1)


class SharedPrototypeBank(nn.Module):
    def __init__(self, count: int = 64, width: int = 64, temperature: float = .1):
        super().__init__()
        if count < 2 or width < 2 or temperature <= 0:
            raise ValueError("invalid prototype geometry")
        self.temperature = float(temperature)
        self.vectors = nn.Parameter(F.normalize(torch.randn(count, width), dim=-1))

    def forward(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(token, dim=-1, eps=1e-8)
        p = F.normalize(self.vectors, dim=-1, eps=1e-8)
        q = torch.softmax(z @ p.T / self.temperature, dim=-1)
        return q, q @ p

    @torch.no_grad()
    def initialize_from_fit(self, fit_tokens: torch.Tensor, seed: int, iterations: int = 12) -> None:
        """Deterministic cosine k-means on *caller-verified fit-only* tokens."""
        if fit_tokens.ndim != 2 or fit_tokens.size(1) != self.vectors.size(1):
            raise ValueError("fit token matrix shape")
        if fit_tokens.size(0) < self.vectors.size(0) or not torch.isfinite(fit_tokens).all():
            raise ValueError("insufficient finite fit-only tokens")
        z = F.normalize(fit_tokens.float().cpu(), dim=-1, eps=1e-8)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        center = z[torch.randperm(len(z), generator=generator)[:len(self.vectors)]].clone()
        for _ in range(iterations):
            labels = (z @ center.T).argmax(-1)
            sums = torch.zeros_like(center).index_add_(0, labels, z)
            counts = torch.bincount(labels, minlength=len(center))
            center = F.normalize(torch.where(counts[:, None] > 0, sums, center), dim=-1)
        self.vectors.copy_(center.to(self.vectors.device, self.vectors.dtype))


class MISTSleep(nn.Module):
    def __init__(self, geometry: Geometry = Geometry()):
        super().__init__()
        self.geometry = geometry
        self.tokenizer = MorphologyTokenizer(geometry)
        self.encoder = MultiScaleEncoder(geometry.channels)
        self.bank = SharedPrototypeBank(geometry.prototypes, geometry.channels, geometry.temperature)
        self.position = nn.Parameter(torch.zeros(1, geometry.token_count, geometry.channels))
        self.temporal = nn.GRU(geometry.channels, geometry.channels, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * geometry.channels), nn.Linear(2 * geometry.channels, 5))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encoder(self.tokenizer(x))
        q, prototype_mix = self.bank(tokens)
        temporal, _ = self.temporal(tokens + self.position)
        embedding = F.normalize(torch.cat((temporal.mean(1), prototype_mix.mean(1)), -1), dim=-1, eps=1e-8)
        logits = self.head(torch.cat((temporal.mean(1), prototype_mix.mean(1)), -1))
        return {"logits": logits, "token": tokens, "q": q, "embedding": embedding}


def mpi_terms(clean: dict[str, torch.Tensor], view: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Within-instance global representation distance and aligned token KL."""
    if clean["q"].shape != view["q"].shape or clean["embedding"].shape != view["embedding"].shape:
        raise ValueError("MPI requires one-to-one token and instance correspondence")
    a = F.normalize(clean["embedding"].float(), dim=-1, eps=1e-8)
    b = F.normalize(view["embedding"].float(), dim=-1, eps=1e-8)
    distance = torch.linalg.vector_norm(a - b, dim=-1).mean()
    p = clean["q"].float().clamp_min(1e-8)
    q = view["q"].float().clamp_min(1e-8)
    kl = (p * (p.log() - q.log())).sum(-1).mean()
    return distance, kl


def supcon_loss(embedding: torch.Tensor, labels: torch.Tensor, temperature: float = .1) -> torch.Tensor:
    """Khosla-style supervised contrastive loss; anchors without positives are omitted."""
    if embedding.ndim != 2 or len(embedding) != len(labels) or temperature <= 0:
        raise ValueError("SupCon shape or temperature")
    z = F.normalize(embedding.float(), dim=-1, eps=1e-8)
    similarity = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    log_probability = similarity - torch.logsumexp(similarity.masked_fill(eye, -torch.inf), dim=1, keepdim=True)
    count = positive.sum(1)
    if not (count > 0).any():
        return embedding.sum() * 0
    return -(log_probability.masked_fill(~positive, 0).sum(1)[count > 0] / count[count > 0]).mean()


def prototype_diagnostics(q: torch.Tensor, vectors: torch.Tensor) -> dict[str, float]:
    usage = q.detach().float().mean((0, 1))
    entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
    p = F.normalize(vectors.detach().float(), dim=-1, eps=1e-8)
    sim = p @ p.T
    off = sim[~torch.eye(len(p), dtype=torch.bool, device=p.device)]
    return {"effective_prototypes": float(entropy.exp()), "dead_prototypes": int((usage < .001).sum()),
            "assignment_entropy": float(-(q.detach().float() * q.detach().float().clamp_min(1e-8).log()).sum(-1).mean()),
            "near_duplicate_pairs": int((off > .99).sum().item() // 2)}

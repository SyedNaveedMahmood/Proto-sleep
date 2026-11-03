"""Local waveform evidence, exact additive context, and a linear-chain CRF.

Prototype waveforms are immutable buffers copied from TRAINING recordings. They
are NOT free latent vectors later visualized with an approximate decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
STAGES = ("Wake", "N1", "N2", "N3", "REM")
SUMMARY_NAMES = tuple([f"relative_power_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
                      + [f"segment_std_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
                      + [f"segment_max_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
                      + ["normalized_line_length_squashed"])


@dataclass(frozen=True)
class ModelConfig:
    fs: int = 100
    samples: int = 3000
    scales: tuple[int, ...] = (100, 200, 400)
    stride: int = 50
    prototypes_per_scale: int = 16
    embedding_dim: int = 32
    radius: int = 10
    neural_fraction_cap: float = 0.5
    use_summary: bool = True
    use_amplitude: bool = True
    summary_only: bool = False
    use_crf: bool = True
    raw_control: bool = False

    def __post_init__(self):
        if self.fs != 100 or self.samples != 3000:
            raise ValueError("v1 requires 30-s, 100-Hz single-channel epochs; no implicit resampling")
        if not self.scales or any(s < 50 or s > self.samples for s in self.scales):
            raise ValueError("invalid waveform scales")
        if len(set(self.scales)) != len(self.scales) or self.stride < 1:
            raise ValueError("scales must be unique and stride positive")
        if self.prototypes_per_scale < 1 or self.embedding_dim < 4 or self.radius < 0:
            raise ValueError("invalid model dimensions")
        if not 0 <= self.neural_fraction_cap <= 0.5:
            raise ValueError("neural metric may contribute at most half of the dissimilarity")
        if self.summary_only and not self.use_summary:
            raise ValueError("summary_only requires use_summary")
        if self.raw_control and self.radius not in (0, 10):
            raise ValueError("raw context control is implemented for radius 0 or 10")

    def dictionary(self):
        return asdict(self)


def normalize_wave(x: Tensor) -> tuple[Tensor, Tensor]:
    x = x.float()
    centered = x - x.mean(-1, keepdim=True)
    energy = centered.square().mean(-1, keepdim=True)
    rms = energy.clamp_min(1e-12).sqrt()
    return centered / rms, energy[..., 0] > 1e-12


def band_distribution(x: Tensor, fs: int = 100) -> Tensor:
    """Hann-windowed relative band energy. Flat input has a uniform distribution."""
    z, _ = normalize_wave(x)
    taper = torch.hann_window(z.shape[-1], periodic=False, device=z.device)
    power = torch.fft.rfft(z * taper, dim=-1).abs().square()
    freq = torch.fft.rfftfreq(z.shape[-1], 1.0 / fs, device=z.device)
    bands = torch.stack([power[..., (freq >= a) & (freq < b)].sum(-1) for a, b in BANDS], -1)
    bands = bands + 1e-8
    return bands / bands.sum(-1, keepdim=True)


def observed_parts(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Local shape, sqrt band distribution, RMS envelope, valid flag."""
    z, valid = normalize_wave(x)
    shape = F.normalize(z, dim=-1, eps=1e-6)
    spectral = band_distribution(x).sqrt()
    flat = z.reshape(-1, 1, z.shape[-1])
    env = F.adaptive_avg_pool1d(flat.square(), 8).clamp_min(1e-12).sqrt()
    env = F.normalize(env.reshape(*z.shape[:-1], 8), dim=-1, eps=1e-6)
    return shape, spectral, env, valid


def epoch_summary(x: Tensor) -> Tensor:
    """16 explicit bounded features, not a learned or hidden bypass."""
    if x.shape[-1] != 3000:
        raise ValueError("epoch_summary requires 3000 samples")
    z, _ = normalize_wave(x)
    full = band_distribution(z)
    segments = band_distribution(z.reshape(*z.shape[:-1], 6, 500))
    line = z.diff(dim=-1).abs().mean(-1, keepdim=True)
    return torch.cat((full, segments.std(-2, unbiased=False), segments.max(-2).values,
                      line / (1.0 + line)), -1)


class TemporalMean(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.mean(-1)


class LocalEncoder(nn.Module):
    """Each batch item is one window. No operation sees another window/epoch."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            TemporalMean(), nn.Linear(32, dim))

    def forward(self, x: Tensor) -> Tensor:
        z, _ = normalize_wave(x)
        return F.normalize(self.net(z[:, None]), dim=-1, eps=1e-6)


class WaveformBank(nn.Module):
    def __init__(self, waves: Tensor, dim: int, stride: int, neural_cap: float):
        super().__init__()
        if waves.ndim != 2 or not torch.isfinite(waves).all():
            raise ValueError("anchor bank requires finite [K, samples] waveforms")
        if not normalize_wave(waves)[1].all():
            raise ValueError("flat anchor waveforms are not allowed")
        self.register_buffer("waveforms", waves.detach().float().clone())
        self.encoder = LocalEncoder(dim)
        k = len(waves)
        self.observed_logits = nn.Parameter(torch.zeros(k, 3))
        self.neural_logits = nn.Parameter(torch.zeros(k))
        self.temperature_logits = nn.Parameter(torch.zeros(k))
        self.stride, self.neural_cap = stride, neural_cap

    def forward(self, epochs: Tensor, detail: bool = False):
        windows = epochs.unfold(-1, self.waveforms.shape[-1], self.stride)
        n, p, length = windows.shape
        flat = windows.reshape(-1, length)
        qs, qb, qe, qvalid = observed_parts(flat)
        ps, pb, pe, _ = observed_parts(self.waveforms)
        dw = (1 - qs @ ps.T).clamp(0, 2) / 2
        db = ((qb[:, None] - pb[None]).square().sum(-1) / 2).clamp(0, 1)
        de = (1 - qe @ pe.T).clamp(0, 2) / 2
        components = torch.stack((dw, db, de), -1)
        observed_weights = 0.1 + 0.7 * self.observed_logits.softmax(-1)
        obs = (components * observed_weights[None]).sum(-1)
        neural = ((1 - self.encoder(flat) @ self.encoder(self.waveforms).T).clamp(0, 2) / 2
                  if self.neural_cap else torch.zeros_like(obs))
        mix = self.neural_cap * self.neural_logits.sigmoid()
        distance = (1 - mix) * obs + mix * neural
        inverse_temp = 2 + 18 * self.temperature_logits.sigmoid()
        scores = torch.exp(-distance * inverse_temp) * qvalid[:, None]
        scores = scores.reshape(n, p, -1)
        peak, locations = scores.max(1)
        features = torch.cat((peak, scores.mean(1)), -1)
        if not detail:
            return features
        return features, {"scores": scores, "peak_window": locations,
                          "observed_distance": obs.reshape(n, p, -1),
                          "neural_distance": neural.reshape(n, p, -1),
                          "distance_components": components.reshape(n, p, -1, 3),
                          "neural_fraction": mix, "inverse_temperature": inverse_temp,
                          "observed_weights": observed_weights}


class AdditiveTemporal(nn.Module):
    """Exact per-feature/per-lag score contributions, with nonlinear univariate bases."""
    def __init__(self, features: int, radius: int):
        super().__init__()
        self.radius = radius
        self.weight = nn.Parameter(torch.empty(5, features, 3, 2 * radius + 1))
        nn.init.normal_(self.weight, std=0.01)
        self.bias = nn.Parameter(torch.zeros(5))

    @staticmethod
    def basis(x: Tensor) -> Tensor:
        return torch.stack((x, x.square(), F.relu(x - 0.5)), -1)

    def forward(self, x: Tensor) -> Tensor:
        b, t, f = x.shape
        basis = self.basis(x).reshape(b, t, f * 3).transpose(1, 2)
        return F.conv1d(basis, self.weight.reshape(5, f * 3, -1), self.bias,
                        padding=self.radius).transpose(1, 2)

    def explain_at(self, x: Tensor, epoch: int) -> Tensor:
        """Return [class, source lag, feature], without bias, for ONE sequence."""
        if x.ndim != 2 or not 0 <= epoch < len(x):
            raise ValueError("expected [T,F] and a valid epoch")
        parts = []
        for ki, lag in enumerate(range(-self.radius, self.radius + 1)):
            j = epoch + lag
            if 0 <= j < len(x):
                parts.append((self.weight[..., ki] * self.basis(x[j])[None]).sum(-1))
            else:
                parts.append(torch.zeros_like(self.weight[:, :, 0, ki]))
        return torch.stack(parts, 1)


class RawTemporal(nn.Module):
    """Non-interpretable capacity control, not called a literature/SOTA baseline."""
    def __init__(self, features: int, radius: int):
        super().__init__()
        self.radius = radius
        self.project = nn.Linear(features, 64)
        self.conv1 = nn.Conv1d(64, 64, 5 if radius else 1, padding=2 if radius else 0)
        self.conv2 = nn.Conv1d(64, 64, 5 if radius else 1, dilation=4 if radius else 1,
                               padding=8 if radius else 0)
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.classifier = nn.Linear(64, 5)

    def forward(self, x: Tensor):
        z = self.project(x)
        z = z + F.gelu(self.norm1(self.conv1(z.transpose(1, 2)).transpose(1, 2)))
        z = z + F.gelu(self.norm2(self.conv2(z.transpose(1, 2)).transpose(1, 2)))
        return self.classifier(z)


class LinearCRF(nn.Module):
    """Transitions[from, to]. Right-padded sequences only. All recurrences use FP32."""
    def __init__(self, classes: int = 5):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(classes, classes))
        self.start = nn.Parameter(torch.zeros(classes))
        self.end = nn.Parameter(torch.zeros(classes))

    def _check(self, emissions: Tensor, valid: Tensor):
        if emissions.ndim != 3 or valid.shape != emissions.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("CRF requires [B,T,C] emissions and boolean [B,T] mask")
        if emissions.shape[-1] != len(self.start) or emissions.shape[1] == 0:
            raise ValueError("empty sequence or incorrect number of classes")
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("CRF mask must be a nonempty contiguous prefix")

    def log_partition(self, emissions: Tensor, valid: Tensor) -> Tensor:
        self._check(emissions, valid)
        e = emissions.float()
        alpha = self.start + e[:, 0]
        for t in range(1, e.shape[1]):
            updated = torch.logsumexp(alpha[:, :, None] + self.transitions[None], dim=1) + e[:, t]
            alpha = torch.where(valid[:, t, None], updated, alpha)
        return torch.logsumexp(alpha + self.end, -1)

    def path_score(self, emissions: Tensor, y: Tensor, valid: Tensor) -> Tensor:
        self._check(emissions, valid)
        if y.shape != valid.shape or ((y[valid] < 0) | (y[valid] >= len(self.start))).any():
            raise ValueError("invalid class labels")
        y = y.masked_fill(~valid, 0)
        e = emissions.float()
        score = self.start[y[:, 0]] + e[:, 0].gather(1, y[:, :1])[:, 0]
        for t in range(1, e.shape[1]):
            term = self.transitions[y[:, t-1], y[:, t]] + e[:, t].gather(1, y[:, t:t+1])[:, 0]
            score = score + torch.where(valid[:, t], term, torch.zeros_like(term))
        last = y.gather(1, (valid.sum(1) - 1)[:, None])[:, 0]
        return score + self.end[last]

    def loss(self, emissions: Tensor, y: Tensor, valid: Tensor) -> Tensor:
        return (self.log_partition(emissions, valid) - self.path_score(emissions, y, valid)).sum() / valid.sum()

    @torch.no_grad()
    def decode(self, emissions: Tensor, valid: Tensor) -> Tensor:
        self._check(emissions, valid)
        result = torch.full(valid.shape, -1, dtype=torch.long, device=emissions.device)
        for b, length in enumerate(valid.sum(1).tolist()):
            alpha = self.start + emissions[b, 0].float()
            pointers = []
            for t in range(1, length):
                best, back = (alpha[:, None] + self.transitions).max(0)
                alpha = best + emissions[b, t].float()
                pointers.append(back)
            state = int((alpha + self.end).argmax())
            result[b, length - 1] = state
            for t in range(length - 2, -1, -1):
                state = int(pointers[t][state])
                result[b, t] = state
        return result

    @torch.no_grad()
    def marginals(self, emissions: Tensor) -> Tensor:
        """Exact marginals for an unpadded single recording segment [T,C]."""
        if emissions.ndim != 2 or not len(emissions):
            raise ValueError("expected nonempty [T,C]")
        e = emissions.float()
        alphas = [self.start + e[0]]
        for t in range(1, len(e)):
            alphas.append(torch.logsumexp(alphas[-1][:, None] + self.transitions, 0) + e[t])
        betas = [self.end]
        for t in range(len(e)-2, -1, -1):
            betas.append(torch.logsumexp(self.transitions + e[t+1][None] + betas[-1][None], 1))
        return (torch.stack(alphas) + torch.stack(list(reversed(betas)))).softmax(-1)


class EvidenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig, anchors: list[Tensor], amplitude_stats: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.cfg = cfg
        if len(amplitude_stats) != 2 or not all(math.isfinite(v) for v in amplitude_stats) or amplitude_stats[1] <= 0:
            raise ValueError("invalid training amplitude statistics")
        self.register_buffer("amplitude_stats", torch.tensor(amplitude_stats, dtype=torch.float32))
        if len(anchors) != len(cfg.scales):
            raise ValueError("one anchor bank per scale is required")
        for wave, length in zip(anchors, cfg.scales):
            if wave.shape != (cfg.prototypes_per_scale, length):
                raise ValueError("anchor shape differs from configuration")
        self.banks = nn.ModuleList([WaveformBank(w, cfg.embedding_dim, cfg.stride, cfg.neural_fraction_cap)
                                    for w in anchors])
        self.feature_names: list[str] = []
        if cfg.raw_control:
            for scale in cfg.scales:
                self.feature_names += [f"raw_{scale}_{mode}_{d}" for mode in ("mean", "max")
                                       for d in range(cfg.embedding_dim)]
        elif not cfg.summary_only:
            for scale in cfg.scales:
                self.feature_names += [f"s{scale}_p{k:02d}_{mode}" for mode in ("peak", "mean")
                                       for k in range(cfg.prototypes_per_scale)]
        if cfg.use_summary:
            self.feature_names += list(SUMMARY_NAMES)
            if cfg.use_amplitude:
                self.feature_names.append("log_rms_train_scaled_sigmoid")
        cls = RawTemporal if cfg.raw_control else AdditiveTemporal
        self.temporal = cls(len(self.feature_names), cfg.radius)
        self.crf: Optional[LinearCRF] = LinearCRF() if cfg.use_crf else None
        # Do not optimize metric parameters which an ablation does not use.
        for bank in self.banks:
            if cfg.summary_only:
                bank.requires_grad_(False)
            elif cfg.raw_control:
                bank.observed_logits.requires_grad_(False)
                bank.neural_logits.requires_grad_(False)
                bank.temperature_logits.requires_grad_(False)
            elif cfg.neural_fraction_cap == 0:
                bank.encoder.requires_grad_(False)
                bank.neural_logits.requires_grad_(False)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, self.cfg.samples):
            raise ValueError("expected [N,1,3000]")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite EEG")
        epochs = x[:, 0].float()
        values = []
        if self.cfg.raw_control:
            for bank in self.banks:
                windows = epochs.unfold(-1, bank.waveforms.shape[-1], self.cfg.stride)
                n, p, length = windows.shape
                embed = bank.encoder(windows.reshape(-1, length)).reshape(n, p, -1)
                values += [embed.mean(1), embed.max(1).values]
        elif not self.cfg.summary_only:
            values += [bank(epochs) for bank in self.banks]
        if self.cfg.use_summary:
            values.append(epoch_summary(epochs))
            if self.cfg.use_amplitude:
                rms = (epochs-epochs.mean(-1, keepdim=True)).square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                values.append(((rms.log()-self.amplitude_stats[0])/self.amplitude_stats[1]).sigmoid())
        return torch.cat(values, -1)

    def emissions(self, features: Tensor) -> Tensor:
        return self.temporal(features)

    def loss(self, emissions: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
        if self.crf is not None:
            return self.crf.loss(emissions, labels, valid)
        return F.cross_entropy(emissions[valid].float(), labels[valid])

    @torch.no_grad()
    def predict(self, emissions: Tensor) -> tuple[Tensor, Tensor]:
        if self.crf is None:
            return emissions.argmax(-1), emissions.float().softmax(-1)
        valid = torch.ones((1, len(emissions)), dtype=torch.bool, device=emissions.device)
        return self.crf.decode(emissions[None], valid)[0], self.crf.marginals(emissions)

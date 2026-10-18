"""Compact spectral event encoder with exact, signed additive logits.

No prototypes, MoE, temporal context, fitted statistics, or dataset imports.
Availability is deliberately a fixed scalar per event, broadcast over stages:
it cannot learn a label bypass and is NOT a calibrated pairwise information
estimate. The experiment must determine whether this restriction is useful.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .evidence import Evidence, wave_digest

BANDS = ((0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30))
BACKGROUND_NAMES = ("delta_fraction", "theta_fraction", "alpha_fraction", "sigma_fraction",
                    "beta_fraction", "log1p_rms_uV_over_10", "mean_uV_over_10")
MODES = {"full", "q_one", "background_only", "q_only", "fixed_event_score", "fixed_patch"}


@dataclass(frozen=True)
class CoveConfig:
    samples: int = 3000
    fs: int = 100
    window: int = 200
    stride: int = 100
    event_budget: int = 4
    concentration_floor: float = 0.35
    noise_scale_uV: float = 5.0
    min_rms_uV: float = 1e-5
    event_scale: float = 8.0

    def __post_init__(self):
        if self.samples != 3000 or self.fs != 100:
            raise ValueError("requires calibrated Fpz-Cz, 100 Hz, 3000 samples")
        if (type(self.window) is not int or type(self.stride) is not int
                or not 100 <= self.window <= 500 or not 1 <= self.stride <= self.window
                or (3000 - self.window) % self.stride
                or type(self.event_budget) is not int or not 1 <= self.event_budget <= 8):
            raise ValueError("invalid local window geometry/budget")
        if (not 0 <= self.concentration_floor < 1
                or any(not math.isfinite(v) or v <= 0 for v in
                       (self.noise_scale_uV, self.min_rms_uV, self.event_scale))):
            raise ValueError("invalid availability/evidence constants")


def spectral_features(wave: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Physical amplitude plus Hann spectral fractions of total AC power."""
    centered = wave - wave.mean(-1, keepdim=True)
    rms = centered.square().mean(-1).clamp_min(1e-20).sqrt()
    taper = torch.hann_window(wave.shape[-1], periodic=False, device=wave.device)
    power = torch.fft.rfft(centered * taper, dim=-1).abs().square()
    freq = torch.fft.rfftfreq(wave.shape[-1], 0.01, device=wave.device)
    denominator = power[..., 1:].sum(-1).clamp_min(1e-12)
    fractions = torch.stack([power[..., (freq >= lo) & (freq < hi)].sum(-1)
                             for lo, hi in BANDS], -1) / denominator[..., None]
    return fractions, rms, wave.mean(-1)


class CoveModel(nn.Module):
    def __init__(self, cfg: CoveConfig = CoveConfig(), *, mean_uV: float = 0., std_uV: float = 1.):
        super().__init__()
        if not math.isfinite(mean_uV) or not math.isfinite(std_uV) or std_uV <= 0:
            raise ValueError("finite frozen normalization with positive std required")
        self.cfg = cfg
        self.register_buffer("mean_uV", torch.tensor(float(mean_uV)))
        self.register_buffer("std_uV", torch.tensor(float(std_uV)))
        # The local encoder is analytic; only these transparent linear heads fit.
        self.background_head = nn.Linear(7, 5)
        self.event_head = nn.Linear(6, 5)
        nn.init.zeros_(self.background_head.bias)
        nn.init.zeros_(self.event_head.bias)

    def physical_uV(self, x: Tensor) -> Tensor:
        if (x.ndim != 3 or x.shape[1:] != (1, 3000) or not len(x)
                or x.dtype != torch.float32 or not torch.isfinite(x).all()):
            raise ValueError("expected nonempty finite FP32 [B,1,3000]")
        result = x * self.std_uV + self.mean_uV
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite physical conversion")
        return result

    def forward(self, x: Tensor, *, return_evidence: bool = False, mode: str = "full"):
        if mode not in MODES:
            raise ValueError("unknown mechanism control")
        wave = self.physical_uV(x)[:, 0]
        cfg = self.cfg
        local = wave.unfold(-1, cfg.window, cfg.stride)
        bands, rms, _ = spectral_features(local)
        features = torch.cat((bands, torch.log1p(rms[..., None] / 10)), -1)
        # Ranking has no class weights. Stable sort breaks ties by physical time.
        scores = rms * bands.sum(-1).sqrt()
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        if mode == "fixed_patch":
            order = torch.arange(local.shape[1], device=x.device).expand(len(x), -1)
        selected = torch.zeros((len(x), cfg.event_budget), dtype=torch.long, device=x.device)
        valid = torch.zeros_like(selected, dtype=torch.bool)
        # Nonoverlap suppression prevents selecting the same samples twice.
        for b in range(len(x)):
            starts = []
            for j in order[b].tolist():
                start = j * cfg.stride
                if rms[b, j] <= cfg.min_rms_uV or any(abs(start - s) < cfg.window for s in starts):
                    continue
                k = len(starts)
                selected[b, k] = j
                valid[b, k] = True
                starts.append(start)
                if len(starts) == cfg.event_budget:
                    break
        gather = selected[..., None].expand(-1, -1, 6)
        picked = features.gather(1, gather)
        peak = bands.max(-1).values.gather(1, selected)
        picked_rms = rms.gather(1, selected)
        q = ((peak - cfg.concentration_floor) / (1 - cfg.concentration_floor)).clamp(0, 1)
        q = q * picked_rms / (picked_rms + cfg.noise_scale_uV)
        q = q[..., None].expand(-1, -1, 5)
        if mode == "q_one":
            q = torch.ones_like(q)
        raw_score = self.event_head(picked)
        if mode in {"q_only", "fixed_event_score"}:
            raw_score = self.event_head.bias.expand_as(raw_score)
        signed = raw_score.tanh() * (cfg.event_scale / cfg.event_budget)
        signed = signed * valid[..., None]
        q = q * valid[..., None]
        contributions = q * signed
        if mode == "background_only":
            contributions = torch.zeros_like(contributions)
            signed = torch.zeros_like(signed)
        full_bands, full_rms, mean = spectral_features(wave)
        background_features = torch.cat((full_bands, torch.log1p(full_rms[:, None] / 10), mean[:, None] / 10), -1)
        background_parts = background_features[:, :, None] * self.background_head.weight.T[None]
        background_bias = self.background_head.bias
        if mode == "q_only":
            background_parts = torch.zeros_like(background_parts)
            background_bias = torch.zeros_like(background_bias)
        background = background_parts.sum(1) + background_bias
        logits = background + contributions.sum(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite MIST-COVE logits")
        if not return_evidence:
            return logits
        intervals = torch.stack((selected * cfg.stride, selected * cfg.stride + cfg.window), -1)
        intervals = torch.where(valid[..., None], intervals, torch.zeros_like(intervals))
        return Evidence(logits, background, background_features, background_parts,
                        background_bias, intervals, valid,
                        scores.gather(1, selected) * valid, signed, q, contributions,
                        {"kind": "fixed_spectral_concentration_times_amplitude_v1",
                         "stage_shared": True, "calibrated_probability": False,
                         "peak_band_fraction": peak, "rms_uV": picked_rms,
                         "concentration_floor": cfg.concentration_floor,
                         "noise_scale_uV": cfg.noise_scale_uV},
                        "physical_time_first" if mode == "fixed_patch" else "band_rms_descending_nonoverlap_time_ties",
                        tuple(wave_digest(epoch) for epoch in wave[:, None]))

"""Raw-exemplar evidence bottleneck with an exact additive local decision ledger.

Exemplars are *real*, immutable training crops, re-encoded by the current encoder.
No decoder-generated waveform is presented as evidence; no free latent prototype
is silently projected onto an unrelated waveform after training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Night, candidates

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
DESCRIPTORS = ("log_relative_delta", "log_relative_theta", "log_relative_alpha",
               "log_relative_sigma", "log_relative_beta", "log_rms_input_units",
               "normalized_line_length", "zero_crossing_fraction")


@dataclass(frozen=True)
class EvidenceConfig:
    scales_samples: tuple[int, ...] = (100, 200, 400)
    anchors_per_class: int = 4
    embedding_dim: int = 64
    descriptor_weight: float = 0.5
    temperature: float = 0.2
    pool_per_subject: int = 256
    arm: str = "evidence"  # evidence, descriptor, dense
    aux_morph_weight: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "scales_samples", tuple(self.scales_samples))
        if not self.scales_samples or len(set(self.scales_samples)) != len(self.scales_samples):
            raise ValueError("Scales must be unique and nonempty")
        if any(type(x) is not int or not 100 <= x <= 3000 for x in self.scales_samples):
            raise ValueError("Scale lengths must be integer samples in [100,3000]")
        if self.anchors_per_class < 1 or self.embedding_dim < 4 or self.pool_per_subject < 5:
            raise ValueError("Invalid model size")
        if not 0 <= self.descriptor_weight <= 1 or self.temperature <= 0:
            raise ValueError("Invalid distance mixture/temperature")
        if self.arm not in {"evidence", "descriptor", "dense"} or self.aux_morph_weight < 0:
            raise ValueError("Invalid arm/loss")
        if self.arm == "descriptor" and self.aux_morph_weight != 0:
            raise ValueError("The descriptor-only control has no trainable encoder for an auxiliary loss")

    @property
    def n_anchors(self):
        return 5 * self.anchors_per_class

    def to_dict(self):
        return asdict(self)


def normalize_crop(x: torch.Tensor) -> torch.Tensor:
    x = x.float() - x.float().mean(-1, keepdim=True)
    return x / x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def morphology(x: torch.Tensor, fs: int = 100) -> torch.Tensor:
    """Eight directly measured quantities; not a spindle/K-complex detector.

    Relative powers use a Hann periodogram on [0.5,30) Hz; log RMS is in
    supplied NPZ units, not assumed microvolts. Short windows have limited
    frequency resolution. Constant inputs remain finite.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        y = x.float().squeeze(-2)
        y = y - y.mean(-1, keepdim=True)
        rms = y.square().mean(-1).sqrt()
        normalized = y / rms.unsqueeze(-1).clamp_min(1e-6)
        power = torch.fft.rfft(normalized * torch.hann_window(y.shape[-1], device=y.device)).abs().square()
        freq = torch.fft.rfftfreq(y.shape[-1], 1.0 / fs).to(y.device)
        bands = torch.stack([power[..., (freq >= lo) & (freq < hi)].sum(-1) for lo, hi in BANDS], -1)
        rel = (bands + 1e-8) / (bands.sum(-1, keepdim=True) + 5e-8)
        ll = normalized.diff(dim=-1).abs().mean(-1)
        zc = ((normalized[..., 1:] * normalized[..., :-1]) < 0).float().mean(-1)
        return torch.cat([rel.log(), rms.clamp_min(1e-8).log().unsqueeze(-1),
                          ll.unsqueeze(-1), zc.unsqueeze(-1)], -1)


class DeterministicAdaptivePool1d(nn.Module):
    """Adaptive mean bins using ordinary reductions, avoiding atomic pool backward.

    Bin boundaries match PyTorch adaptive average pooling. This avoids relying on
    CUDA adaptive-pooling backward kernels that can reject deterministic mode.
    """
    def __init__(self, bins: int = 4):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        length = x.shape[-1]
        return torch.stack([x[..., (i * length) // self.bins:
                                      ((i + 1) * length + self.bins - 1) // self.bins].mean(-1)
                            for i in range(self.bins)], -1)


class CropEncoder(nn.Module):
    """Independent-crop CNN. GroupNorm avoids cross-crop BatchNorm leakage/state drift."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 16, 11, 2, 5), nn.GroupNorm(4, 16), nn.SiLU(),
                                 nn.Conv1d(16, 32, 9, 2, 4), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv1d(32, 48, 7, 2, 3), nn.GroupNorm(8, 48), nn.SiLU(),
                                 DeterministicAdaptivePool1d(4), nn.Flatten(), nn.Linear(192, dim))

    def forward(self, x):
        return F.normalize(self.net(normalize_crop(x)).float(), dim=-1, eps=1e-6)


def build_bank(nights: list[Night], cfg: EvidenceConfig, seed: int) -> dict:
    """Farthest-point selection in measured morphology, stratified by epoch stage.

    Candidate pool and standardization use train records only. Per-class labels
    organize sampling; they do not assign clinical event identities to the crops.
    """
    scales = []
    for length in cfg.scales_samples:
        raw, labels, refs = candidates(nights, length, cfg.pool_per_subject, seed)
        with torch.no_grad():
            d = morphology(raw)
            center = d.mean(0)
            scale = d.std(0, unbiased=False).clamp_min(0.1)
            norm = (d - center) / scale
        selected = []
        for c in range(5):
            available = np.flatnonzero(labels == c).tolist()
            if len(available) < cfg.anchors_per_class:
                raise ValueError(f"Not enough train candidates for stage {c}; enlarge pool or supply more train data")
            chosen = []
            for k in range(cfg.anchors_per_class):
                counts = {sid: sum(refs[j]["subject"] == sid for j in chosen)
                          for sid in {refs[j]["subject"] for j in available}}
                smallest = min(counts.values())
                eligible = [j for j in available if counts[refs[j]["subject"]] == smallest]
                if k == 0:
                    score = -(norm[eligible] - norm[available].mean(0)).square().mean(-1)
                else:
                    score = torch.cdist(norm[eligible], norm[chosen]).amin(-1)
                j = eligible[int(score.argmax())]
                chosen.append(j)
                available.remove(j)
            selected.extend(chosen)
        scales.append({"length": length, "raw": raw[selected], "center": center, "scale": scale,
                       "refs": [refs[j] for j in selected], "candidate_count": len(raw)})
    return {"schema": 1, "train_subjects": sorted({n.subject for n in nights}), "scales": scales}


class EvidenceNet(nn.Module):
    def __init__(self, cfg: EvidenceConfig, bank: dict):
        super().__init__()
        self.cfg = cfg
        if bank.get("schema") != 1 or len(bank["scales"]) != len(cfg.scales_samples):
            raise ValueError("Bank schema/config mismatch")
        self.refs = []
        self.encoder = CropEncoder(cfg.embedding_dim) if cfg.arm != "descriptor" else nn.Identity()
        self.morph_head = nn.Linear(cfg.embedding_dim, len(DESCRIPTORS)) if cfg.aux_morph_weight and cfg.arm != "descriptor" else None
        self.feature_names = []
        for si, (length, b) in enumerate(zip(cfg.scales_samples, bank["scales"])):
            if b["length"] != length or tuple(b["raw"].shape) != (cfg.n_anchors, 1, length):
                raise ValueError("Bank shape mismatch")
            self.register_buffer(f"anchors_{si}", b["raw"].float().clone())
            self.register_buffer(f"center_{si}", b["center"].float().clone())
            self.register_buffer(f"scale_{si}", b["scale"].float().clone())
            self.refs.append(b["refs"])
            if cfg.arm == "evidence":
                for pool in ("max", "mean"):
                    self.feature_names.extend(f"s{si}.p{k}.{pool}" for k in range(cfg.n_anchors))
            elif cfg.arm == "dense":
                for pool in ("max", "mean"):
                    self.feature_names.extend(f"s{si}.latent{k}.{pool}" for k in range(cfg.embedding_dim))
            self.feature_names.extend(f"s{si}.{name}.mean" for name in DESCRIPTORS)
        self.classifier = nn.Linear(len(self.feature_names), 5)

    def forward(self, x: torch.Tensor, details: bool = False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, 3000):
            raise ValueError("EvidenceNet expects [B,1,3000]")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite input")
        parts, all_details, aux = [], [], []
        for si, length in enumerate(self.cfg.scales_samples):
            stride = length // 2
            starts = list(range(0, 3000 - length + 1, stride))
            if starts[-1] != 3000 - length:
                starts.append(3000 - length)
            crops = torch.stack([x[..., s:s + length] for s in starts], dim=1)
            b, p, _, _ = crops.shape
            flat = crops.reshape(b * p, 1, length)
            d_raw = morphology(flat).reshape(b, p, -1)
            center, scale = getattr(self, f"center_{si}"), getattr(self, f"scale_{si}")
            d = (d_raw - center) / scale
            z = None
            if self.cfg.arm != "descriptor":
                z = self.encoder(flat).reshape(b, p, -1)
                if self.cfg.aux_morph_weight:
                    aux.append(F.smooth_l1_loss(self.morph_head(z).float(), d.detach()))
            info = {"starts": starts, "length": length, "descriptors": d_raw}
            if self.cfg.arm == "evidence":
                anchor_raw = getattr(self, f"anchors_{si}")
                a = self.encoder(anchor_raw)  # live shared-encoder embedding, not a stale projection
                ad_raw = morphology(anchor_raw)
                ad = (ad_raw - center) / scale
                with torch.autocast(device_type=x.device.type, enabled=False):
                    latent_dist = (z.float().unsqueeze(2) - a.float()).square().sum(-1) / 4.0
                    measured_dist = 1.0 - torch.exp(-0.5 * (d.unsqueeze(2) - ad).square().mean(-1))
                    rho = self.cfg.descriptor_weight
                    distance = (1 - rho) * latent_dist + rho * measured_dist
                    sim = torch.exp(-distance / self.cfg.temperature)
                maximum, where = sim.max(1)
                parts.extend((maximum, sim.mean(1)))
                info.update({"similarity": sim, "argmax": where, "latent_distance": latent_dist,
                             "measured_distance": measured_dist, "anchor_descriptors": ad_raw})
            elif self.cfg.arm == "dense":
                parts.extend((z.max(1).values, z.mean(1)))
            parts.append(d.mean(1))
            if details:
                all_details.append(info)
        features = torch.cat(parts, -1)
        logits = self.classifier(features)
        aux_loss = torch.stack(aux).mean() if aux else logits.sum() * 0.0
        return {"logits": logits, "features": features, "aux_loss": aux_loss, "details": all_details}

    def margin_ledger(self, features: torch.Tensor, predicted: int, alternative: int):
        """Exact local c-vs-r logit decomposition, not a causal explanation."""
        if not (0 <= predicted < 5 and 0 <= alternative < 5) or predicted == alternative:
            raise ValueError("Choose distinct valid classes")
        w = self.classifier.weight[predicted] - self.classifier.weight[alternative]
        b = self.classifier.bias[predicted] - self.classifier.bias[alternative]
        contribution = features * w
        return contribution, b, contribution.sum(-1) + b

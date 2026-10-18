"""Same 75-parameter architecture, explicit q=1 default and fixed-witness path."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from mist_cove.evidence import Evidence, wave_digest
from mist_cove.model import CoveConfig, CoveModel, spectral_features

METHODS = {
    "q1": ("q_one", "hard"),
    "availability": ("full", "hard"),
    "background": ("background_only", "hard"),
    # Exact Phase I fixed_patch comparison includes its original availability.
    "fixed_patch": ("full", "fixed_patch"),
    "rounded_q1": ("q_one", "rounded"),
}


@dataclass(frozen=True)
class WitnessSet:
    intervals: Tensor
    valid: Tensor

    @classmethod
    def from_evidence(cls, evidence):
        return cls(evidence.event_intervals.detach().clone(), evidence.valid.detach().clone())


class WitnessModel(CoveModel):
    def __init__(self, cfg=CoveConfig(), *, decision_mode="q_one", selection_policy="hard",
                 rank_bin_uV=.5, mean_uV=0., std_uV=1.):
        super().__init__(cfg, mean_uV=mean_uV, std_uV=std_uV)
        if decision_mode not in {"q_one", "full", "background_only"}:
            raise ValueError("unknown Phase II decision mode")
        if selection_policy not in {"hard", "rounded", "fixed_patch"}:
            raise ValueError("unknown witness selection policy")
        if not math.isfinite(rank_bin_uV) or rank_bin_uV <= 0:
            raise ValueError("positive physical ranking bin required")
        self.decision_mode = decision_mode
        self.selection_policy = selection_policy
        self.rank_bin_uV = rank_bin_uV

    def select(self, scores, rms, *, rank_delta=None):
        ranking = scores
        if rank_delta is not None:
            if rank_delta.shape != scores.shape or not torch.isfinite(rank_delta).all():
                raise ValueError("ranking intervention must match all candidate scores")
            ranking = ranking + rank_delta
        if self.selection_policy == "rounded":
            ranking = torch.floor(ranking / self.rank_bin_uV + .5)
        order = torch.argsort(ranking, dim=1, descending=True, stable=True)
        if self.selection_policy == "fixed_patch":
            order = torch.arange(scores.shape[1], device=scores.device).expand(len(scores), -1)
        indices = torch.zeros((len(scores), self.cfg.event_budget), dtype=torch.long, device=scores.device)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        for b in range(len(scores)):
            starts = []
            for j in order[b].tolist():
                start = j * self.cfg.stride
                if rms[b, j] <= self.cfg.min_rms_uV or any(abs(start - old) < self.cfg.window for old in starts):
                    continue
                k = len(starts)
                indices[b, k] = j; valid[b, k] = True; starts.append(start)
                if len(starts) == self.cfg.event_budget:
                    break
        return indices, valid

    def fixed_indices(self, witnesses, batch, device):
        intervals, valid = witnesses.intervals, witnesses.valid
        if (intervals.shape != (batch, self.cfg.event_budget, 2) or valid.shape != intervals.shape[:2]
                or intervals.dtype != torch.long or valid.dtype != torch.bool
                or intervals.device != device or valid.device != device):
            raise ValueError("fixed witnesses must match model budget, batch, dtype and device")
        lo, hi = intervals.unbind(-1)
        good = (lo >= 0) & (hi <= 3000) & (hi - lo == self.cfg.window) & (lo % self.cfg.stride == 0)
        if not good[valid].all() or intervals[~valid].any():
            raise ValueError("invalid fixed half-open witness intervals")
        for b in range(batch):
            starts = lo[b, valid[b]].sort().values
            if (starts.diff() < self.cfg.window).any():
                raise ValueError("fixed witnesses may not overlap or duplicate")
        return lo // self.cfg.stride, valid

    def forward(self, x, *, return_evidence=False, witnesses=None, rank_delta=None):
        if witnesses is not None and rank_delta is not None:
            raise ValueError("fixed identities and ranking intervention are distinct estimands")
        wave = self.physical_uV(x)[:, 0]
        bands, rms, _ = spectral_features(wave.unfold(-1, self.cfg.window, self.cfg.stride))
        features = torch.cat((bands, torch.log1p(rms[..., None] / 10)), -1)
        scores = rms * bands.sum(-1).sqrt()
        if witnesses is None:
            indices, valid = self.select(scores, rms, rank_delta=rank_delta)
        else:
            indices, valid = self.fixed_indices(witnesses, len(x), x.device)
        picked = features.gather(1, indices[..., None].expand(-1, -1, 6))
        q = torch.ones((*indices.shape, 5), device=x.device)
        peak = bands.max(-1).values.gather(1, indices)
        picked_rms = rms.gather(1, indices)
        if self.decision_mode in {"full", "background_only"}:
            scalar = ((peak - self.cfg.concentration_floor) / (1 - self.cfg.concentration_floor)).clamp(0, 1)
            scalar = scalar * picked_rms / (picked_rms + self.cfg.noise_scale_uV)
            q = scalar[..., None].expand(-1, -1, 5)
        signed = self.event_head(picked).tanh() * (self.cfg.event_scale / self.cfg.event_budget)
        signed = signed * valid[..., None]
        q = q * valid[..., None]
        if self.decision_mode == "background_only":
            signed = torch.zeros_like(signed)
        contributions = q * signed
        whole, whole_rms, mean = spectral_features(wave)
        bg_features = torch.cat((whole, torch.log1p(whole_rms[:, None] / 10), mean[:, None] / 10), -1)
        bg_parts = bg_features[:, :, None] * self.background_head.weight.T[None]
        background = bg_parts.sum(1) + self.background_head.bias
        logits = background + contributions.sum(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite Phase II logits")
        if not return_evidence:
            return logits
        intervals = torch.stack((indices * self.cfg.stride, indices * self.cfg.stride + self.cfg.window), -1)
        intervals = torch.where(valid[..., None], intervals, torch.zeros_like(intervals))
        metadata = {"kind": "q_equals_one" if self.decision_mode == "q_one" else "phase1_fixed_availability",
                    "stage_shared": True, "calibrated_probability": False,
                    "rms_uV": picked_rms, "peak_band_fraction": peak,
                    "fixed_witness_internal_intervention": witnesses is not None}
        return Evidence(logits, background, bg_features, bg_parts, self.background_head.bias,
                        intervals, valid, scores.gather(1, indices) * valid, signed, q, contributions,
                        metadata, self.selection_policy + ("_fixed_identities" if witnesses is not None else ""),
                        tuple(wave_digest(epoch) for epoch in wave[:, None]))


def make_model(method, cfg=CoveConfig(), *, rank_bin_uV=.5, mean_uV=0., std_uV=1.):
    if method not in METHODS:
        raise ValueError("unknown versioned Phase II method")
    mode, policy = METHODS[method]
    return WitnessModel(cfg, decision_mode=mode, selection_policy=policy,
                        rank_bin_uV=rank_bin_uV, mean_uV=mean_uV, std_uV=std_uV)

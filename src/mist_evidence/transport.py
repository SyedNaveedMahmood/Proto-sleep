"""Morphology transport layer for MIST-Morph v2.

The transport model keeps the v1 source-waveform banks and constrained local
morphology distance, but replaces independent peak/mean prototype evidence with
an explicit per-window assignment plan. Every query window has a fixed source
mass. A semi-unbalanced entropic transport problem allocates that mass among
real training-waveform anchors and a background sink. Anchor target masses are
soft priors, not hard quotas, so an epoch is not forced to use every anchor.

This is a research mechanism, not a clinical event detector. Transport mass is
not called an event count because windows overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .model import AdditiveTemporal, EvidenceModel, ModelConfig, SUMMARY_NAMES, epoch_summary


@dataclass(frozen=True)
class TransportModelConfig(ModelConfig):
    """V2 transport hyperparameters.

    Source marginals are exact and uniform over query windows. Target marginals
    are KL-relaxed toward a prior over K real anchors plus one background sink.
    The background cost is expressed in the same bounded morphology-distance
    units as the v1 local comparison.
    """

    transport_epsilon: float = 0.08
    transport_target_rho: float = 0.20
    transport_iterations: int = 30
    transport_background_cost: float = 0.40
    transport_background_prior: float = 0.35
    transport_mass_floor: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.raw_control or self.summary_only:
            raise ValueError("TransportModelConfig is only for the morphology-transport model")
        if not (0.0 < self.transport_epsilon <= 1.0):
            raise ValueError("transport_epsilon must be in (0,1]")
        if not (0.0 < self.transport_target_rho <= 10.0):
            raise ValueError("transport_target_rho must be positive")
        if self.transport_iterations < 5:
            raise ValueError("transport_iterations must be >=5")
        if not (0.0 <= self.transport_background_cost <= 2.0):
            raise ValueError("transport_background_cost must be in [0,2]")
        if not (0.0 < self.transport_background_prior < 1.0):
            raise ValueError("transport_background_prior must be in (0,1)")
        if self.transport_mass_floor <= 0:
            raise ValueError("transport_mass_floor must be positive")


def semi_unbalanced_transport(
    morphology_cost: Tensor,
    valid_windows: Tensor,
    *,
    epsilon: float,
    target_rho: float,
    iterations: int,
    background_cost: float,
    background_prior: float,
) -> Tensor:
    """Return [N,P,K+1] differentiable transport plan.

    Rows have an exact uniform source marginal (up to numerical precision).
    Columns are unbalanced: their masses are softly regularized toward a target
    prior. The final column is a background sink. Invalid/flat windows are made
    cheap only for the background sink.

    This is a semi-unbalanced Sinkhorn scaling. The target relaxation exponent
    is rho/(rho+epsilon), while the source update uses exponent 1 so that every
    query window allocates exactly its source mass somewhere.
    """

    if morphology_cost.ndim != 3:
        raise ValueError("morphology_cost must be [N,P,K]")
    if valid_windows.shape != morphology_cost.shape[:2] or valid_windows.dtype != torch.bool:
        raise ValueError("valid_windows must be boolean [N,P]")
    if not torch.isfinite(morphology_cost).all() or (morphology_cost < 0).any():
        raise ValueError("transport costs must be finite and nonnegative")
    n, p, k = morphology_cost.shape
    if min(n, p, k) < 1:
        raise ValueError("transport dimensions must be nonempty")

    dtype, device = morphology_cost.dtype, morphology_cost.device
    # Flat windows cannot be explained by a morphology anchor. They are routed
    # toward the background by making morphology assignment deliberately costly.
    invalid_anchor_cost = torch.full_like(morphology_cost, float(background_cost) + 1.0)
    anchor_cost = torch.where(valid_windows[..., None], morphology_cost, invalid_anchor_cost)
    bg = torch.full((n, p, 1), float(background_cost), dtype=dtype, device=device)
    bg = torch.where(valid_windows[..., None], bg, torch.zeros_like(bg))
    cost = torch.cat((anchor_cost, bg), dim=-1)

    source = torch.full((n, p), 1.0 / float(p), dtype=dtype, device=device)
    target = torch.full((k + 1,), (1.0 - float(background_prior)) / float(k), dtype=dtype, device=device)
    target[-1] = float(background_prior)

    kernel = torch.exp(-cost / float(epsilon)).clamp_min(1e-12)
    v = torch.ones((n, k + 1), dtype=dtype, device=device)
    tau_target = float(target_rho) / (float(target_rho) + float(epsilon))

    for _ in range(int(iterations)):
        kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        u = source / kv
        ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = (target[None] / ktu).clamp_min(1e-12).pow(tau_target)

    # One final exact-source update. This makes the interpretation "each window
    # allocates 1/P mass" a checked numerical property rather than an intention.
    kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    u = source / kv
    plan = u[..., None] * kernel * v[:, None, :]
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("non-finite transport plan")
    return plan


def _transport_cost_from_bank_details(info: dict) -> Tensor:
    components = info["distance_components"]
    observed_weights = info["observed_weights"]
    neural = info["neural_distance"]
    mix = info["neural_fraction"]
    observed = (components * observed_weights[None, None]).sum(-1)
    return (1.0 - mix[None, None]) * observed + mix[None, None] * neural


class TransportEvidenceModel(EvidenceModel):
    """MIST-Morph v2: source-waveform evidence coupled by transport.

    Per scale and real anchor the classifier receives:
      * transported mass,
      * transport-weighted morphology similarity,
      * transported normalized position moment.
    It also receives one explicit background-mass feature per scale and the same
    named summary/amplitude evidence as v1. There is no free epoch embedding.
    """

    def __init__(self, cfg: TransportModelConfig, anchors: list[Tensor], amplitude_stats=(0.0, 1.0)):
        if not isinstance(cfg, TransportModelConfig):
            raise TypeError("TransportEvidenceModel requires TransportModelConfig")
        super().__init__(cfg, anchors, amplitude_stats)
        self.feature_names = []
        for scale in cfg.scales:
            self.feature_names += [f"ot_s{scale}_p{k:02d}_mass" for k in range(cfg.prototypes_per_scale)]
            self.feature_names += [f"ot_s{scale}_p{k:02d}_affinity" for k in range(cfg.prototypes_per_scale)]
            self.feature_names += [f"ot_s{scale}_p{k:02d}_position_moment" for k in range(cfg.prototypes_per_scale)]
            self.feature_names.append(f"ot_s{scale}_background_mass")
        if cfg.use_summary:
            self.feature_names += list(SUMMARY_NAMES)
            if cfg.use_amplitude:
                self.feature_names.append("log_rms_train_scaled_sigmoid")
        # Replace the v1 peak/mean temporal head. The CRF constructed by the
        # parent is retained, but its emissions now come only from transport and
        # named summary evidence.
        self.temporal = AdditiveTemporal(len(self.feature_names), cfg.radius)

    def _scale_transport(self, bank, epochs: Tensor, detail: bool):
        _, info = bank(epochs, detail=True)
        scores = info["scores"]
        valid = scores.sum(-1) > 0
        cost = _transport_cost_from_bank_details(info)
        plan = semi_unbalanced_transport(
            cost,
            valid,
            epsilon=self.cfg.transport_epsilon,
            target_rho=self.cfg.transport_target_rho,
            iterations=self.cfg.transport_iterations,
            background_cost=self.cfg.transport_background_cost,
            background_prior=self.cfg.transport_background_prior,
        )
        k = bank.waveforms.shape[0]
        morphology = plan[..., :k]
        mass = morphology.sum(1)
        weighted = (morphology * scores).sum(1)
        affinity = weighted / mass.clamp_min(self.cfg.transport_mass_floor)
        affinity = torch.where(mass > self.cfg.transport_mass_floor, affinity, torch.zeros_like(affinity))
        p = morphology.shape[1]
        if p == 1:
            position = torch.zeros((1,), dtype=epochs.dtype, device=epochs.device)
        else:
            position = torch.linspace(0.0, 1.0, p, dtype=epochs.dtype, device=epochs.device)
        position_moment = (morphology * position[None, :, None]).sum(1)
        background = plan[..., -1].sum(1, keepdim=True)
        features = torch.cat((mass, affinity, position_moment, background), dim=-1)
        if not detail:
            return features, None
        return features, {
            **info,
            "transport_cost": cost,
            "transport_plan": plan,
            "transport_anchor_mass": mass,
            "transport_affinity": affinity,
            "transport_position_moment": position_moment,
            "transport_background_mass": background,
            "transport_row_mass": plan.sum(-1),
        }

    def encode_with_details(self, x: Tensor):
        if x.ndim != 3 or x.shape[1:] != (1, self.cfg.samples):
            raise ValueError("expected [N,1,3000]")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite EEG")
        epochs = x[:, 0].float()
        values, scale_details = [], []
        for bank in self.banks:
            feature, detail = self._scale_transport(bank, epochs, detail=True)
            values.append(feature)
            scale_details.append(detail)
        if self.cfg.use_summary:
            values.append(epoch_summary(epochs))
            if self.cfg.use_amplitude:
                rms = (epochs - epochs.mean(-1, keepdim=True)).square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                values.append(((rms.log() - self.amplitude_stats[0]) / self.amplitude_stats[1]).sigmoid())
        feature = torch.cat(values, -1)
        if not torch.isfinite(feature).all():
            raise FloatingPointError("non-finite transport evidence feature")
        return feature, scale_details

    def encode(self, x: Tensor) -> Tensor:
        feature, _ = self.encode_with_details(x)
        return feature

    @torch.no_grad()
    def transport_audit(self, x: Tensor) -> dict:
        """Small numerical audit used by reports/tests, not a performance metric."""
        _, details = self.encode_with_details(x)
        audits = []
        for scale, item in zip(self.cfg.scales, details):
            row = item["transport_row_mass"]
            expected = 1.0 / float(row.shape[1])
            audits.append({
                "scale_samples": int(scale),
                "max_source_marginal_error": float((row - expected).abs().max()),
                "mean_background_mass": float(item["transport_background_mass"].mean()),
                "total_morphology_mass_mean": float(item["transport_anchor_mass"].sum(-1).mean()),
            })
        return {"transport": audits}

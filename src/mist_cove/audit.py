"""Counterfactual model-use audit with same-epoch donors and artifact checks.

Recompute selection and background after intervention. Claimed contributions
are bookkeeping, not assumed equal to a total counterfactual logit change.
Neither donor can certify physiological validity. All arrays stay in memory.
"""
from dataclasses import dataclass
import torch
from torch import Tensor

from .model import CoveModel


@dataclass
class AuditResult:
    interval: tuple[int, int]
    donor_strategy: str
    donor_interval: tuple[int, int] | None
    claimed_contribution: Tensor
    logit_delta: Tensor  # original minus replaced
    background_delta: Tensor
    event_delta: Tensor
    diagnostics: dict


def validate_replacement(original: Tensor, replacement: Tensor, *, max_rms_ratio=4., max_edge_ratio=4.) -> dict:
    if (original.shape != replacement.shape or original.ndim != 1 or original.numel() < 2
            or not torch.isfinite(original).all() or not torch.isfinite(replacement).all()):
        raise ValueError("invalid counterfactual replacement shape/values")
    old_rms = original.square().mean().sqrt().clamp_min(1.)
    ratio = float(replacement.square().mean().sqrt() / old_rms)
    edge = max(float((replacement[0] - original[0]).abs()),
               float((replacement[-1] - original[-1]).abs()))
    derivative_ratio = float(replacement.diff().abs().max() / original.diff().abs().max().clamp_min(1.))
    if ratio > max_rms_ratio or edge > max_edge_ratio * float(old_rms) or derivative_ratio > 4.:
        raise ValueError("counterfactual OOD energy/edge/slope guard failed")
    return {"replacement_rms_ratio": ratio, "endpoint_change_uV": edge,
            "max_derivative_ratio": derivative_ratio, "physiological_validity": "NOT_ESTABLISHED"}


@torch.no_grad()
def audit_interval(model: CoveModel, x: Tensor, interval: tuple[int, int], *,
                   strategy: str = "linear_bridge", donor_interval: tuple[int, int] | None = None,
                   mode: str = "full") -> AuditResult:
    if model.training:
        raise ValueError("audit requires eval mode")
    if len(x) != 1:
        raise ValueError("audit one physical epoch at a time")
    start, end = interval
    if not (type(start) is int and type(end) is int and 0 <= start < end <= 3000 and end - start >= 40):
        raise ValueError("invalid half-open audit interval")
    physical = model.physical_uV(x).clone()
    old = physical[0, 0, start:end].clone()
    bridge = torch.linspace(0, 1, end - start, device=x.device) * (old[-1] - old[0]) + old[0]
    if strategy == "linear_bridge":
        if donor_interval is not None:
            raise ValueError("bridge has no donor interval")
        donor = bridge
    elif strategy == "same_epoch_background":
        if donor_interval is None:
            raise ValueError("explicit donor interval required")
        ds, de = donor_interval
        if not (type(ds) is int and type(de) is int and 0 <= ds < de <= 3000
                and de - ds == end - start and (de <= start or ds >= end)):
            raise ValueError("donor must be disjoint and length matched")
        donor = physical[0, 0, ds:de].clone()
        # Explicit conservative background compatibility check in physical uV.
        if donor.std(unbiased=False) > max(3., float(old.std(unbiased=False)) * .5):
            raise ValueError("donor is not background compatible")
        donor = donor - donor.mean() + bridge
    else:
        raise ValueError("unregistered donor strategy")
    # 20-sample crossfade leaves original endpoints exactly unchanged.
    blend = torch.ones_like(old)
    blend[:20] = torch.linspace(0, 1, 20, device=x.device)
    blend[-20:] = torch.linspace(1, 0, 20, device=x.device)
    replacement = old * (1 - blend) + donor * blend
    diagnostics = validate_replacement(old, replacement)
    physical[0, 0, start:end] = replacement
    changed = (physical - model.mean_uV) / model.std_uV
    before = model(x, return_evidence=True, mode=mode)
    after = model(changed, return_evidence=True, mode=mode)
    overlaps = ((before.event_intervals[0, :, 0] < end)
                & (before.event_intervals[0, :, 1] > start) & before.valid[0])
    diagnostics.update({"selection_changed": not torch.equal(before.event_intervals, after.event_intervals),
                        "boundary_blend_samples": 20, "recipe_version": "same_epoch_audit_v1",
                        "overlapping_claims": int(overlaps.sum())})
    return AuditResult(interval, strategy, donor_interval, before.contributions[0, overlaps].sum(0),
                       before.logits[0] - after.logits[0],
                       before.background_logits[0] - after.background_logits[0],
                       before.contributions[0].sum(0) - after.contributions[0].sum(0), diagnostics)

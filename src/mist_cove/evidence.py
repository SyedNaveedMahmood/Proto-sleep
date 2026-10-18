"""Tensor evidence and physical-grid metadata. No file or dataset access."""
from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class Evidence:
    logits: Tensor
    background_logits: Tensor
    background_features: Tensor
    background_contributions: Tensor
    background_bias: Tensor
    event_intervals: Tensor
    valid: Tensor
    event_scores: Tensor
    signed_evidence: Tensor
    availability: Tensor
    contributions: Tensor
    acquisition_observability_metadata: dict
    selection_policy: str
    physical_input_sha256: tuple[str, ...]


def wave_digest(wave_uV: Tensor) -> str:
    return hashlib.sha256(wave_uV.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class SyntheticTrace:
    """An in-memory synthetic identity, never authority to open a real source.

    Epoch indices may have gaps and scoring_mask never filters inference. Hashes
    identify generated physical-uV bytes, not real source proofs. This phase has
    no real-data trace adapter; future adapters must verify SourceProofs first.
    """
    recording_id: str
    night: str
    original_epoch_index: int
    source_sha256: str
    scoring_mask: bool = True
    role: str = "synthetic"
    channel: str = "EEG Fpz-Cz"

    def __post_init__(self):
        if self.role != "synthetic":
            raise ValueError("Phase I permits synthetic traces only")
        if (type(self.original_epoch_index) is not int or self.original_epoch_index < 0
                or not self.recording_id or not self.night or self.channel != "EEG Fpz-Cz"
                or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("invalid physical-grid trace")

    @classmethod
    def from_wave(cls, wave_uV: Tensor, **kwargs):
        if wave_uV.shape != (1, 3000) or not torch.isfinite(wave_uV).all():
            raise ValueError("trace requires one finite physical epoch")
        digest = wave_digest(wave_uV)
        return cls(source_sha256=digest, **kwargs)


def interval_ledger(evidence: Evidence, traces: list[SyntheticTrace]) -> list[dict]:
    if len(traces) != len(evidence.logits):
        raise ValueError("one trace per physical epoch required")
    rows = []
    for b, trace in enumerate(traces):
        if trace.source_sha256 != evidence.physical_input_sha256[b]:
            raise ValueError("trace does not match the inferred physical waveform")
        for k in range(evidence.valid.shape[1]):
            if not bool(evidence.valid[b, k]):
                continue
            start, end = evidence.event_intervals[b, k].tolist()
            base = 3000 * trace.original_epoch_index
            rows.append({**trace.__dict__, "event": k, "start_sample": start,
                         "end_sample": end, "absolute_start_sample": base + start,
                         "absolute_end_sample": base + end,
                         "start_seconds": (base + start) / 100,
                         "end_seconds": (base + end) / 100,
                         "selection_policy": evidence.selection_policy,
                         "signed_evidence": evidence.signed_evidence[b, k].detach().tolist(),
                         "availability": evidence.availability[b, k].detach().tolist(),
                         "contributions": evidence.contributions[b, k].detach().tolist()})
    return rows


def masked_loss(logits: Tensor, labels: Tensor, scoring_mask: Tensor) -> Tensor:
    if (logits.ndim != 2 or logits.shape[1] != 5 or labels.shape != logits.shape[:1]
            or scoring_mask.shape != labels.shape or scoring_mask.dtype != torch.bool):
        raise ValueError("expected five logits and a boolean physical-grid scoring mask")
    if not torch.isfinite(logits).all() or not torch.all((labels[scoring_mask] >= 0) & (labels[scoring_mask] < 5)):
        raise ValueError("invalid scored inputs")
    # Preserve a differentiable zero on an entirely unscored batch.
    return F.cross_entropy(logits[scoring_mask], labels[scoring_mask]) if scoring_mask.any() else logits.sum() * 0

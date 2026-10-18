"""Separate adapter for the immutable epoch-local source trainer/evaluator."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F

from mist_cove.model import CoveConfig
from .model import make_model


@dataclass(frozen=True)
class AdapterConfig:
    method: str
    mean_uV: float
    std_uV: float
    radius: int = 0
    use_crf: bool = False

    def dictionary(self):
        return {**asdict(self), "version": "mist-cove-source-adapter-v1", "model": asdict(CoveConfig())}


class SourceAdapter(nn.Module):
    def __init__(self, method="q1", *, mean_uV=0., std_uV=1.):
        super().__init__()
        self.cfg = AdapterConfig(method, mean_uV, std_uV)
        self.network = make_model(method, mean_uV=mean_uV, std_uV=std_uV)

    def encode(self, x):
        return self.network(x)

    def emissions(self, features):
        if features.shape[-1] != 5:
            raise ValueError("adapter emissions require five encoded logits")
        return features

    def loss(self, emissions, labels, valid):
        if valid.dtype != torch.bool or labels.shape != emissions.shape[:-1] or valid.shape != labels.shape:
            raise ValueError("boolean physical-grid loss mask required")
        if not torch.isfinite(emissions).all() or not ((labels[valid] >= 0) & (labels[valid] < 5)).all():
            raise ValueError("invalid supervised logits/labels")
        return F.cross_entropy(emissions[valid], labels[valid]) if valid.any() else emissions.sum() * 0

    @torch.no_grad()
    def predict(self, logits):
        if logits.shape[-1] != 5 or not torch.isfinite(logits).all():
            raise ValueError("finite five-stage logits required")
        return logits.argmax(-1), logits.float().softmax(-1)


def build_adapter(name, normalization, bank=None):
    if name in {"raw_context", "evidence"}:
        if bank is None:
            raise ValueError("unchanged historical definitions require a verified TRAIN bank")
        from mist_transfer.baselines import make_baseline
        return make_baseline(name, bank, normalization)
    return SourceAdapter(name, mean_uV=normalization["mean_uV"], std_uV=normalization["std_uV"])

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from protosleep.legacy_wavesleepnet_objective import HistoricalProtoObjective


class DummyTrainer:
    def _diversity_cal(self, x):
        return x.square().mean() + 1.0

    def protop_loss(self, outputs, labels):
        ce = F.cross_entropy(outputs, labels)
        proto = self.model.module.prototype_vectors
        pd = 1.0 / torch.log(self._diversity_cal(proto) + 1.0)
        weight = self.model.module.fc.weight.abs().sum()
        self.loss_ensemble["cross_entropy"] = ce
        self.loss_ensemble["pd_loss"] = pd
        self.loss_ensemble["weight_loss"] = weight
        return ce + pd + 0.01 * weight


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prototype_vectors = nn.Parameter(torch.rand(2, 3, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_historical_objective_harness_binds_dataparallel_style_model():
    module = SimpleNamespace(OneFoldTrainer=DummyTrainer)
    model = DummyModel()
    objective = HistoricalProtoObjective(module, model, {"classifier": {}})

    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 0])
    logits = model(x)
    loss, terms = objective(logits, y)

    assert torch.isfinite(loss)
    assert set(terms) == {"cross_entropy", "pd_loss", "weight_loss"}
    loss.backward()
    assert model.fc.weight.grad is not None
    assert model.prototype_vectors.grad is not None

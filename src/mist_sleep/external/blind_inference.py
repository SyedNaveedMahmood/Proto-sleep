"""Inference guard: no optimizers, backward calls, training mode or state mutation.

The guard is process-global and intended for a dedicated single-threaded runner.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from unittest.mock import patch
import numpy as np
import torch


def state_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _forbidden(*args, **kwargs):
    raise RuntimeError("Phase M1 forbids optimizer initialization and backward passes")


@contextmanager
def frozen_execution(model):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model must be eval with every parameter frozen")
    before = state_digest(model)
    with patch.object(torch.optim.Optimizer, "__init__", _forbidden), \
         patch.object(torch.Tensor, "backward", _forbidden), \
         patch.object(torch.autograd, "backward", _forbidden), \
         patch.object(torch.autograd, "grad", _forbidden), torch.inference_mode():
        try:
            yield
        finally:
            if before != state_digest(model) or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("frozen source model mutated during inference")


def predict(model, epochs, *, batch_size=32):
    x = np.asarray(epochs)
    if x.ndim != 3 or x.shape[1:] != (1, 3000) or not len(x) or not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("invalid complete physical epochs")
    outputs = {"logits": [], "embedding": [], "prototype_usage": []}
    device = next(model.parameters()).device
    with frozen_execution(model):
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size].astype(np.float32)).to(device)
            result = model(batch)
            for name, tensor in (("logits", result["logits"]), ("embedding", result["embedding"]),
                                 ("prototype_usage", result["q"].mean(1))):
                if not torch.isfinite(tensor).all():
                    raise ValueError("nonfinite prediction")
                outputs[name].append(tensor.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in outputs.items()}

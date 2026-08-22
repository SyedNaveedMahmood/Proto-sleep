from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .legacy_wavesleepnet import (
    _install_bundled_attnsleep_alias,
    _pop_module_tree,
    _restore_module_tree,
    legacy_paths,
    load_historical_edf20_config,
    patched_mist_config,
    validate_legacy_snapshot,
)


def import_legacy_trainer_module(legacy_root: Path | str):
    """Import the audited historical WaveSleepNet trainer without modifying the archive.

    The recovered trainer uses absolute imports such as ``models.protop``, ``loader`` and
    ``utils``. Import it inside an isolated namespace, using the same bundled-AttnSleep
    compatibility alias already validated by the ProtoPNet smoke test.
    """
    root = Path(legacy_root).expanduser().resolve()
    validate_legacy_snapshot(root)
    paths = legacy_paths(root)
    wavesleep_root = paths["trainer"].parent
    attnsleep_root = root / "external" / "AttnSleep-main"
    module_path = paths["trainer"]

    old_path = list(sys.path)
    saved = {
        prefix: _pop_module_tree(prefix)
        for prefix in ("models", "loader", "utils")
    }
    try:
        for path_str in (str(wavesleep_root), str(attnsleep_root)):
            while path_str in sys.path:
                sys.path.remove(path_str)
        sys.path.insert(0, str(attnsleep_root))
        sys.path.insert(0, str(wavesleep_root))

        _install_bundled_attnsleep_alias(root, wavesleep_root)

        spec = importlib.util.spec_from_file_location(
            "protosleep_legacy_wavesleepnet_train_mtcl", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        for prefix in ("utils", "loader", "models"):
            _restore_module_tree(prefix, saved[prefix])

    if not hasattr(module, "OneFoldTrainer"):
        raise AttributeError(f"{module_path} does not define OneFoldTrainer")
    cls = module.OneFoldTrainer
    for name in ("protop_loss", "_diversity_cal"):
        if not hasattr(cls, name):
            raise AttributeError(f"Historical OneFoldTrainer is missing {name}")
    return module


class HistoricalProtoObjective:
    """Bind the exact archived ``OneFoldTrainer.protop_loss`` to a direct ProtoPNet.

    The historical method expects ``self.model.module`` because its original runner used
    DataParallel. We deliberately avoid DataParallel here and provide only that compatibility
    wrapper. The loss function and its helper methods are executed from the frozen archived
    trainer source unchanged.
    """

    def __init__(self, trainer_module: Any, model: nn.Module, cfg: Mapping[str, Any]):
        cls = trainer_module.OneFoldTrainer
        instance = cls.__new__(cls)  # bypass historical __init__ / dataloader side effects
        instance.cfg = dict(cfg)
        instance.model = SimpleNamespace(module=model)
        instance.loss_ensemble = {}
        # Defensive compatibility only: these are harmless if the archived method does not
        # reference them, and standard CE if it does.
        instance.criterion = nn.CrossEntropyLoss()
        instance.ce_loss = instance.criterion
        self.instance = instance
        self.model = model
        self.fn = cls.protop_loss

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.instance.model.module = self.model
        self.instance.loss_ensemble = {}
        loss = self.fn(self.instance, logits, labels)
        if not torch.is_tensor(loss):
            raise TypeError(f"Historical protop_loss returned {type(loss)!r}, expected Tensor")
        terms = {
            str(k): v for k, v in self.instance.loss_ensemble.items()
            if torch.is_tensor(v)
        }
        return loss, terms


def historical_training_config(legacy_root: Path | str) -> Dict[str, Any]:
    """Return the validated historical EDF-2013 config with the documented 27->30 bridge."""
    cfg = load_historical_edf20_config(legacy_root)
    return patched_mist_config(cfg)


def finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Historical objective produced non-finite tensor: {name}")


def one_historical_optimizer_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    trainer_module: Any,
    cfg: Mapping[str, Any],
    device: torch.device | str,
) -> Dict[str, Any]:
    """Run exactly one FP32 Adam step using the archived prototype objective."""
    device = torch.device(device)
    model = model.to(device)
    model.train()
    x = x.to(device=device, dtype=torch.float32, non_blocking=True)
    y = y.to(device=device, dtype=torch.long, non_blocking=True)

    tp = cfg["training_params"]
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tp["lr"]),
        weight_decay=float(tp["weight_decay"]),
    )
    objective = HistoricalProtoObjective(trainer_module, model, cfg)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        if len(logits) != 1:
            raise RuntimeError(
                "Historical ProtoPNet returned multiple scale outputs in objective smoke; "
                "the validated EDF-2013 configuration was expected to produce one tensor."
            )
        logits = logits[0]
    if not torch.is_tensor(logits):
        raise TypeError(f"Historical ProtoPNet returned {type(logits)!r}, expected Tensor")
    finite_tensor("logits", logits)

    loss, terms = objective(logits, y)
    finite_tensor("loss", loss)
    for name, value in terms.items():
        finite_tensor(name, value)

    loss.backward()

    grad_sq = 0.0
    n_grad = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        finite_tensor(f"grad:{name}", p.grad)
        grad_sq += float(torch.sum(p.grad.detach().float() ** 2).cpu())
        n_grad += 1
    if n_grad == 0:
        raise RuntimeError("Historical objective produced no parameter gradients")
    grad_norm = math.sqrt(grad_sq)
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError(f"Invalid historical objective gradient norm: {grad_norm}")

    before = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    optimizer.step()

    changed = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before:
            continue
        finite_tensor(f"parameter:{name}", p.detach())
        if not torch.equal(before[name], p.detach().cpu()):
            changed += 1
    if changed == 0:
        raise RuntimeError("Historical optimizer step changed no trainable parameters")

    return {
        "loss": float(loss.detach().cpu()),
        "terms": {k: float(v.detach().cpu()) for k, v in terms.items()},
        "grad_norm": float(grad_norm),
        "n_parameter_tensors_with_grad": int(n_grad),
        "n_parameter_tensors_changed": int(changed),
        "logit_shape": list(logits.shape),
        "optimizer": "Adam",
        "lr": float(tp["lr"]),
        "weight_decay": float(tp["weight_decay"]),
        "precision": "fp32",
    }

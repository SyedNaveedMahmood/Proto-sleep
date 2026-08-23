from __future__ import annotations

import importlib.util
from pathlib import Path

from protosleep.morphmae_transfer import MAE_PROBE


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_morphspec_frozen_transfer_confirm.py"
    spec = importlib.util.spec_from_file_location("run_morphspec_frozen_transfer_confirm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_transfer_recipe_keeps_encoder_frozen():
    assert MAE_PROBE.name == "mae_probe"
    assert MAE_PROBE.encoder_lr == 0.0
    assert MAE_PROBE.freeze_encoder is True


def test_runner_defaults_to_last_untouched_folds(monkeypatch):
    module = _load_runner_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_morphspec_frozen_transfer_confirm.py",
            "--data-dir",
            "/tmp/data",
            "--original-checkpoint-pattern",
            "/tmp/original_{fold}.pt",
            "--morphspec-checkpoint-pattern",
            "/tmp/morphspec_{fold}.pt",
        ],
    )
    args = module.parse_args()
    assert args.folds == [15, 16, 17, 18, 19]
    assert args.seeds == [123, 456, 789]


def test_gate_requires_positive_mean_and_three_of_five():
    module = _load_runner_module()
    assert module._passes_gate(0.001, 3, 5) is True
    assert module._passes_gate(0.001, 2, 5) is False
    assert module._passes_gate(-0.001, 5, 5) is False


def test_exact_sign_test_five_of_five():
    module = _load_runner_module()
    assert module._exact_sign_p(5, 0) == 0.0625

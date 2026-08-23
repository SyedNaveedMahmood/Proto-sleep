from __future__ import annotations

import importlib.util
from pathlib import Path

from protosleep.morphmae_transfer import MAE_STAGE_1E4


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_morphspec_downstream_confirm.py"
    spec = importlib.util.spec_from_file_location("run_morphspec_downstream_confirm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_downstream_transfer_recipe_is_stage_1e4():
    assert MAE_STAGE_1E4.name == "mae_stage_1e4"
    assert MAE_STAGE_1E4.warmup_epochs == 5
    assert MAE_STAGE_1E4.encoder_lr == 1e-4
    assert MAE_STAGE_1E4.head_lr == 1e-3
    assert MAE_STAGE_1E4.freeze_encoder is False


def test_exact_sign_test_five_of_five_is_one_sixteenth_two_sided():
    module = _load_runner_module()
    assert module._exact_sign_p(5, 0) == 0.0625


def test_runner_uses_new_downstream_folds_by_default(monkeypatch):
    module = _load_runner_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_morphspec_downstream_confirm.py",
            "--data-dir",
            "/tmp/data",
            "--original-checkpoint-pattern",
            "/tmp/original_{fold}.pt",
            "--morphspec-checkpoint-pattern",
            "/tmp/morphspec_{fold}.pt",
        ],
    )
    args = module.parse_args()
    assert args.folds == [10, 11, 12, 13, 14]
    assert args.seeds == [123, 456, 789]

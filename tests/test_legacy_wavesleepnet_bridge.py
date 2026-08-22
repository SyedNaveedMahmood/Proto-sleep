from __future__ import annotations

import copy
from collections.abc import Mapping

from protosleep.legacy_wavesleepnet import (
    EXPECTED_HISTORICAL_CONFIG,
    patched_mist_config,
    validate_historical_edf20_config,
)


def historical_config_fixture():
    cfg = {
        "classifier": {
            "prototype_num": 10,
            "prototype_shape": [10, 27, 1],
            "afr_reduced_dim": 27,
            "dist_lambda": 17.8373,
            "class_lambda": 50,
            "identity_lambda": 8.9351,
            "pd_lambda": 7.9252,
            "weight_lambda": 0.3,
        },
        "training_params": {
            "max_epochs": 5000,
            "batch_size": 64,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "early_stopping": {"mode": "min", "patience": 50},
        },
        "unrelated": {"keep": "identical"},
    }
    # Guard the fixture itself against accidental drift from the audited constants.
    for path, expected in EXPECTED_HISTORICAL_CONFIG.items():
        node = cfg
        for key in path:
            assert isinstance(node, Mapping)
            node = node[key]
        assert node == expected
    return cfg


def test_patch_changes_only_documented_channel_bridge():
    cfg = historical_config_fixture()
    original = copy.deepcopy(cfg)

    validate_historical_edf20_config(cfg)
    patched = patched_mist_config(cfg)

    assert cfg == original
    assert patched["classifier"]["afr_reduced_dim"] == 30
    assert patched["classifier"]["prototype_shape"] == [10, 30, 1]

    expected = copy.deepcopy(original)
    expected["classifier"]["afr_reduced_dim"] = 30
    expected["classifier"]["prototype_shape"] = [10, 30, 1]
    assert patched == expected


def test_historical_config_validation_fails_closed():
    cfg = historical_config_fixture()
    cfg["classifier"]["pd_lambda"] = 0.0

    try:
        validate_historical_edf20_config(cfg)
    except RuntimeError as exc:
        assert "pd_lambda" in str(exc)
    else:
        raise AssertionError("Expected audited historical config mismatch to fail")


def test_historical_config_requires_nested_early_stopping():
    cfg = historical_config_fixture()
    early_stopping = cfg["training_params"].pop("early_stopping")
    cfg["early_stopping"] = early_stopping

    try:
        validate_historical_edf20_config(cfg)
    except RuntimeError as exc:
        assert "training_params.early_stopping.patience" in str(exc)
    else:
        raise AssertionError("Expected misplaced early-stopping config to fail")

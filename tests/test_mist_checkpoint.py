from __future__ import annotations

import torch

from protosleep.attnsleep import MRCNN
from protosleep.mist import extract_mrcnn_state_dict


def _clone_state(model: MRCNN):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def test_extract_raw_mrcnn_state_dict(tmp_path):
    model = MRCNN(30)
    expected = _clone_state(model)

    path = tmp_path / "raw_mrcnn.pt"
    torch.save(expected, path)

    found, meta = extract_mrcnn_state_dict(path)

    assert meta["container"] == "root"
    assert meta["prefix"] == ""
    assert meta["n_tensors"] == len(expected)
    assert set(found) == set(expected)
    for key in expected:
        assert torch.equal(found[key], expected[key])


def test_extract_prefixed_wrapped_mrcnn_state_dict(tmp_path):
    model = MRCNN(30)
    expected = _clone_state(model)

    wrapped = {
        "state_dict": {
            f"module.encoder.{key}": value.clone()
            for key, value in expected.items()
        },
        "train_subjects": [0, 1, 2],
        "fold": 0,
        "seed": 123,
    }

    path = tmp_path / "wrapped_morphmae.pt"
    torch.save(wrapped, path)

    found, meta = extract_mrcnn_state_dict(path)

    assert meta["container"] == "state_dict"
    assert meta["prefix"] == "module.encoder."
    assert meta["train_subjects"] == [0, 1, 2]
    assert meta["fold"] == 0
    assert meta["seed"] == 123
    assert set(found) == set(expected)
    for key in expected:
        assert torch.equal(found[key], expected[key])

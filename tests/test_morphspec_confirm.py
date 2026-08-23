from __future__ import annotations

import torch

from protosleep.attnsleep import MRCNN
from protosleep.morphspec_confirm import build_matched_probe_triplet


def _state_with_shift(shift: float):
    model = MRCNN(30)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    first_float = next(k for k, v in state.items() if torch.is_floating_point(v))
    state[first_float] = state[first_float] + float(shift)
    return state


def test_build_matched_probe_triplet(tmp_path):
    original = tmp_path / "original.pt"
    refined = tmp_path / "refined.pt"
    torch.save({"state_dict": _state_with_shift(0.1)}, original)
    torch.save({"state_dict": _state_with_shift(0.2)}, refined)

    random_model, original_model, morphspec_model, meta = build_matched_probe_triplet(
        123, original, refined
    )

    assert meta["non_mrcnn_initialization_match"] is True
    assert len({
        meta["random_mrcnn_sha256"],
        meta["original_mae_mrcnn_sha256"],
        meta["morphspec_mrcnn_sha256"],
    }) == 3

    sr = random_model.state_dict()
    so = original_model.state_dict()
    sm = morphspec_model.state_dict()
    for key in sr:
        if key.startswith("mrcnn."):
            continue
        assert torch.equal(sr[key], so[key])
        assert torch.equal(sr[key], sm[key])

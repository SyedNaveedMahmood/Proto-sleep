from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .attnsleep import AttnSleepBaseline, init_attnsleep_weights
from .mist import load_mrcnn_checkpoint, mrcnn_state_digest
from .utils import cpu_state_dict, seed_everything


def _non_mrcnn_mismatches(a: torch.nn.Module, b: torch.nn.Module) -> list[str]:
    sa = cpu_state_dict(a)
    sb = cpu_state_dict(b)
    bad: list[str] = []
    for key, value in sa.items():
        if key.startswith("mrcnn."):
            continue
        if key not in sb or not torch.equal(value, sb[key]):
            bad.append(key)
    return bad


def build_matched_probe_triplet(
    seed: int,
    original_mae_checkpoint: Path | str,
    morphspec_checkpoint: Path | str,
) -> Tuple[AttnSleepBaseline, AttnSleepBaseline, AttnSleepBaseline, Dict[str, Any]]:
    """Build matched random/original-MAE/MorphSpec frozen-probe models.

    All three members begin with byte-identical TCE/classifier initialization. Only the
    MRCNN state differs: random initialization, original leakage-safe MorphMAE-v2, or the
    frozen MorphSpec-R1 refinement of that fold's MorphMAE-v2 checkpoint.
    """

    seed_everything(int(seed))
    template = AttnSleepBaseline()
    template.apply(init_attnsleep_weights)

    random_model = copy.deepcopy(template)
    original_model = copy.deepcopy(template)
    morphspec_model = copy.deepcopy(template)

    random_digest = mrcnn_state_digest(random_model.mrcnn)
    original_meta = load_mrcnn_checkpoint(original_model.mrcnn, original_mae_checkpoint)
    morphspec_meta = load_mrcnn_checkpoint(morphspec_model.mrcnn, morphspec_checkpoint)
    original_digest = mrcnn_state_digest(original_model.mrcnn)
    morphspec_digest = mrcnn_state_digest(morphspec_model.mrcnn)

    if random_digest == original_digest:
        raise RuntimeError("Original MorphMAE MRCNN equals random initialization")
    if random_digest == morphspec_digest:
        raise RuntimeError("MorphSpec MRCNN equals random initialization")
    if original_digest == morphspec_digest:
        raise RuntimeError("MorphSpec refinement did not change the original MorphMAE MRCNN")

    for name, model in (("original_mae", original_model), ("morphspec", morphspec_model)):
        bad = _non_mrcnn_mismatches(random_model, model)
        if bad:
            raise RuntimeError(f"{name} non-MRCNN initialization mismatch: {bad[:8]}")

    return random_model, original_model, morphspec_model, {
        "random_mrcnn_sha256": random_digest,
        "original_mae_mrcnn_sha256": original_digest,
        "morphspec_mrcnn_sha256": morphspec_digest,
        "original_mae_metadata": original_meta,
        "morphspec_metadata": morphspec_meta,
        "non_mrcnn_initialization_match": True,
    }

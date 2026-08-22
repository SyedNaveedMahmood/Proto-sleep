from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch

from .attnsleep import AttnSleepBaseline, init_attnsleep_weights
from .mist import extract_mrcnn_state_dict, load_mrcnn_checkpoint, mrcnn_state_digest
from .utils import cpu_state_dict, seed_everything


def verify_mae_checkpoint_for_fold(
    checkpoint_path: Path | str,
    fold: int,
    train_subjects: Sequence[int],
) -> Dict[str, Any]:
    """Fail closed unless a canonical MorphMAE checkpoint belongs to this fold/train split."""
    _, meta = extract_mrcnn_state_dict(checkpoint_path)
    expected = sorted(int(x) for x in train_subjects)

    declared = None
    for key in ("train_subjects", "pretrain_subjects", "subjects"):
        if key in meta:
            declared = sorted(int(x) for x in meta[key])
            break
    if declared is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no train/pretrain subject metadata; "
            "refusing an unverified A2 run."
        )
    if declared != expected:
        raise RuntimeError(
            f"Checkpoint subjects {declared} do not equal fold-{fold} training subjects {expected}."
        )

    if "fold" in meta and int(meta["fold"]) != int(fold):
        raise RuntimeError(
            f"Checkpoint declares fold={meta['fold']}, but the requested downstream fold is {fold}."
        )
    return meta


def build_matched_a1_a2(
    seed: int,
    mae_checkpoint: Path | str,
) -> Tuple[AttnSleepBaseline, AttnSleepBaseline, Dict[str, Any]]:
    """Build the clean AttnSleep causal pair for testing MorphMAE initialization.

    A1: random AttnSleep initialization.
    A2: byte-identical A1 initialization outside MRCNN, with only MRCNN replaced by the
        strictly compatible MorphMAE checkpoint.
    """
    seed_everything(int(seed))
    template = AttnSleepBaseline()
    template.apply(init_attnsleep_weights)

    a1 = copy.deepcopy(template)
    a2 = copy.deepcopy(template)

    a1_digest = mrcnn_state_digest(a1.mrcnn)
    mae_meta = load_mrcnn_checkpoint(a2.mrcnn, mae_checkpoint)
    a2_digest = mrcnn_state_digest(a2.mrcnn)
    if a1_digest == a2_digest:
        raise RuntimeError(
            "MorphMAE MRCNN digest equals the random A1 MRCNN digest; refusing an inactive A1/A2 manipulation."
        )

    s1 = cpu_state_dict(a1)
    s2 = cpu_state_dict(a2)
    mismatch = []
    for key, value in s1.items():
        if key.startswith("mrcnn."):
            continue
        if key not in s2 or not torch.equal(value, s2[key]):
            mismatch.append(key)
    if mismatch:
        raise RuntimeError(f"A1/A2 non-MRCNN initialization mismatch: {mismatch[:8]}")

    return a1, a2, {
        **mae_meta,
        "a1_random_mrcnn_sha256": a1_digest,
        "a2_mae_mrcnn_sha256": a2_digest,
        "non_mrcnn_initialization_match": True,
    }

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from .attnsleep import MRCNN, AttnSleepBaseline, init_attnsleep_weights
from .config import ENABLE_MICRO_MASK, PROTO_TRIALS
from .prototypes import ProtoAttnSleep
from .utils import cpu_state_dict, seed_everything


class MRCNNCheckpointError(RuntimeError):
    """Raised when an MAE checkpoint cannot be mapped exactly onto AttnSleep MRCNN."""


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_torch_load(path: Path) -> Any:
    """
    Load ordinary tensor/state-dict checkpoints without executing arbitrary pickled code.

    Modern PyTorch supports weights_only=True. Older versions do not, so a compatibility
    fallback is retained for checkpoints created by the user's own experiment code.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping) and bool(obj) and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def _candidate_state_mappings(checkpoint: Any) -> Iterable[Tuple[str, Mapping[str, torch.Tensor]]]:
    """Yield plausible state-dict containers from common checkpoint layouts."""
    if _tensor_mapping(checkpoint):
        yield "root", checkpoint

    if not isinstance(checkpoint, Mapping):
        return

    preferred = (
        "mrcnn_state_dict",
        "encoder_state_dict",
        "state_dict",
        "model_state_dict",
        "model",
        "encoder",
        "mrcnn",
    )
    seen = set()
    for key in preferred:
        value = checkpoint.get(key)
        if _tensor_mapping(value):
            seen.add(id(value))
            yield key, value

    # One additional level handles layouts such as {"model": {"state_dict": ...}}
    # without recursively walking arbitrary checkpoint objects.
    for outer_key, outer_value in checkpoint.items():
        if not isinstance(outer_value, Mapping):
            continue
        for inner_key in preferred:
            value = outer_value.get(inner_key)
            if _tensor_mapping(value) and id(value) not in seen:
                seen.add(id(value))
                yield f"{outer_key}.{inner_key}", value


def _match_mrcnn_state(
    state: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    Find a single prefix whose removal gives an exact MRCNN state dict.

    This handles raw MRCNN checkpoints as well as full-model checkpoints such as
    ``encoder.features1...``, ``mrcnn.features1...`` or ``module.encoder...`` without
    fuzzy/partial key matching.
    """
    target_keys = list(target.keys())
    if not target_keys:
        return None

    first = target_keys[0]
    prefixes = {""}
    for key in state.keys():
        if key.endswith(first):
            prefixes.add(key[: -len(first)])

    for prefix in sorted(prefixes, key=lambda s: (len(s), s)):
        if not all(prefix + key in state for key in target_keys):
            continue

        canonical: Dict[str, torch.Tensor] = {}
        shape_bad = []
        for key, ref in target.items():
            value = state[prefix + key]
            if tuple(value.shape) != tuple(ref.shape):
                shape_bad.append((key, tuple(value.shape), tuple(ref.shape)))
            canonical[key] = value.detach().cpu()
        if shape_bad:
            continue
        return prefix, canonical

    return None


def extract_mrcnn_state_dict(checkpoint_path: Path | str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract an AttnSleep-compatible MRCNN state dict using exact key/shape matching.

    No approximate key matching is used. This is deliberately strict because the historical
    MIST experiments previously suffered a checkpoint-loading failure; a silent partial load
    would invalidate the A3-vs-A4 comparison.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = _safe_torch_load(path)
    target_model = MRCNN(30)
    target_state = target_model.state_dict()

    attempts = []
    for container_name, state in _candidate_state_mappings(checkpoint):
        match = _match_mrcnn_state(state, target_state)
        attempts.append(container_name)
        if match is None:
            continue
        prefix, canonical = match
        target_model.load_state_dict(canonical, strict=True)
        metadata = {
            "path": str(path),
            "sha256": sha256_file(path),
            "container": container_name,
            "prefix": prefix,
            "n_tensors": len(canonical),
        }
        if isinstance(checkpoint, Mapping):
            for key in (
                "train_subjects",
                "pretrain_subjects",
                "subjects",
                "fold",
                "seed",
                "project_version",
            ):
                if key in checkpoint:
                    value = checkpoint[key]
                    if torch.is_tensor(value):
                        value = value.detach().cpu().tolist()
                    metadata[key] = value
        return canonical, metadata

    raise MRCNNCheckpointError(
        "Could not find an exact AttnSleep MRCNN state dict in checkpoint. "
        f"Tried containers: {attempts or ['<none>']}. "
        "The checkpoint must contain all MRCNN parameters/buffers with exact shapes."
    )


def load_mrcnn_checkpoint(module: MRCNN, checkpoint_path: Path | str) -> Dict[str, Any]:
    state, metadata = extract_mrcnn_state_dict(checkpoint_path)
    module.load_state_dict(state, strict=True)
    return metadata


def save_canonical_mrcnn_checkpoint(source: Path | str, destination: Path | str) -> Dict[str, Any]:
    """
    Convert any strictly compatible source checkpoint to a compact MRCNN-only checkpoint.

    Provenance and split metadata are preserved at the top level so the leakage guard in
    ``run_mist_stability.py`` remains effective after conversion.
    """
    state, metadata = extract_mrcnn_state_dict(source)
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "state_dict": state,
        "source_checkpoint": metadata["path"],
        "source_sha256": metadata["sha256"],
        "source_container": metadata["container"],
        "source_prefix": metadata["prefix"],
    }
    for key in (
        "train_subjects",
        "pretrain_subjects",
        "subjects",
        "fold",
        "seed",
        "project_version",
    ):
        if key in metadata:
            payload[key] = metadata[key]

    torch.save(payload, destination)
    return {**metadata, "canonical_path": str(destination)}


def mrcnn_state_digest(module: MRCNN) -> str:
    """Deterministic digest of MRCNN tensors for experiment provenance."""
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        h.update(key.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def build_a1(seed: int) -> AttnSleepBaseline:
    """A1: standard AttnSleep random initialization."""
    seed_everything(seed)
    model = AttnSleepBaseline()
    model.apply(init_attnsleep_weights)
    return model


def build_matched_a3_a4(
    seed: int,
    mae_checkpoint: Path | str,
) -> Tuple[ProtoAttnSleep, ProtoAttnSleep, Dict[str, Any]]:
    """
    Build a matched current-prototype pair for the MIST mechanism audit.

    A3-current: random AttnSleep MRCNN initialization + prototype model.
    A4-current: identical non-MRCNN initialization, but MRCNN is replaced by the
    strictly loaded MAE-pretrained checkpoint.

    One template plus deepcopy makes prototypes, TCE, classifier, beta and every other
    parameter identical at initialization. The controlled difference is MRCNN initialization.
    """
    seed_everything(seed)
    template = ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)
    template.apply(init_attnsleep_weights)

    a3 = copy.deepcopy(template)
    a4 = copy.deepcopy(template)

    a3_digest = mrcnn_state_digest(a3.mrcnn)
    mae_metadata = load_mrcnn_checkpoint(a4.mrcnn, mae_checkpoint)
    a4_digest = mrcnn_state_digest(a4.mrcnn)

    s3 = cpu_state_dict(a3)
    s4 = cpu_state_dict(a4)
    mismatch = []
    for key in s3:
        if key.startswith("mrcnn."):
            continue
        if key not in s4 or not torch.equal(s3[key], s4[key]):
            mismatch.append(key)
    if mismatch:
        raise RuntimeError(f"A3/A4 non-MRCNN initialization mismatch: {mismatch[:8]}")

    if a3_digest == a4_digest:
        raise RuntimeError(
            "MAE checkpoint produced the same MRCNN digest as the random A3 initialization. "
            "Refusing to run because the A3-vs-A4 manipulation is not active."
        )

    metadata = {
        **mae_metadata,
        "a3_random_mrcnn_sha256": a3_digest,
        "a4_mae_mrcnn_sha256": a4_digest,
        "proto_cfg": dict(PROTO_TRIALS[0]),
    }
    return a3, a4, metadata

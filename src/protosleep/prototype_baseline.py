from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import torch

from .attnsleep import AttnSleepBaseline, init_attnsleep_weights
from .config import ENABLE_MICRO_MASK
from .prototypes import ProtoAttnSleep
from .utils import seed_everything


def _state_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        h.update(key.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def build_matched_a1_a3(seed: int) -> Tuple[AttnSleepBaseline, ProtoAttnSleep, Dict[str, Any]]:
    """Build a clean AttnSleep-vs-prototype pair with identical shared initialization.

    A1 is the plain AttnSleep classifier. A3_current is the current spherical prototype
    classifier. Their MRCNN, TCE and classifier parameters are copied exactly at
    initialization, so the only additional initialized state in A3 is the prototype pathway
    itself (prototype bank and beta parameter, plus the optional micro-mask module when
    enabled by configuration).

    Both downstream trainers are subsequently called with the same seed. Because
    ``train_micro_model`` resets all RNGs at entry and fresh loaders are used for each member,
    data shuffling and shared dropout stochasticity are matched as closely as the two
    architectures permit.
    """
    seed = int(seed)

    seed_everything(seed)
    a1 = AttnSleepBaseline()
    a1.apply(init_attnsleep_weights)

    # Re-seed so prototype-specific initialization is deterministic for this experimental seed.
    seed_everything(seed)
    a3 = ProtoAttnSleep(enable_micro_mask=ENABLE_MICRO_MASK)
    a3.apply(init_attnsleep_weights)

    # Explicitly match every parameter/buffer shared by the two architectures.
    a3.mrcnn.load_state_dict(a1.mrcnn.state_dict(), strict=True)
    a3.tce.load_state_dict(a1.tce.state_dict(), strict=True)
    a3.fc.load_state_dict(a1.fc.state_dict(), strict=True)

    shared = {
        "mrcnn": (_state_digest(a1.mrcnn), _state_digest(a3.mrcnn)),
        "tce": (_state_digest(a1.tce), _state_digest(a3.tce)),
        "fc": (_state_digest(a1.fc), _state_digest(a3.fc)),
    }
    bad = [name for name, (left, right) in shared.items() if left != right]
    if bad:
        raise RuntimeError(f"A1/A3 shared initialization mismatch: {bad}")

    metadata: Dict[str, Any] = {
        "seed": seed,
        "shared_initialization_match": True,
        "a1_mrcnn_sha256": shared["mrcnn"][0],
        "a3_mrcnn_sha256": shared["mrcnn"][1],
        "a1_tce_sha256": shared["tce"][0],
        "a3_tce_sha256": shared["tce"][1],
        "a1_fc_sha256": shared["fc"][0],
        "a3_fc_sha256": shared["fc"][1],
        "a3_prototype_bank_sha256": _state_digest(a3.prototype_bank),
        "a3_beta_init": float(torch.sigmoid(a3.beta_logit.detach()).cpu()),
    }
    return a1, a3, metadata

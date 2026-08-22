from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import _canonicalize_epoch_shape
from .mist import extract_mrcnn_state_dict
from .morphmae_bridge import discover_npz_subject_files, fold_subjects_from_npz


LEGACY_RELATIVE_FILES: Dict[str, str] = {
    "protop": "external/WaveSleepNet-main/models/protop.py",
    "trainer": "external/WaveSleepNet-main/train_mtcl.py",
    "loader": "external/WaveSleepNet-main/loader.py",
    "utils": "external/WaveSleepNet-main/utils.py",
    "mae_patch": "morphmae/integrations/wavesleepnet_mae_patch.py",
    "edf20_config": "external/WaveSleepNet-main/configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json",
}

# WaveSleepNet's historical protop.py imports ``models.attnsleep.AttnSleep``, but the
# archived WaveSleepNet tree does not contain models/attnsleep.py. The same recovered
# codebase does bundle the original AttnSleep implementation under external/AttnSleep-main.
# At import time we expose that bundled historical implementation under the module name
# expected by WaveSleepNet. This is an import-compatibility bridge only; protop.py itself
# remains byte-for-byte unchanged and its own MRCNN implementation is still the one used
# by ProtoPNet.
LEGACY_ATTNSLEEP_MODEL = "external/AttnSleep-main/model/model.py"

# Frozen from the 2026-08-22 read-only legacy prototype audit. These hashes are a
# deliberate guardrail: the recovery experiment should fail closed if the historical
# source snapshot changes underneath it.
EXPECTED_SHA256: Dict[str, str] = {
    "protop": "1861dc78b97c6b89f3fbf84943b24cb654845933b3b9409d4aa65d3fb741f538",
    "trainer": "70e534112cdc8658fd8b94f587571f4a3449fb3afdc4251939c42c05989e9a8f",
    "loader": "d2ba0f92aeb109fd424878fde5892316bfdaae6aafa396d98e5b580cbb76520d",
    "utils": "aa4310020720bbb4b9f10ae3e9334dba3cd814613d5557beb034580070be16fc",
    "mae_patch": "65defa9375ee06415215bb15a8d4db2ed7a1f34873fc7ec59682a0bdd4bf558c",
    "edf20_config": "5238092311277fb35ebf85bb7ce313035178d5fc41a1e3673766f0dd84acb564",
}

# Historical Sleep-EDF-2013 WaveSleepNet values recovered by source audit. The only
# intentional architecture adaptation for the MIST recovery is 27 -> 30 channels in
# afr_reduced_dim/prototype_shape so MorphMAE's exact AttnSleep MRCNN can load strictly.
#
# Note that early_stopping is nested under training_params in the historical JSON.
# The original trainer confirms this structure by reading self.es_cfg from the training
# configuration before using self.es_cfg['patience'].
EXPECTED_HISTORICAL_CONFIG: Dict[Tuple[str, ...], Any] = {
    ("classifier", "prototype_num"): 10,
    ("classifier", "prototype_shape"): [10, 27, 1],
    ("classifier", "dist_lambda"): 17.8373,
    ("classifier", "class_lambda"): 50,
    ("classifier", "identity_lambda"): 8.9351,
    ("classifier", "pd_lambda"): 7.9252,
    ("classifier", "weight_lambda"): 0.3,
    ("training_params", "max_epochs"): 5000,
    ("training_params", "batch_size"): 64,
    ("training_params", "lr"): 0.0005,
    ("training_params", "weight_decay"): 0.0001,
    ("training_params", "early_stopping", "patience"): 50,
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_paths(legacy_root: Path | str) -> Dict[str, Path]:
    root = Path(legacy_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {name: root / rel for name, rel in LEGACY_RELATIVE_FILES.items()}


def validate_legacy_snapshot(legacy_root: Path | str, allow_source_drift: bool = False) -> Dict[str, Any]:
    paths = legacy_paths(legacy_root)
    report: Dict[str, Any] = {}
    errors: List[str] = []
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path}")
            continue
        actual = sha256_file(path)
        expected = EXPECTED_SHA256[name]
        report[name] = {"path": str(path), "sha256": actual, "expected_sha256": expected}
        if actual != expected and not allow_source_drift:
            errors.append(f"{name} sha256={actual}, expected {expected}")
    if errors:
        raise RuntimeError(
            "Historical WaveSleepNet snapshot validation failed:\n- " + "\n- ".join(errors)
        )
    return report


def load_historical_edf20_config(legacy_root: Path | str) -> Dict[str, Any]:
    path = legacy_paths(legacy_root)["edf20_config"]
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return cfg


def _nested_config_value(cfg: Mapping[str, Any], path: Sequence[str]) -> Tuple[bool, Any]:
    current: Any = cfg
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def validate_historical_edf20_config(cfg: Mapping[str, Any]) -> None:
    errors: List[str] = []
    for path, expected in EXPECTED_HISTORICAL_CONFIG.items():
        dotted = ".".join(path)
        present, actual = _nested_config_value(cfg, path)
        if not present:
            errors.append(f"missing {dotted}")
            continue
        if isinstance(expected, float):
            try:
                ok = abs(float(actual) - expected) <= 1e-12
            except Exception:
                ok = False
        else:
            ok = actual == expected
        if not ok:
            errors.append(f"{dotted}={actual!r}, expected {expected!r}")
    if errors:
        raise RuntimeError(
            "Historical Sleep-EDF-2013 WaveSleepNet config does not match the audited snapshot:\n- "
            + "\n- ".join(errors)
        )


def patched_mist_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the audited historical config with only the documented 27 -> 30 MRCNN bridge patch."""
    validate_historical_edf20_config(cfg)
    out = copy.deepcopy(dict(cfg))
    classifier = out["classifier"]
    classifier["afr_reduced_dim"] = 30
    shape = list(classifier["prototype_shape"])
    if shape != [10, 27, 1]:
        raise RuntimeError(f"Unexpected historical prototype_shape before patch: {shape}")
    shape[1] = 30
    classifier["prototype_shape"] = shape
    return out


def _pop_module_tree(prefix: str) -> Dict[str, Any]:
    """Temporarily remove an import namespace so legacy imports cannot hit unrelated modules."""
    saved: Dict[str, Any] = {}
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_module_tree(prefix: str, saved: Mapping[str, Any]) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def _install_bundled_attnsleep_alias(legacy_root: Path, wavesleep_root: Path) -> Dict[str, str]:
    """
    Satisfy WaveSleepNet's missing ``models.attnsleep`` import from bundled AttnSleep.

    The archived WaveSleepNet protop.py imports ``from models.attnsleep import AttnSleep``
    but its models directory does not contain attnsleep.py. The recovery codebase includes
    the original AttnSleep model.py, so load that exact historical file under the module name
    WaveSleepNet expects instead of creating a stub or substituting the modern repo model.
    """
    native = wavesleep_root / "models" / "attnsleep.py"
    if native.is_file():
        return {"mode": "wavesleepnet-native", "path": str(native), "sha256": sha256_file(native)}

    source = legacy_root / LEGACY_ATTNSLEEP_MODEL
    if not source.is_file():
        raise ModuleNotFoundError(
            "Historical WaveSleepNet requires models.attnsleep, but the WaveSleepNet snapshot "
            f"has no {native} and bundled AttnSleep source is missing: {source}"
        )

    # Ensure the WaveSleepNet ``models`` package/namespace is the parent package. The caller
    # has already put wavesleep_root first on sys.path and isolated any pre-existing models.*.
    importlib.import_module("models")
    spec = importlib.util.spec_from_file_location("models.attnsleep", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for bundled AttnSleep source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.attnsleep"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("models.attnsleep", None)
        raise
    if not hasattr(module, "AttnSleep"):
        raise AttributeError(f"Bundled historical AttnSleep source has no AttnSleep class: {source}")
    return {"mode": "bundled-attnsleep-alias", "path": str(source), "sha256": sha256_file(source)}


def import_legacy_protop_module(legacy_root: Path | str):
    root = Path(legacy_root).expanduser().resolve()
    paths = legacy_paths(root)
    wavesleep_root = paths["protop"].parents[1]
    attnsleep_root = root / "external" / "AttnSleep-main"
    module_path = paths["protop"]

    # Put the historical repositories first while executing protop.py. Isolate ``models``
    # so an unrelated installed package cannot satisfy WaveSleepNet's absolute imports.
    wave_str = str(wavesleep_root)
    attn_str = str(attnsleep_root)
    old_path = list(sys.path)
    saved_models = _pop_module_tree("models")
    try:
        for path_str in (attn_str, wave_str):
            if path_str in sys.path:
                sys.path.remove(path_str)
        # WaveSleepNet must remain first because protop.py's other ``models.*`` imports are
        # intended to resolve inside its own repository. AttnSleep is second for dependencies.
        sys.path.insert(0, attn_str)
        sys.path.insert(0, wave_str)

        _install_bundled_attnsleep_alias(root, wavesleep_root)

        spec = importlib.util.spec_from_file_location("protosleep_legacy_wavesleepnet_protop", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        _restore_module_tree("models", saved_models)

    if not hasattr(module, "ProtoPNet"):
        raise AttributeError(f"{module_path} does not define ProtoPNet")
    return module


def instantiate_legacy_protopnet(legacy_root: Path | str, cfg: Mapping[str, Any]) -> Tuple[torch.nn.Module, str]:
    module = import_legacy_protop_module(legacy_root)
    cls = module.ProtoPNet
    signature = str(inspect.signature(cls))
    try:
        model = cls(copy.deepcopy(dict(cfg)))
    except TypeError as exc:
        raise RuntimeError(
            f"Audited legacy ProtoPNet could not be instantiated as ProtoPNet(config); signature={signature}"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"ProtoPNet returned {type(model)!r}, expected torch.nn.Module")
    if not hasattr(model, "mrcnn"):
        raise AttributeError("Legacy ProtoPNet has no mrcnn attribute")
    if not hasattr(model, "prototype_vectors"):
        raise AttributeError("Legacy ProtoPNet has no prototype_vectors attribute")
    return model, signature


def state_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode("utf-8"))
        arr = value.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def build_matched_legacy_a3_a4(
    legacy_root: Path | str,
    mae_checkpoint: Path | str,
    seed: int,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    cfg0 = load_historical_edf20_config(legacy_root)
    cfg = patched_mist_config(cfg0)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

    template, signature = instantiate_legacy_protopnet(legacy_root, cfg)
    a3 = copy.deepcopy(template)
    a4 = copy.deepcopy(template)

    random_digest = state_digest(a3.mrcnn)
    mrcnn_state, ckpt_meta = extract_mrcnn_state_dict(mae_checkpoint)
    a4.mrcnn.load_state_dict(mrcnn_state, strict=True)
    mae_digest = state_digest(a4.mrcnn)
    if random_digest == mae_digest:
        raise RuntimeError("MorphMAE MRCNN digest equals the random legacy A3 MRCNN digest")

    s3 = a3.state_dict()
    s4 = a4.state_dict()
    mismatch: List[str] = []
    for key, value in s3.items():
        if key.startswith("mrcnn."):
            continue
        if key not in s4 or not torch.equal(value.detach().cpu(), s4[key].detach().cpu()):
            mismatch.append(key)
    if mismatch:
        raise RuntimeError(f"Legacy A3/A4 non-MRCNN initialization mismatch: {mismatch[:10]}")

    proto_shape = list(a3.prototype_vectors.shape)
    if proto_shape != [10, 30, 1]:
        raise RuntimeError(f"Patched legacy prototype_vectors shape is {proto_shape}, expected [10, 30, 1]")

    return a3, a4, {
        "constructor_signature": signature,
        "patched_config": cfg,
        "a3_random_mrcnn_sha256": random_digest,
        "a4_mae_mrcnn_sha256": mae_digest,
        "mae_checkpoint": ckpt_meta,
        "prototype_shape": proto_shape,
    }


def sample_train_only_fold_batch(
    data_dir: Path | str,
    fold: int,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    split = fold_subjects_from_npz(data_dir, fold)
    by_subject = discover_npz_subject_files(data_dir)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    used_files: List[str] = []
    remaining = int(batch_size)
    for sid in split["train_subjects"]:
        for path in by_subject[int(sid)]:
            with np.load(path, allow_pickle=False) as d:
                if "x" not in d.files or "y" not in d.files:
                    raise KeyError(f"{path.name}: expected x and y arrays")
                x = _canonicalize_epoch_shape(d["x"], path.name)
                y = np.asarray(d["y"]).reshape(-1).astype(np.int64, copy=False)
            take = min(remaining, int(y.shape[0]))
            if take:
                xs.append(x[:take])
                ys.append(y[:take])
                used_files.append(str(path))
                remaining -= take
            if remaining == 0:
                break
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"Could only collect {batch_size - remaining}/{batch_size} train epochs")

    x_batch = torch.from_numpy(np.concatenate(xs, axis=0).astype(np.float32, copy=False))
    y_batch = torch.from_numpy(np.concatenate(ys, axis=0).astype(np.int64, copy=False))
    return x_batch, y_batch, {
        "fold": int(fold),
        "train_subjects": split["train_subjects"],
        "val_subjects": split["val_subjects"],
        "test_subjects": split["test_subjects"],
        "used_files": used_files,
    }


def tensor_tree_shapes(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return [tensor_tree_shapes(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): tensor_tree_shapes(v) for k, v in obj.items()}
    return {"type": type(obj).__name__}


def tensor_leaves(obj: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from tensor_leaves(value)
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from tensor_leaves(value)

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from .config import ATTNSLEEP_R_PERMUTE_20


EXPECTED_V2 = {
    ("mask", "patch_size"): 25,
    ("mask", "n_patches"): 120,
    ("mask", "min_span"): 2,
    ("mask", "max_span"): 24,
    ("mask_schedule", "start"): 0.50,
    ("mask_schedule", "mid"): 0.65,
    ("mask_schedule", "final"): 0.75,
    ("loss", "w_time"): 1.0,
    ("loss", "w_stft"): 0.5,
    ("loss", "w_diff"): 0.3,
    ("loss", "w_band"): 0.15,
    ("loss", "fs"): 100,
    ("train", "epochs"): 100,
    ("train", "batch_size"): 128,
    ("train", "lr"): 0.0002,
    ("train", "weight_decay"): 0.01,
    ("train", "amp"): False,
}


def sleep_edf_subject_id_from_name(name: str) -> int:
    """Match the project-wide Sleep-EDF subject rule: filename[3:5]."""
    base = Path(name).name
    if len(base) >= 5 and base[3:5].isdigit():
        return int(base[3:5])
    raise ValueError(f"Cannot parse Sleep-EDF subject ID from {base!r} using filename[3:5]")


def discover_npz_subject_files(data_dir: Path | str) -> Dict[int, List[Path]]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files under {root}")
    out: Dict[int, List[Path]] = {}
    for path in paths:
        sid = sleep_edf_subject_id_from_name(path.name)
        out.setdefault(sid, []).append(path)
    return out


def fold_subjects_from_npz(data_dir: Path | str, fold: int) -> Dict[str, List[int]]:
    """
    Resolve the same 20-subject permutation as the main Proto-sleep runner without
    opening any NPZ signal/label arrays. Only filenames are inspected.
    """
    by_subject = discover_npz_subject_files(data_dir)
    subjects = sorted(by_subject)
    if subjects != list(range(20)):
        raise ValueError(
            "Fold-specific MorphMAE pretraining is currently locked to the verified "
            f"Sleep-EDF-20 subject set 0..19; found {subjects}"
        )
    if not 0 <= fold < 20:
        raise ValueError("fold must be in 0..19")

    order = [subjects[i] for i in ATTNSLEEP_R_PERMUTE_20]
    test_subject = order[fold]
    val_subject = order[(fold + 1) % len(order)]
    train_subjects = sorted(s for s in order if s not in {test_subject, val_subject})
    return {
        "order": order,
        "train_subjects": train_subjects,
        "val_subjects": [val_subject],
        "test_subjects": [test_subject],
    }


def create_train_only_npz_view(
    data_dir: Path | str,
    view_dir: Path | str,
    train_subjects: Sequence[int],
) -> List[Path]:
    """
    Create a symlink-only dataset view containing *only* the permitted SSL subjects.

    This is stronger than relying on a legacy `exclude_subject_ids` option: the old
    MorphMAE process cannot open validation/test recordings because those paths are
    absent from its configured NPZ root.
    """
    source = discover_npz_subject_files(data_dir)
    allowed = set(int(x) for x in train_subjects)
    missing = sorted(allowed - set(source))
    if missing:
        raise RuntimeError(f"Missing train subjects in source NPZ directory: {missing}")

    view = Path(view_dir).expanduser().resolve()
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True, exist_ok=False)

    linked: List[Path] = []
    for sid in sorted(allowed):
        for src in source[sid]:
            dst = view / src.name
            dst.symlink_to(src.resolve())
            linked.append(dst)

    actual_subjects = sorted({sleep_edf_subject_id_from_name(p.name) for p in linked})
    if actual_subjects != sorted(allowed):
        raise RuntimeError(
            f"Train-only view subject mismatch: expected {sorted(allowed)}, got {actual_subjects}"
        )
    return linked


def _nested_get(mapping: Dict[str, Any], path: Tuple[str, str]) -> Any:
    a, b = path
    if a not in mapping or not isinstance(mapping[a], dict) or b not in mapping[a]:
        raise KeyError(f"Missing historical MorphMAE-v2 config key {a}.{b}")
    return mapping[a][b]


def validate_historical_v2_config(cfg: Dict[str, Any]) -> None:
    """Fail closed if the supplied legacy YAML is not the validated MorphMAE-v2 recipe."""
    errors = []
    for path, expected in EXPECTED_V2.items():
        try:
            actual = _nested_get(cfg, path)
        except KeyError as exc:
            errors.append(str(exc))
            continue
        if isinstance(expected, float):
            try:
                ok = abs(float(actual) - expected) <= 1e-12
            except Exception:
                ok = False
        else:
            ok = actual == expected
        if not ok:
            errors.append(f"{path[0]}.{path[1]}={actual!r}, expected {expected!r}")
    if errors:
        raise ValueError(
            "Legacy config does not match the frozen MorphMAE-v2 signature:\n- "
            + "\n- ".join(errors)
        )


def load_prepare_v2_config(
    base_config: Path | str,
    train_view: Path | str,
    legacy_output_dir: Path | str,
    seed: int,
) -> Dict[str, Any]:
    base = Path(base_config).expanduser().resolve()
    with base.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping YAML in {base}")
    validate_historical_v2_config(cfg)

    cfg = json.loads(json.dumps(cfg))  # plain deep copy with deterministic scalar types
    cfg["seed"] = int(seed)
    cfg["output_dir"] = str(Path(legacy_output_dir).expanduser().resolve())
    cfg.setdefault("data", {})
    cfg["data"]["npz_root"] = str(Path(train_view).expanduser().resolve())
    # The physical train-only view is the leakage barrier; keep the legacy exclusion list empty.
    cfg["data"]["exclude_subject_ids"] = []
    return cfg


def write_yaml(cfg: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_legacy_source_tree(legacy_root: Path | str) -> str:
    """Hash the legacy MorphMAE Python source plus YAML configs used to define the recipe."""
    root = Path(legacy_root).expanduser().resolve()
    files: List[Path] = []
    for base in (root / "morphmae", root / "configs"):
        if not base.exists():
            continue
        files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml"})
    if not files:
        raise FileNotFoundError(f"No MorphMAE source/config files found under {root}")

    h = hashlib.sha256()
    for path in sorted(files):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def discover_legacy_pretrain_shell(legacy_root: Path | str) -> Path:
    """Find the historical shell launcher, preferring one explicitly tied to v2."""
    root = Path(legacy_root).expanduser().resolve()
    candidates = []
    for path in sorted((root / "scripts").glob("*.sh")):
        text = path.read_text(encoding="utf-8", errors="replace")
        low = text.lower()
        if "pretrain" not in low or "mae" not in low:
            continue
        score = 0
        if "mae_npz_edf78_v2.yaml" in low:
            score += 100
        if "pretrain_mae" in low:
            score += 20
        if "edf78" in low:
            score += 5
        candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError(f"Could not find a legacy MorphMAE pretraining shell script under {root / 'scripts'}")
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return candidates[0][1]


_CONFIG_RE = re.compile(r"configs/mae_npz_[A-Za-z0-9_\-]+\.ya?ml")


def render_fold_launcher(
    legacy_root: Path | str,
    generated_config: Path | str,
    destination: Path | str,
) -> Tuple[Path, Path]:
    """
    Copy the historical shell launcher and replace only its MAE YAML path.

    The copied launcher is stored with the run provenance. The original legacy codebase is
    never modified. We assert that the generated config path is actually present before run.
    """
    source_script = discover_legacy_pretrain_shell(legacy_root)
    text = source_script.read_text(encoding="utf-8")
    generated = str(Path(generated_config).expanduser().resolve())
    replaced, n = _CONFIG_RE.subn(generated, text)
    if n == 0:
        raise RuntimeError(
            f"Found legacy pretrain launcher {source_script}, but could not locate its configs/mae_npz_*.yaml token"
        )
    if generated not in replaced:
        raise RuntimeError("Generated fold config was not injected into copied legacy launcher")

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(replaced, encoding="utf-8")
    destination.chmod(0o755)
    return source_script, destination

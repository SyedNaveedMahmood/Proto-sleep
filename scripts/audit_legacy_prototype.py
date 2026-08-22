#!/usr/bin/env python3
"""Read-only audit of the historical WaveSleepNet prototype pathway used by MIST.

The current repository's ``ProtoAttnSleep`` is not claimed to be byte-for-byte identical to
that historical implementation. Before porting a historical A3/A4 branch, this utility
extracts the exact prototype architecture, objective terms, trainer settings, split logic,
and MAE/MRCNN bridge evidence from the legacy source tree.

No training is started and no legacy file is modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".sh", ".md", ".txt", ".cfg", ".ini"}

EXPECTED_FILES = (
    "external/WaveSleepNet-main/models/protop.py",
    "external/WaveSleepNet-main/train_mtcl.py",
    "external/WaveSleepNet-main/loader.py",
    "external/WaveSleepNet-main/utils.py",
    "morphmae/integrations/wavesleepnet_mae_patch.py",
)

CATEGORY_PATTERNS: Dict[str, Sequence[str]] = {
    "architecture": (
        r"\bclass\s+Proto",
        r"prototype_vectors",
        r"gate_conv",
        r"wave_similarity",
        r"min_dist",
        r"\bMRCNN\b",
        r"\bAFR\b",
        r"prototype",
    ),
    "objective": (
        r"protop_loss",
        r"dist_loss",
        r"pd_loss",
        r"identity_loss",
        r"weight_loss",
        r"diversity",
        r"min_dist",
        r"fc_weight",
        r"loss_cfg",
        r"lambda",
    ),
    "initialization": (
        r"kmeans",
        r"k-means",
        r"prototype.*init",
        r"init.*prototype",
        r"load_path",
        r"load_state_dict",
        r"pretrain",
        r"mae",
    ),
    "training": (
        r"optimizer",
        r"AdamW?",
        r"\blr\b",
        r"learning_rate",
        r"batch_size",
        r"epochs?",
        r"patience",
        r"EarlyStopping",
        r"weight_decay",
        r"scheduler",
        r"backward\(",
    ),
    "splits": (
        r"fold",
        r"split",
        r"train",
        r"val",
        r"test",
        r"subject",
        r"Sleep-EDF",
        r"SHHS",
    ),
    "checkpoint_transfer": (
        r"checkpoint",
        r"torch\.save",
        r"torch\.load",
        r"state_dict",
        r"strict",
        r"mrcnn",
        r"MorphMAE",
        r"wavesleepnet_mae_patch",
    ),
}

CONFIG_ACCESS_RE = re.compile(
    r"(?:self\.)?(?:cfg|config)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 3_000_000:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def symbols_from_python(path: Path, root: Path) -> List[Dict[str, Any]]:
    text = read_text(path)
    if text is None:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    rows: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            rows.append({"file": rel(path, root), "line": node.lineno, "kind": "class", "name": node.name})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append({"file": rel(path, root), "line": node.lineno, "kind": "function", "name": node.name})
    return sorted(rows, key=lambda x: (x["line"], x["kind"], x["name"]))


def context_snippet(lines: List[str], line_no: int, radius: int = 2) -> List[Dict[str, Any]]:
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    return [
        {"line": idx, "text": lines[idx - 1].rstrip()[:500]}
        for idx in range(start, end + 1)
    ]


def audit(root: Path) -> Dict[str, Any]:
    files = list(iter_text_files(root))
    matches: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CATEGORY_PATTERNS}
    ranked: List[Dict[str, Any]] = []
    symbols: List[Dict[str, Any]] = []
    config_keys: Dict[str, List[str]] = {}

    for path in files:
        text = read_text(path)
        if text is None:
            continue
        lines = text.splitlines()
        file_counts: Dict[str, int] = {}

        for category, patterns in CATEGORY_PATTERNS.items():
            hit_lines: List[int] = []
            for idx, line in enumerate(lines, start=1):
                if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
                    hit_lines.append(idx)
            file_counts[category] = len(hit_lines)
            for idx in hit_lines[:80]:
                if len(matches[category]) >= 250:
                    break
                matches[category].append(
                    {
                        "file": rel(path, root),
                        "line": idx,
                        "text": lines[idx - 1].strip()[:500],
                        "context": context_snippet(lines, idx),
                    }
                )

        score = sum(file_counts.values())
        if score:
            ranked.append({"file": rel(path, root), "score": score, **file_counts})

        if path.suffix.lower() == ".py":
            symbols.extend(symbols_from_python(path, root))
            keys = sorted(set(CONFIG_ACCESS_RE.findall(text)))
            if keys:
                config_keys[rel(path, root)] = keys

    ranked.sort(key=lambda row: (-int(row["score"]), str(row["file"])))

    expected = []
    for name in EXPECTED_FILES:
        path = root / name
        expected.append(
            {
                "file": name,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )

    prototype_configs = []
    cfg_root = root / "external" / "WaveSleepNet-main" / "configs"
    if cfg_root.is_dir():
        for path in sorted(cfg_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
                prototype_configs.append(
                    {"file": rel(path, root), "sha256": sha256_file(path)}
                )

    return {
        "root": str(root),
        "n_text_files": len(files),
        "expected_files": expected,
        "prototype_configs": prototype_configs,
        "top_files": ranked[:40],
        "symbols": symbols,
        "config_keys": config_keys,
        "matches": matches,
    }


def render(report: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("=" * 100)
    out.append("LEGACY WAVESLEEPNET PROTOTYPE AUDIT")
    out.append("=" * 100)
    out.append(f"codebase: {report['root']}")
    out.append(f"text files scanned: {report['n_text_files']}")

    out.append("\nEXPECTED HISTORICAL FILES")
    out.append("-" * 100)
    for row in report["expected_files"]:
        out.append(json.dumps(row, sort_keys=True))

    out.append("\nWAVESLEEPNET CONFIG FILES")
    out.append("-" * 100)
    for row in report["prototype_configs"]:
        out.append(json.dumps(row, sort_keys=True))

    out.append("\nTOP PROTOTYPE-RELEVANT FILES")
    out.append("-" * 100)
    for row in report["top_files"]:
        out.append(json.dumps(row, sort_keys=True))

    out.append("\nPYTHON CLASS / FUNCTION SYMBOLS IN HIGH-VALUE FILES")
    out.append("-" * 100)
    high_value = {row["file"] for row in report["top_files"][:12]}
    for row in report["symbols"]:
        if row["file"] in high_value:
            out.append(f"{row['file']}:{row['line']} | {row['kind']} {row['name']}")

    out.append("\nCONFIG KEYS ACCESSED BY SOURCE")
    out.append("-" * 100)
    for file_name, keys in sorted(report["config_keys"].items()):
        if file_name in high_value or "WaveSleepNet-main" in file_name:
            out.append(f"{file_name} | {', '.join(keys)}")

    for category, rows in report["matches"].items():
        out.append(f"\n{category.upper()} MATCHES")
        out.append("-" * 100)
        for row in rows:
            out.append(f"{row['file']}:{row['line']} | {row['text']}")

    out.append("\n" + "=" * 100)
    out.append("END LEGACY WAVESLEEPNET PROTOTYPE AUDIT")
    out.append("=" * 100)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("codebase", help="Historical MorphMAE_Sleep_Codebase root")
    p.add_argument("--report", default="legacy_prototype_audit.txt")
    p.add_argument("--json", default="legacy_prototype_audit.json")
    args = p.parse_args()

    root = Path(args.codebase).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    report = audit(root)
    rendered = render(report)

    report_path = Path(args.report).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(rendered)
    print(f"\nSaved text report: {report_path}")
    print(f"Saved JSON report: {json_path}")
    print("\nREAD-ONLY AUDIT: no legacy files modified; no training started.")


if __name__ == "__main__":
    main()

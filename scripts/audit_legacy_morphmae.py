#!/usr/bin/env python3
"""Audit the historical MorphMAE codebase before porting fold-specific pretraining.

This utility is intentionally read-only. It does not train a model and it does not modify
legacy files. Its job is to recover the exact historical implementation details that are not
fully recorded in the project report (optimizer, epochs, batch size, masking helpers, loss
implementations, decoder layout, checkpoint schema, etc.) so that the new repository does
not silently invent a "MorphMAE v2" implementation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".txt", ".md"
}

CATEGORY_PATTERNS: Dict[str, Sequence[str]] = {
    "architecture": (
        r"\bMorphMAE\b",
        r"\bMRCNN\b",
        r"\bAFR\b",
        r"decoder",
        r"ConvTranspose1d",
        r"Conv1d",
    ),
    "masking": (
        r"mask[_ ]?ratio",
        r"mask[_ ]?schedule",
        r"patch[_ ]?size",
        r"span",
        r"masked",
    ),
    "losses": (
        r"stft",
        r"spectral",
        r"band[_ ]?power",
        r"band[_ ]?loss",
        r"derivative",
        r"diff[_ ]?loss",
        r"wave[_ ]?loss",
        r"time[_ ]?loss",
    ),
    "training": (
        r"AdamW?",
        r"learning[_ ]?rate",
        r"\blr\b",
        r"epochs?",
        r"batch[_ ]?size",
        r"patience",
        r"weight[_ ]?decay",
        r"scheduler",
        r"GradScaler",
        r"autocast",
    ),
    "splits": (
        r"subject",
        r"split",
        r"train[_ ]?subjects",
        r"val[_ ]?subjects",
        r"test[_ ]?subjects",
        r"Sleep[-_ ]?EDF",
        r"EDF[-_ ]?78",
        r"EDF[-_ ]?20",
    ),
    "checkpointing": (
        r"torch\.save",
        r"torch\.load",
        r"best_morphmae",
        r"checkpoint",
        r"state_dict",
    ),
}

ASSIGNMENT_NAME = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$"
)

INTERESTING_ASSIGNMENT = re.compile(
    r"(epoch|batch|lr|learning|weight_decay|patch|mask|span|stft|spec|band|diff|deriv|"
    r"wave|time|loss|lambda|seed|worker|patience|warmup|decoder|channel|hidden)",
    re.IGNORECASE,
)


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 2_000_000:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _summarize_obj(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    if torch.is_tensor(obj):
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, Mapping):
        if depth >= max_depth:
            return {"type": "mapping", "keys": [str(k) for k in obj.keys()]}
        return {
            str(k): _summarize_obj(v, depth + 1, max_depth)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        if len(obj) <= 20 and all(
            isinstance(x, (str, int, float, bool, type(None))) for x in obj
        ):
            return list(obj)
        if depth >= max_depth:
            return {"type": type(obj).__name__, "length": len(obj)}
        return {
            "type": type(obj).__name__,
            "length": len(obj),
            "items": [_summarize_obj(x, depth + 1, max_depth) for x in obj[:10]],
        }
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return {"type": type(obj).__name__, "repr": repr(obj)[:200]}


def _tensor_branch_stats(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(obj, Mapping):
        return rows

    tensor_values = {
        str(k): v for k, v in obj.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }
    if tensor_values:
        n_params = int(sum(int(v.numel()) for v in tensor_values.values()))
        sample = [
            {
                "key": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
            for k, v in list(tensor_values.items())[:12]
        ]
        rows.append(
            {
                "branch": prefix or "<root>",
                "n_tensors": len(tensor_values),
                "n_values": n_params,
                "sample": sample,
            }
        )

    for key, value in obj.items():
        if isinstance(value, Mapping):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_tensor_branch_stats(value, child))
    return rows


def audit_source(root: Path) -> Dict[str, Any]:
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )

    ranked: List[Tuple[int, str, Dict[str, int]]] = []
    assignments: List[Dict[str, str]] = []
    matches: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CATEGORY_PATTERNS}

    for path in files:
        text = _read_text(path)
        if text is None:
            continue
        rel = _relative(path, root)
        lines = text.splitlines()
        category_counts: Dict[str, int] = {}

        for category, patterns in CATEGORY_PATTERNS.items():
            count = 0
            for line_no, line in enumerate(lines, start=1):
                if any(re.search(pat, line, flags=re.IGNORECASE) for pat in patterns):
                    count += 1
                    if len(matches[category]) < 80:
                        matches[category].append(
                            {
                                "file": rel,
                                "line": line_no,
                                "text": line.strip()[:300],
                            }
                        )
            category_counts[category] = count

        score = sum(category_counts.values())
        if score:
            ranked.append((score, rel, category_counts))

        for line_no, line in enumerate(lines, start=1):
            m = ASSIGNMENT_NAME.match(line)
            if not m:
                continue
            name, value = m.groups()
            if not INTERESTING_ASSIGNMENT.search(name):
                continue
            assignments.append(
                {
                    "file": rel,
                    "line": str(line_no),
                    "name": name,
                    "value": value[:500],
                }
            )

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return {
        "root": str(root),
        "n_text_files": len(files),
        "top_files": [
            {"score": score, "file": rel, **counts}
            for score, rel, counts in ranked[:30]
        ],
        "candidate_assignments": assignments[:300],
        "matches": matches,
    }


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("LEGACY MORPHMAE AUDIT")
    lines.append("=" * 100)
    lines.append(f"codebase: {report['source']['root']}")
    lines.append(f"text files scanned: {report['source']['n_text_files']}")

    if report.get("checkpoint"):
        ck = report["checkpoint"]
        lines.append(f"checkpoint: {ck['path']}")
        lines.append("checkpoint top-level schema:")
        lines.append(json.dumps(ck["schema"], indent=2, default=str))
        lines.append("checkpoint tensor branches:")
        lines.append(json.dumps(ck["tensor_branches"], indent=2, default=str))

    lines.append("\nTOP SOURCE/CONFIG FILES BY RELEVANT KEYWORD HITS")
    lines.append("-" * 100)
    for row in report["source"]["top_files"]:
        lines.append(json.dumps(row, sort_keys=True))

    lines.append("\nCANDIDATE HYPERPARAMETER / CONFIG ASSIGNMENTS")
    lines.append("-" * 100)
    for row in report["source"]["candidate_assignments"]:
        lines.append(
            f"{row['file']}:{row['line']} | {row['name']} = {row['value']}"
        )

    for category, rows in report["source"]["matches"].items():
        lines.append(f"\n{category.upper()} MATCHES")
        lines.append("-" * 100)
        for row in rows:
            lines.append(f"{row['file']}:{row['line']} | {row['text']}")

    lines.append("\n" + "=" * 100)
    lines.append("END LEGACY MORPHMAE AUDIT")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("codebase", help="Historical MorphMAE_Sleep_Codebase directory")
    p.add_argument("--checkpoint", default=None, help="Optional historical .pt checkpoint")
    p.add_argument(
        "--report",
        default="./legacy_morphmae_audit.txt",
        help="Text report destination (default: ./legacy_morphmae_audit.txt)",
    )
    p.add_argument(
        "--json",
        default=None,
        help="Optional JSON report destination",
    )
    args = p.parse_args()

    root = Path(args.codebase).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    report: Dict[str, Any] = {"source": audit_source(root)}

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _safe_load_checkpoint(checkpoint_path)
        report["checkpoint"] = {
            "path": str(checkpoint_path),
            "schema": _summarize_obj(checkpoint, max_depth=2),
            "tensor_branches": _tensor_branch_stats(checkpoint),
        }

    rendered = render_text(report)
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")

    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(rendered)
    print(f"\nSaved text report: {report_path}")
    if args.json:
        print(f"Saved JSON report: {Path(args.json).expanduser().resolve()}")
    print("\nREAD-ONLY AUDIT: no legacy files modified; no training started.")


if __name__ == "__main__":
    main()

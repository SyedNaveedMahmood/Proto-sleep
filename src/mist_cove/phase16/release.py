"""Verify the public Phase XVI package without enumerating private runs."""

from hashlib import sha256
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "reports/mist_cove/phase16_v1/publication_manifest.json"


def public_paths() -> list[Path]:
    paths = []
    paths.extend((ROOT / "docs/mist_cove/phase16").glob("*.md"))
    paths.extend((ROOT / "src/mist_cove/phase16").glob("*.py"))
    paths.append(ROOT / "tests/mist_cove/test_phase16.py")
    report = ROOT / "reports/mist_cove/phase16_v1"
    paths.extend(report.glob("*.json"))
    paths.append(report / "DASHBOARD.md")
    excluded = {"progress.json", "publication_manifest.json"}
    result = sorted(p for p in set(paths) if p.is_file() and p.name not in excluded)
    if any(p.is_symlink() for p in result):
        raise ValueError("public release contains symlink")
    return result


def create_manifest() -> dict:
    files = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in public_paths()}
    value = {"schema": "mist-cove-phase16-publication-v1", "files_sha256": files,
             "file_count": len(files),
             "excluded_mutable": ["reports/mist_cove/phase16_v1/progress.json"],
             "excluded_self": "reports/mist_cove/phase16_v1/publication_manifest.json",
             "no_raw_EEG_or_annotation_arrays": True,
             "DREAMS_archives_downloaded": 0,
             "real_human_event_benchmark": "NOT_TESTED_RIGHTS_BLOCKED"}
    MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    return value


def verify_manifest() -> int:
    value = json.loads(MANIFEST.read_text())
    expected = value["files_sha256"]
    actual = {str(p.relative_to(ROOT)) for p in public_paths()}
    if set(expected) != actual:
        raise ValueError("Phase XVI public release file set differs")
    for name, digest in expected.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Phase XVI public release hash differs: " + name)
    return len(expected)

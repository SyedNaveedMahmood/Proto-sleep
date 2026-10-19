"""Read-only Gate VII-0 inspection. Never deserialize or reseal old weights.

The completed Phase VI receipts authenticate metrics and code, but contain no
weight digests. Hashes observed here are inventory observations, not historical
checkpoint seals. Consequently this version cannot unlock scientific execution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = "8ec83fed182f4ca69889db17272eec84d05e9771"
PRIVATE_RUN = "mist_transfer_runs/mist_cove/phase6_v1/confirmatory_v1"
OUTPUT_ROOT = "mist_transfer_runs/mist_cove/phase7_v1"
CONFIG = "configs/mist_cove_phase7_synthetic_v1.json"
MANIFESTS = (
    "experiments/mist_cove/phase6_preregistration_v1.json",
    "experiments/mist_cove/phase6_execution_manifest_v1.json",
    "reports/mist_cove/phase6_v1/publication_manifest_v1.json",
)
TASKS = ("single", "multiple")
SEEDS = (123, 456, 789)


class IntegrityError(ValueError):
    """An integrity prerequisite failed; downstream execution must stop."""


def require_scope(scope):
    if scope != "SYNTHETIC_ONLY":
        raise IntegrityError("synthetic-only scope required before filesystem access")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def confined_file(root, relative):
    """Reject absolute/traversal paths and symlinks, including ancestor symlinks."""
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise IntegrityError("invalid confined path")
    current = Path(root)
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise IntegrityError("symlink forbidden in integrity input")
    return current


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def verified_bytes(root, relative, expected):
    """Check a plain file against an already trusted digest before parsing it."""
    data = confined_file(root, relative).read_bytes()
    if sha256(data) != expected:
        raise IntegrityError("digest mismatch: " + relative)
    return data


def snapshot(root):
    """Anchor manifests in the specified completed Git commit, not mutable JSON."""
    git(root, "merge-base", "--is-ancestor", BASE, "HEAD")
    changes = git(root, "diff", "--name-status", "--no-renames", BASE, "--").decode()
    if any(line.split("\t", 1)[0] != "A" for line in changes.splitlines()):
        raise IntegrityError("historical tracked files changed since Phase VI")
    files = {}
    for manifest in MANIFESTS:
        anchor = git(root, "show", f"{BASE}:{manifest}")
        verified_bytes(root, manifest, sha256(anchor))
        files[manifest] = sha256(anchor)
        for relative, expected in json.loads(anchor)["files"].items():
            # A manifest must agree with both the completed Git tree and disk.
            if sha256(git(root, "show", f"{BASE}:{relative}")) != expected:
                raise IntegrityError("manifest disagrees with completed Git tree")
            verified_bytes(root, relative, expected)
            files[relative] = expected
    # Check the repaired CI environment without modifying it.
    for name in (".github/workflows/ci.yml", "pyproject.toml",
                 "configs/ci_cpu_constraints_v1.txt"):
        expected = sha256(git(root, "show", f"{BASE}:{name}"))
        verified_bytes(root, name, expected)
        files[name] = expected
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"status": "PASS", "files": files, "inventory_sha256": sha256(encoded)}


def verify_results(root, execution):
    """Authenticate only the result JSON. This does not authenticate weights."""
    data = verified_bytes(root, PRIVATE_RUN + "/results.json",
                          execution["private_receipt_sha256"])
    report = json.loads(data)
    rows = report["rows"]
    if (report["fits"] != 72 or len(rows) != 72
            or sum(r["training"]["optimizer_updates"] for r in rows) != 21600
            or sum(r["training"]["example_exposures"] for r in rows) != 345600
            or report["gates"]["VI-3"]["status"] != "FAIL"
            or report["gates"]["VI-4"]["optimizer_updates"] != 0):
        raise IntegrityError("completed Phase VI accounting disagrees with contract")
    full = [r for r in rows if r["stage"] == "VI-3" and r["method"] == "full"]
    if sorted((r["task"], r["seed"]) for r in full) != sorted(
            (t, s) for t in TASKS for s in SEEDS):
        raise IntegrityError("six historical full M1 rows required")
    return {"status": "PASS", "sha256": sha256(data), "fits": len(rows),
            "optimizer_updates": 21600, "example_exposures": 345600,
            "historical_robustness": "FAIL", "historical_head_updates": 0}


def checkpoint_inventory(root):
    """Observe bytes; never promote a new hash or sidecar into old provenance.

    No completed Phase VI manifest binds these weights. Even a sidecar found
    today cannot establish that it existed at the original execution boundary.
    Recovery needs independent historical evidence and a separately reviewed
    versioned verifier, not a new local seal generated by this program.
    """
    records = []
    for task in TASKS:
        for seed in SEEDS:
            name = f"{task}_{seed}_full.pt"
            relative = PRIVATE_RUN + "/" + name
            path = confined_file(root, relative)
            present = path.is_file()
            data = path.read_bytes() if present else None
            sidecars = []
            for candidate in (path.with_suffix(".seal.json"), Path(str(path) + ".seal.json"),
                              path.with_suffix(".sha256"), Path(str(path) + ".sha256")):
                candidate = confined_file(root, candidate.relative_to(root).as_posix())
                if candidate.exists():
                    sidecars.append(candidate.name)
            records.append({
                "task": task, "seed": seed, "checkpoint": name,
                "present": present, "size_bytes": len(data) if present else None,
                "observed_sha256": sha256(data) if present else None,
                "digest_is_historical_seal": False,
                "sidecar_names_present": sidecars,
                "status": "BLOCKED",
                "reason": ("NO_TRUSTED_HISTORICAL_WEIGHT_DIGEST" if present
                           else "CHECKPOINT_MISSING"),
            })
    return records


def inspect(root=ROOT, *, scope="SYNTHETIC_ONLY"):
    require_scope(scope)  # Must precede Path(), Git, stat, reads or enumeration.
    started = time.perf_counter()
    root = Path(root)
    config = json.loads(confined_file(root, CONFIG).read_text())
    if (config["scope"] != scope or config["mode"] != "integrity_only"
            or config["base_commit"] != BASE
            or config["allowed_fits"] != 0 or config["allowed_optimizer_updates"] != 0
            or config["real_data_authorized"] is not False):
        raise IntegrityError("integrity-only zero-fit configuration required")
    proof = snapshot(root)
    execution = json.loads((root / "reports/mist_cove/phase6_v1/execution_receipt_v1.json").read_bytes())
    blockers = []
    try:
        results = verify_results(root, execution)
    except (OSError, ValueError, KeyError) as error:
        results = {"status": "BLOCKED", "reason": str(error)}
        blockers.append("PRIVATE_RESULT_PROOF_UNAVAILABLE_OR_INVALID")
    checkpoints = checkpoint_inventory(root)
    blockers.extend(sorted({r["reason"] for r in checkpoints}))
    return {
        "version": "phase7-integrity-v1", "status": "BLOCKED",
        "base_commit": BASE, "inspection_commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "branch", "--show-current").decode().strip(),
        "config_sha256": sha256((root / CONFIG).read_bytes()),
        "historical_snapshot": proof, "historical_private_results": results,
        "historical_m1_checkpoints": checkpoints, "blockers": blockers,
        "gates": {"VII-0": "BLOCKED", **{f"VII-{i}": "NOT TESTED" for i in range(1, 7)}},
        "checkpoint_deserializations": 0, "scientific_fits": 0,
        "optimizer_updates": 0, "example_exposures": 0,
        "real_data_accessed": False, "cuda_model_operations": 0,
        "inspection_wall_seconds": time.perf_counter() - started,
    }


def write_receipt(root, relative, receipt, *, scope="SYNTHETIC_ONLY"):
    """Atomically publish a new ignored receipt without overwriting any file."""
    require_scope(scope)
    rel = PurePosixPath(relative)
    if not rel.is_relative_to(OUTPUT_ROOT):
        raise IntegrityError("receipt must be under ignored Phase VII output root")
    target = confined_file(root, relative)
    if target.exists():
        raise FileExistsError("preserve existing receipt")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, allow_nan=False).encode() + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes complete bytes atomically and fails if another
        # writer got here first; os.replace would overwrite that writer's file.
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)

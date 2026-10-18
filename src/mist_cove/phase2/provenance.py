"""Explicit code/config identities; no dataset discovery or source traversal."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_snapshot():
    # Only source directories; never enumerate local runs, manifests or datasets.
    paths = []
    for directory in ("src/mist_cove", "src/mist_transfer", "src/mist_evidence", "src/protosleep"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "scripts").glob("*mist_cove*.py"))
    files = {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)}
    return {"files": files, "sha256": digest(files)}


def environment(device="cpu"):
    result = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "device": str(device), "cuda_build": torch.version.cuda,
              "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "packages": {name: metadata.version(name) for name in ("numpy", "torch", "scikit-learn", "pyEDFlib")}}
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                      gpu_capability=[props.major, props.minor], cudnn=torch.backends.cudnn.version())
    return result


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verify_preregistration():
    path = ROOT / "experiments/mist_cove/phase2_preregistration_v1.json"
    proof = json.loads(path.read_text())
    for name, expected in proof["sha256"].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError("Phase II preregistration hash mismatch: " + name)
    return proof

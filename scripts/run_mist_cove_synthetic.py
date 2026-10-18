#!/usr/bin/env python
"""Run only in-memory synthetic experiments; never imports data loaders."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mist_cove.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mist_cove_synthetic_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "mist_transfer_runs/mist_cove/synthetic_v1/results.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an experiment receipt; choose another output")
    config_bytes = args.config.read_bytes()
    report = run(json.loads(config_bytes), progress=lambda line: print(line, flush=True))
    report["config_file_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    report["implementation_base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = sorted((ROOT / "src/mist_cove").glob("*.py")) + [Path(__file__).resolve()]
    report["implementation_file_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Synthetic receipt: {args.output}")


if __name__ == "__main__":
    main()

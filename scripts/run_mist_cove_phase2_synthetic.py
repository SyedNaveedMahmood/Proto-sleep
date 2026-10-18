#!/usr/bin/env python
"""Run the frozen Phase II synthetic protocol, never source EEG."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mist_cove.phase2.experiment import run
from mist_cove.phase2.provenance import implementation_snapshot, verify_preregistration, git_head, file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-version", type=int, choices=(1, 2), default=1)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    args.output = args.output or ROOT / f"mist_transfer_runs/mist_cove/phase2_v{args.protocol_version}/results.json"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a research receipt")
    preregistration = verify_preregistration(args.protocol_version)
    code = implementation_snapshot()
    config_path = ROOT / f"configs/mist_cove_phase2_synthetic_v{args.protocol_version}.json"
    config = json.loads(config_path.read_text())
    report = run(config, progress=lambda line: print(line, flush=True))
    if implementation_snapshot() != code:
        raise RuntimeError("code changed during confirmatory experiment; receipt invalid")
    report.update(implementation=code, implementation_commit=git_head(), preregistration=preregistration,
                  config_file_sha256=file_sha(config_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    print(f"Phase II synthetic receipt: {args.output}")


if __name__ == "__main__":
    main()

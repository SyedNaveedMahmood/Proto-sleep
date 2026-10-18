#!/usr/bin/env python
"""Prepared runner: real source access requires separate authorization + receipt."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mist_cove.phase2.source import run_source


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/mist_cove_source_prepared_v1.json")
    p.add_argument("--split", type=Path, default=ROOT / "experiments/splits/p3a_sc_source_v1.json")
    p.add_argument("--output", type=Path, default=ROOT / "mist_transfer_runs/mist_cove/phase2_source_v1")
    p.add_argument("--authorize-source-train-val", action="store_true")
    p.add_argument("--launch-receipt", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute-training", action="store_true", help="separately authorized future gate only; default is preflight")
    a = p.parse_args()
    print(run_source(a.config, a.split, a.output, authorize_source_train_val=a.authorize_source_train_val,
                     launch_receipt_path=a.launch_receipt, resume=a.resume, preflight_only=not a.execute_training))

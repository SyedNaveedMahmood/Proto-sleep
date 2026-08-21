#!/usr/bin/env python3
"""CLI entry point for ProtoMAE-Sleep.

Environment variables are set before importing the package so module-level research
configuration remains deterministic and easy to audit.
"""
from __future__ import annotations

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=False, help="Directory containing Sleep-EDF *.npz files.")
    p.add_argument("--output-dir", default="./protomae_sleep_runs")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast/GradScaler.")
    p.add_argument("--no-reuse", action="store_true", help="Ignore existing checkpoints/caches.")
    p.add_argument("--run-f", action="store_true", help="Run optional continuous-latent MAE control.")
    p.add_argument("--run-g", action="store_true", help="Run optional geometry MAE control.")
    p.add_argument("--final-test", action="store_true", help="Unlock final test evaluation for this run.")
    p.add_argument("--allow-test-rerun", action="store_true")
    p.add_argument("--self-test-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.data_dir:
        os.environ["SLEEP_EDF_NPZ_DIR"] = args.data_dir
    os.environ["PROTOMAE_OUT"] = args.output_dir
    os.environ["PROTOSLEEP_FOLD"] = str(args.fold)
    os.environ["PROTOSLEEP_SEED"] = str(args.seed)
    os.environ["PROTOSLEEP_AMP"] = "0" if args.no_amp else "1"
    os.environ["PROTOSLEEP_REUSE"] = "0" if args.no_reuse else "1"
    os.environ["PROTOSLEEP_RUN_F"] = "1" if args.run_f else "0"
    os.environ["PROTOSLEEP_RUN_G"] = "1" if args.run_g else "0"
    os.environ["PROTOSLEEP_FINAL_TEST"] = "1" if args.final_test else "0"
    os.environ["PROTOSLEEP_ALLOW_TEST_RERUN"] = "1" if args.allow_test_rerun else "0"
    os.environ["PROTOMAE_SELF_TEST_ONLY"] = "1" if args.self_test_only else "0"

    from protosleep.runner import main as run
    run()


if __name__ == "__main__":
    main()

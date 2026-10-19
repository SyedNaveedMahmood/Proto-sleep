#!/usr/bin/env python3
"""Inspect Phase VII prerequisites only; exit 2 at the missing-seal stop gate."""
import argparse
import json

from mist_cove.phase7.integrity import ROOT, inspect, write_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new ignored receipt filename, relative to repository")
    args = parser.parse_args()
    receipt = inspect()
    write_receipt(ROOT, args.output, receipt)
    print(json.dumps({"status": receipt["status"], "blockers": receipt["blockers"],
                      "checkpoint_deserializations": 0, "scientific_fits": 0}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

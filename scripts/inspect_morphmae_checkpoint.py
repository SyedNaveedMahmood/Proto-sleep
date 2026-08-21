#!/usr/bin/env python3
"""Strictly inspect/convert an MAE encoder checkpoint for MIST A4 initialization."""
from __future__ import annotations

import argparse
import json

from protosleep.mist import extract_mrcnn_state_dict, save_canonical_mrcnn_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Historical MAE/MorphMAE checkpoint to inspect.")
    p.add_argument(
        "--write-canonical",
        default=None,
        help="Optional destination for a canonical MRCNN-only checkpoint.",
    )
    args = p.parse_args()

    _, metadata = extract_mrcnn_state_dict(args.checkpoint)
    print("STRICT MRCNN COMPATIBILITY: PASS")
    print(json.dumps(metadata, indent=2, default=str))

    if args.write_canonical:
        converted = save_canonical_mrcnn_checkpoint(args.checkpoint, args.write_canonical)
        print("\nCanonical checkpoint written:")
        print(json.dumps(converted, indent=2, default=str))


if __name__ == "__main__":
    main()

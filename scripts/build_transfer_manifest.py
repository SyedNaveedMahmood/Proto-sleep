#!/usr/bin/env python
"""P2-SC geometry-first canonical manifest, bounded lazy loader and real pilot."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.manifest import build_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('mist_transfer_runs/p2_sc/pilot'))
    parser.add_argument('--mode',choices=['pilot','complete'],default='pilot')
    parser.add_argument('--pilot-dir',type=Path)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:return build_manifest(args.output_dir,args.mode,args.seed,args.resume,args.dry_run,pilot_dir=args.pilot_dir)
    except (ValueError,OSError,KeyError) as exc:
        print('P2-SC BLOCKED: '+str(exc),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())

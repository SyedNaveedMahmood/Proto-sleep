#!/usr/bin/env python
"""P1 read-only dataset inventory and leakage audit; no preprocessing/training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.audit import audit, load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/data_paths.example.yaml'))
    p.add_argument('--output-dir',type=Path,default=Path('mist_transfer_runs/p1_audit'))
    p.add_argument('--set-path',action='append',default=[],metavar='KEY=PATH',help='Explicit path override; env MIST_<KEY> also supported')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Discover files without opening datasets')
    p.add_argument('--split-manifest',type=Path,help='Optional split integrity preflight; reserved test opening prohibited')
    a=p.parse_args()
    config=load_config(a.config,a.set_path)
    return audit(config,a.output_dir,a.seed,a.resume,a.dry_run,a.split_manifest)

if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python
"""P3A-only matched SC development integration; no test/target training path."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.experiment import run_integration
from mist_transfer.protocol import DEFAULT_SPLIT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,default=DEFAULT_SPLIT)
    p.add_argument('--config',type=Path,default=Path('configs/p3a_sc_smoke.json'))
    p.add_argument('--output-dir',type=Path,default=Path('mist_transfer_runs/p3a/cuda_smoke'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    try:sys.exit(run_integration(a.split,a.config,a.output_dir,a.device,a.resume,a.dry_run))
    except (ValueError,OSError,KeyError,FloatingPointError) as exc:
        print('P3A BLOCKED: '+str(exc),file=sys.stderr,flush=True);sys.exit(2)

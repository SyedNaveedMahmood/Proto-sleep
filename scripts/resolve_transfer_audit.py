#!/usr/bin/env python
"""P1.5 read-only blocker resolution, reusing hash-validated P1 waveform proofs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.p1_5 import resolve
from mist_transfer.provenance import git_info, environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-audit',type=Path,default=Path('reports/data_audit/audit.json'))
    p.add_argument('--output-dir',type=Path,default=Path('mist_transfer_runs/p1_5'))
    p.add_argument('--archive',type=Path,help='Optional adjacent archive; read directory and tiny manifest only')
    p.add_argument('--reference',type=Path,action='append',default=[],help='Hash-only local candidate code reference; never executed')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:
        print(json.dumps({'git':git_info(),'environment':environment(),'prior_audit':str(a.prior_audit),
                         'output_dir':str(a.output_dir),'seed':a.seed,'split_ids':{},'training':False,
                         'preprocessing':False,'waveform_realignments':0},indent=2))
        return 0
    return resolve(a.prior_audit,a.output_dir,a.archive,a.seed,a.resume,a.reference)

if __name__=='__main__':
    raise SystemExit(main())

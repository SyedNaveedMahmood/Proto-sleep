#!/usr/bin/env python
"""Authorized preflight; training additionally requires recorded launch review."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mist_cove.phase3.source import preflight, run_reviewed

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/mist_cove_phase3_source_v1.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--authorize-source-train-val', action='store_true')
    p.add_argument('--mode', choices=['preflight', 'gate-a', 'gate-b'], default='preflight')
    p.add_argument('--proposal', type=Path)
    p.add_argument('--review', type=Path)
    a = p.parse_args()
    if a.mode == 'preflight':
        result = preflight(a.config, a.output, authorized=a.authorize_source_train_val)
        print(result['status'])
    else:
        if a.proposal is None or a.review is None:
            p.error('training requires --proposal and independently recorded --review')
        print(run_reviewed(a.config, a.proposal, a.review, a.output, gate=a.mode[-1], authorized=a.authorize_source_train_val))

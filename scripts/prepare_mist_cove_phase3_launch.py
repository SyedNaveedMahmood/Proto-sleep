#!/usr/bin/env python
"""Write an UNAPPROVED proposal for independent user review; never approve it."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from mist_cove.phase2.training import configure_precision
from mist_cove.phase3.source import launch_proposal

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['preflight', 'cpu', 'cuda', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    configure_precision('cuda'); torch.set_num_threads(2)
    print(launch_proposal(a.preflight, a.cpu, a.cuda, a.output)['status'])

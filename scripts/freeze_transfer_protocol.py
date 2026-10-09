#!/usr/bin/env python
"""Freeze identity-only SC TRAIN/VAL and reserved extension before model scores."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mist_transfer.protocol import freeze_source_split, DEFAULT_CANONICAL, DEFAULT_SPLIT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical',type=Path,default=DEFAULT_CANONICAL)
    p.add_argument('--output',type=Path,default=DEFAULT_SPLIT)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:freeze_source_split(a.canonical,a.output,a.seed,a.resume)
    except (ValueError,OSError,KeyError) as exc:
        print('P3A BLOCKED: '+str(exc),file=sys.stderr);sys.exit(2)

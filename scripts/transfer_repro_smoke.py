#!/usr/bin/env python
"""CPU-only reproducibility smoke; no model construction or training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from mist_transfer.provenance import AuditRun, atomic_json, atomic_csv, atomic_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps({'seed': a.seed, 'output_dir': str(a.output_dir), 'training': False}))
        return
    run = AuditRun(a.output_dir, {'purpose': 'P0 CPU reproducibility smoke'}, {}, a.seed, a.resume)
    values = np.random.default_rng(a.seed).normal(size=8)
    result = {'seed': a.seed, 'values': values.tolist(), 'sum': float(values.sum())}
    atomic_json(result, run.output/'smoke.json')
    atomic_csv([{'i': i, 'value': v} for i, v in enumerate(values)], ['i','value'], run.output/'smoke.csv')
    atomic_text(json.dumps(result, indent=2)+'\n', run.output/'FINAL_COPY_PASTE.txt')
    run.finish(['smoke.json','smoke.csv','FINAL_COPY_PASTE.txt'], 'PASS')
    print(json.dumps({'git_sha': run.identity['git']['sha'], 'signature': run.signature, 'result': result}))

if __name__ == '__main__':
    main()

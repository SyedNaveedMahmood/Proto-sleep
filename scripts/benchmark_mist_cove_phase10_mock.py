#!/usr/bin/env python
"""Bounded in-memory mocks only; cannot accept source paths or authorize EEG."""
import argparse
import json
from mist_cove.phase10.benchmark import cpu_benchmark,cuda_smoke
from mist_cove.phase10.safety import PUBLIC,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-smoke',action='store_true',help='idle-only bounded synthetic smoke; never interrupts another process')
    args=parser.parse_args(argv)
    if args.cuda_smoke:
        result=cuda_smoke();write_json(PUBLIC/'cuda_receipt.json',result)
    else:result=cpu_benchmark()
    print(json.dumps({'status':result['status'],'wall_seconds':result.get('wall_seconds'),'real_artifacts_opened':0}))


if __name__=='__main__':main()

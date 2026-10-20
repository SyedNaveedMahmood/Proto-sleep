#!/usr/bin/env python
"""Default metadata verification. Explicit probes generate synthetic mocks only."""
import argparse
import json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',default='verify',choices=('verify','behavior','circuits','comparators'))
    args=parser.parse_args(argv)
    if args.mode=='verify':
        from mist_cove.phase10.review import metadata_verify
        result=metadata_verify()
    else:
        from mist_cove.phase10.mock_study import behavior_probe,circuit_probe,comparator_probe
        result={'behavior':behavior_probe,'circuits':circuit_probe,'comparators':comparator_probe}[args.mode]()
    print(json.dumps({'status':result['status'],'real_artifacts_opened':0}))


if __name__=='__main__':main()

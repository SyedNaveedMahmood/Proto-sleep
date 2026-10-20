#!/usr/bin/env python
"""Pinned Git metadata or fresh synthetic fixtures only; no real-artifact flag."""
import argparse
import json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',default='verify',choices=('verify','fixtures'))
    args=parser.parse_args(argv)
    if args.mode=='verify':
        from mist_cove.phase11a.publication import metadata_check
        result=metadata_check()
    else:
        from mist_cove.phase11a.verification import run_all_scenarios
        result=run_all_scenarios()
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()

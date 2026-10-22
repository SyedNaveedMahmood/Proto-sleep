"""Explicit post-training studies; no new fits or expanded source access."""
import argparse
from mist_cove.phase13.statistics import summarize
from mist_cove.phase13.mechanistic import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',choices=('classification','mechanistic'));a=p.parse_args()
    if a.study=='classification':summarize()
    else:run()

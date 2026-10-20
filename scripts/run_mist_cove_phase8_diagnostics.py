"""Frozen prospective synthetic diagnostics only."""
import argparse
from mist_cove.phase8.diagnostics import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)

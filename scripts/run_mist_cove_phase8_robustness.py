"""Frozen synthetic campaign; calibration and evidence follow registered gates."""
import argparse
from mist_cove.phase8.robustness import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)

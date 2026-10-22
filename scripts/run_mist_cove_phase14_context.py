"""Frozen six-fit campaign or independent coarse support verification."""
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.audit:
  from mist_cove.phase14.context_audit import run
 else:
  from mist_cove.phase14.context_model import run
 run()

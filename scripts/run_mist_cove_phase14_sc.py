"""Explicit SC stages; repeated scientific execution is refused."""
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('sources','witnesses','statistics'));a=p.parse_args()
 if a.stage=='sources':
  from mist_cove.phase14.source_audit import run
  run('SC_TRAIN')
 elif a.stage=='witnesses':
  from mist_cove.phase14.witness_audit import run
  run('SC_TRAIN')
 else:
  from mist_cove.phase14.statistics import summarize
  summarize('SC_TRAIN')

"""Frozen STstages only; no fitting or target-adaptation interface."""
import argparse
from mist_cove.phase14.protocol import verify_st_seal
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('sources','witnesses','statistics'));a=p.parse_args();verify_st_seal()
 if a.stage=='sources':
  from mist_cove.phase14.source_audit import run
  run('ST_EVAL')
 elif a.stage=='witnesses':
  from mist_cove.phase14.witness_audit import run
  run('ST_EVAL')
 else:
  from mist_cove.phase14.statistics import summarize
  summarize('ST_EVAL')

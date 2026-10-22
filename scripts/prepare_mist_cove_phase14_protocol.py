"""Prepare once or verify committed scientific seal and engineering correction."""
import argparse
from mist_cove.phase14.protocol import freeze_st,verify_st_seal
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze-ST',action='store_true');a=p.parse_args()
 if a.freeze_ST:freeze_st()
 else:verify_st_seal();print('Committed STseal and bounded transport correction verified')

"""Explicit root registration, bounded inventory, or safe aggregate status."""
import argparse
from mist_cove.phase14.common import PRIVATE,PUBLIC,ROOT_AUTH_SHA256,sha256,write_new,read_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--authorize-root');p.add_argument('--initialize',action='store_true');a=p.parse_args()
 if a.authorize_root:
  if sha256(a.authorize_root.encode())!=ROOT_AUTH_SHA256:p.error('root differs from exact user authorization')
  write_new(PRIVATE/'local_authorization.json',{'root':a.authorize_root})
 if a.initialize:
  from mist_cove.phase14.manifest import inventory
  from mist_cove.phase14.edf_reader import inspect
  inventory();inspect()
 else:
  d=read_json(PUBLIC/'file_integrity_receipt.json');print({k:d[k] for k in ('SC_verified','ST_verified','failed_records','independent_official_hashes')})

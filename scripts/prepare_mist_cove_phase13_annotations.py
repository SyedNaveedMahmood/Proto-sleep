"""Prepare once, or serve only after explicit approved local reviewer registry."""
import argparse
import json
from pathlib import Path
from mist_cove.phase13.provenance import PUBLIC
from mist_cove.phase13.annotation_packets import prepare_packets,real_store
from mist_cove.phase13.annotation_interface import serve

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--serve',action='store_true')
    p.add_argument('--approved-registry',type=Path);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    if a.prepare:prepare_packets()
    elif a.serve:
        if a.approved_registry is None:p.error('explicit separately approved local registry required')
        serve(real_store(a.approved_registry),a.port)
    else:
        receipt=json.loads((PUBLIC/'annotation_readiness.json').read_text())
        print(json.dumps({k:receipt[k] for k in ('engineering_status','human_annotation','approved_reviewers','total_presentations')}))

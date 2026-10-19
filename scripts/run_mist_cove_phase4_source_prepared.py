"""Metadata inspection by default; real launches need fresh independent gates."""
import argparse,json
from pathlib import Path
from mist_cove.phase4.source import validate_config,run_reviewed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mist_cove_phase4_source_prepared_v1.json'))
    p.add_argument('--mode',choices=['inspect','gate-a','gate-b'],default='inspect')
    p.add_argument('--authorize-source-train-val',action='store_true')
    p.add_argument('--proposal',type=Path);p.add_argument('--review',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();c=validate_config(json.loads(a.config.read_text()))
    if a.mode=='inspect':print(json.dumps({'status':'PREPARED_NOT_AUTHORIZED','variants':c['variants'],'seeds':c['seeds'],'real_data_accessed':False}));return
    if not a.proposal or not a.review or not a.output:p.error('proposal, actual separate review and private output required')
    run_reviewed(c,a.proposal,a.review,a.output,authorized=a.authorize_source_train_val,gate=a.mode[-1])

if __name__=='__main__':main()

"""Execute only the hash-frozen Phase IV generated-data experiment."""
import argparse,json
from pathlib import Path
from mist_cove.phase4.experiment import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads(Path('configs/mist_cove_phase4_synthetic_v1.json').read_text());run(c,a.output)

"""Read-only final aggregate/source integrity audit; never opens EEG or weights."""
import json
import subprocess
from pathlib import Path
from collect import ROOT,PUBLIC,sha,safe


def main():
    from mist_cove.phase8.provenance import BASE,freeze_proof
    changes=subprocess.check_output(['git','diff','--name-status',BASE,'--'],cwd=ROOT,text=True).splitlines()
    if any(not line.startswith('A\t') for line in changes):raise ValueError('historical tracked artifact changed')
    manifest=json.loads((PUBLIC/'publication_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected:raise ValueError('publication mismatch: '+name)
    for p in PUBLIC.glob('*.json'):safe(json.loads(p.read_text()))
    freeze_proof('diagnostic');freeze_proof('confirmatory')
    for name in ('figure_manifest.json','descriptive_figure_manifest.json'):
        figure=json.loads((PUBLIC/name).read_text())
        for relative,expected in figure['figures'].items():
            if sha(PUBLIC/'figures'/relative)!=expected:raise ValueError('figure mismatch')
    print(json.dumps({'status':'PASS','verified_artifacts':len(manifest['files']),
        'historical_tracked_modifications':0,'private_model_or_real_data_opens':0,
        'scientific_primary':'FAIL','additive_confirmation':'NOT TESTED','cuda':'BLOCKED'}))

if __name__=='__main__':main()

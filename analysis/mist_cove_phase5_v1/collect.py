"""Project only generated aggregate results; never opens a waveform/checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from mist_cove.phase5.runtime import ROOT,historical_proofs,frozen_proofs,execution_proofs


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);a=parser.parse_args()
    allowed=(ROOT/'mist_transfer_runs/mist_cove/phase5_v1').resolve()
    if allowed not in a.input.resolve().parents or a.input.name!='results.json':raise ValueError('only private Phase V aggregate receipt accepted')
    result=json.loads(a.input.read_text())
    assert result['status']=='COMPLETE_SYNTHETIC_SEQUENTIAL_STUDY' and not result['real_data_accessed']
    assert result['execution']==execution_proofs() and result['preregistration']==frozen_proofs() and result['historical']==historical_proofs()
    rows=result['rows'];assert result['optimizer_updates']==sum(r['training']['optimizer_updates'] for r in rows)
    assert result['example_exposures']==sum(r['training']['example_exposures'] for r in rows)
    assert all(r['training']['optimizer_updates']==result['config']['updates'] for r in rows if r['regime']!='fixed_encoder')
    assert len({(r['family'],r['seed'],r['method'],r['regime']) for r in rows})==len(rows)
    public=ROOT/'reports/mist_cove/phase5_v1'
    public.mkdir(parents=True,exist_ok=True)
    with (public/'per_run_v1.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    receipt={k:v for k,v in result.items() if k not in {'rows','diagnostics'}}
    receipt.update(private_results_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),completed_fits=len(rows))
    write_new(public/'execution_receipt_v1.json',receipt)
    write_new(public/'signal_audits_v1.json',result['diagnostics'])
    groups={};csvrows=[]
    for row in rows:
        for condition,values in row['conditions'].items():
            key=(row['family'],row['method'],row['regime'],condition)
            groups.setdefault(key,[]).append(values)
            csvrows.append({'family':row['family'],'seed':row['seed'],'method':row['method'],'regime':row['regime'],'condition':condition,
                **{k:values[k] for k in ['accuracy','macro_f1','balanced_accuracy','nll','brier','ece15','confidence']},
                'accuracy_ci99_low':values['accuracy_ci99'][0],'accuracy_ci99_high':values['accuracy_ci99'][1],
                'restricted_equivalence':values['restricted_equivalence'],'parameters':row['training']['parameters'],
                'updates':row['training']['optimizer_updates'],'training_seconds':row['training']['training_seconds']})
    summary=[]
    for key,values in groups.items():
        summary.append({'family':key[0],'method':key[1],'regime':key[2],'condition':key[3],
            'n_seeds':len(values),'metrics':{k:{'mean':statistics.mean(r[k] for r in values),'seed_sd':statistics.stdev(r[k] for r in values) if len(values)>1 else None} for k in ['accuracy','macro_f1','balanced_accuracy','nll','brier','ece15','confidence']}})
    write_new(public/'summary_v1.json',summary)
    with (public/'metrics_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    print('Published',len(rows),'lossless generated aggregate rows; original proofs verified.')


if __name__=='__main__':main()

"""Read only completed generated aggregate results, never waveforms or weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from mist_cove.phase6.runtime import ROOT,frozen,historical


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    allowed=(ROOT/'mist_transfer_runs/mist_cove/phase6_v1').resolve()
    if allowed not in a.input.resolve().parents or a.input.name!='results.json':raise ValueError('private completed Phase VI aggregate receipt required')
    r=json.loads(a.input.read_text());assert r['status']=='COMPLETE_SYNTHETIC_STUDY' and not r['real_data_accessed']
    assert r['frozen']==frozen() and r['historical']==historical()
    rows=r['rows'];assert len(rows)==r['fits'] and sum(v['training']['optimizer_updates'] for v in rows)==r['optimizer_updates']
    assert sum(v['training']['example_exposures'] for v in rows)==r['example_exposures']
    assert len({(v['task'],v['seed'],v['stage'],v['method']) for v in rows})==len(rows)
    public=ROOT/'reports/mist_cove/phase6_v1';public.mkdir(parents=True,exist_ok=True)
    with (public/'per_run_v1.jsonl').open('x') as f:
        for v in rows:f.write(json.dumps(v,allow_nan=False)+'\n')
    receipt={k:v for k,v in r.items() if k!='rows'}
    receipt['private_receipt_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    write(public/'execution_receipt_v1.json',receipt)
    groups={};flat=[]
    for row in rows:
        for condition,values in row['conditions'].items():
            groups.setdefault((row['task'],row['stage'],row['method'],condition),[]).append(values)
            flat.append({'task':row['task'],'seed':row['seed'],'stage':row['stage'],'method':row['method'],'condition':condition,
                **{k:values[k] for k in ['accuracy','macro_f1','balanced_accuracy','nll','brier','ece15','confidence','false_confidence_0_9']},
                'accuracy_ci99_low':values['accuracy_ci99'][0],'accuracy_ci99_high':values['accuracy_ci99'][1],
                'parameters':row['training']['parameters'],'updates':row['training']['optimizer_updates'],'training_seconds':row['training']['training_seconds']})
    summary=[{'task':key[0],'stage':key[1],'method':key[2],'condition':key[3],
        'metrics':{k:{'mean':statistics.mean(v[k] for v in values),'seed_sd':statistics.stdev(v[k] for v in values)} for k in ['accuracy','macro_f1','balanced_accuracy','nll','brier','ece15','confidence','false_confidence_0_9']}} for key,values in groups.items()]
    write(public/'summary_v1.json',summary)
    with (public/'metrics_v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    print('Published',len(rows),'lossless generated fits; frozen and historical identities PASS')


if __name__=='__main__':main()

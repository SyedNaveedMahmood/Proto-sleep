"""CPU-only descriptive diagnostics of locally created, digest-sealed best states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from mist_cove.phase2.provenance import file_sha
from mist_evidence.runtime import load_trusted
BASE=ROOT/'mist_transfer_runs/mist_cove/phase3_source_v1';RUNS=BASE/'training_v1'
torch.set_num_threads(1);seen={};latest={}
with (BASE/'best_prediction_diagnostics_v1.jsonl').open('x') as stream:
 while not (BASE/'gate_b_execution_v1.json').exists():
  for seed in [123,456,789]:
   for name in ['q1','availability','background','rounded_q1','raw_context','evidence']:
    run=RUNS/f'seed_{seed}'/name;seal_path=run/'EPOCH_SEAL.json'
    if not seal_path.exists():continue
    seal=json.loads(seal_path.read_text());sha=seal['outputs']['best.pt'];key=f'{seed}_{name}'
    if seen.get(key)==sha:continue
    if file_sha(run/'best.pt')!=sha:continue # normal short atomic-write/seal interval
    best=load_trusted(run/'best.pt')
    if file_sha(run/'best.pt')!=sha:continue
    assert best['signature']==seal['signature']
    cm=np.asarray(best['metrics']['confusion_matrix']);counts=cm.sum(0)
    info={'seed':seed,'variant':name,'best_epoch':best['best_epoch'],'observed_at_epoch_boundary':seal['epoch'],
     'participant_macro_f1':best['metrics']['mean_subject_macro_f1'],'prediction_counts':counts.tolist(),
     'one_class':bool((counts>0).sum()==1),'largest_class_fraction':float(counts.max()/counts.sum()),
     'absent_prediction_stages':np.flatnonzero(counts==0).tolist(),'best_checkpoint_sha256':sha,
     'scope':'selected-best checkpoint diagnostics, not all non-best epochs'}
    stream.write(json.dumps(info)+'\n');stream.flush();latest[key]=info;seen[key]=sha
  tmp=BASE/'best_prediction_diagnostics_live_v1.tmp';tmp.write_text(json.dumps(latest,indent=2));os.replace(tmp,BASE/'best_prediction_diagnostics_live_v1.json')
  time.sleep(10)
print('Best-checkpoint diagnostics monitor stopped; training execution marker present')

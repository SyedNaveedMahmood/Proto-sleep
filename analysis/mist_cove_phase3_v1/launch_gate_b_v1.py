"""Authorized operational supervisor; frozen scientific implementation unchanged."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from mist_cove.phase3.source import load_sources,prepared,validate_review,reconciled_environment,require_idle_gpu,VARIANTS
from mist_cove.phase2.source import training_config
from mist_cove.phase2.provenance import digest,file_sha,implementation_snapshot,environment,git_head
from mist_cove.phase2.adapter import build_adapter
from mist_cove.phase2.training import configure_precision,view_identity,verify_seal
from mist_evidence.runtime import tensor_digest,load_trusted,atomic_json
BASE=Path(__file__).resolve().parent;RUNS=BASE/'training_v1'
PROPOSAL=BASE/'launch_proposal_v1.json';REVIEW=BASE/'gate_a_actual_user_review_v1.json'
AUTH=BASE/'gate_b_actual_user_authorization_v1.json'
expected='8282be8a6d7124a0fd6e83a4ae8b8e273919fc0b7976a756389e8088d3ec38b0'
auth=json.loads(AUTH.read_text())
assert auth['decision']=='APPROVE_GATE_B' and auth['gate_b_authorized'] is True and auth['scope']=='SC_TRAIN_VAL_ONLY'
assert auth['proposal_sha256']==file_sha(PROPOSAL)==expected
assert file_sha(auth['authorization_message_path'])==auth['authorization_message_sha256']
assert auth['approved_gate_a_commit']==git_head()=='ed963a828ea92a551a554f2e332cd097b25cb013'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
configure_precision('cuda');torch.set_num_threads(2)
require_idle_gpu();proposal=validate_review(PROPOSAL,REVIEW)
pre=json.loads(Path(proposal['artifacts']['preflight']['path']).read_text())
cfgpath=ROOT/'configs/mist_cove_phase3_source_v1.json';c=json.loads(cfgpath.read_text());original=prepared(c)
assert auth['seeds']==c['seeds'] and auth['variants']==c['variants']==VARIANTS
assert digest(c)==proposal['config_sha256']
train,val,norm,banks,grid,proofs=load_sources(c)
assert proofs==pre['proofs'] and grid==pre['grid'] and reconciled_environment()==pre['environment']
code=implementation_snapshot();assert code['sha256']==proposal['implementation_sha256']
gate=json.loads((RUNS/'gate_a.json').read_text())
assert gate['status']=='PASS' and gate['maximum_gpu_hours']<=24 and gate['proposal_sha256']==expected
public=json.loads((ROOT/'reports/mist_cove/phase3_gate_a_v1.json').read_text())
assert file_sha(RUNS/'gate_a.json')==public['gate_a_receipt_sha256']
provenance={'scope':'SC_TRAIN_VAL_ONLY','config':c,'reviewed_proposal_sha256':expected,'grid':grid,'proofs':proofs}
identities={};initial_hashes={}
for name in VARIANTS:
 run=RUNS/'seed_123'/name;bank=banks[123];model=build_adapter(name,norm,bank)
 identity=digest({'version':'mist-cove-sealed-training-v1','model':model.cfg.dictionary(),
 'train':asdict(training_config(original,123)),'sources':[view_identity(r) for r in train+val],
 'provenance':provenance,'code':code,'environment':environment('cuda'),
 'bank_waveform_digest':tensor_digest({str(i):v for i,v in enumerate(bank['waveforms'])}),
 'bank_metadata_digest':digest({k:v for k,v in bank.items() if k!='waveforms'})})
 seal=verify_seal(run,identity);assert seal['epoch']==1
 paused=json.loads((run/'PAUSED.json').read_text());assert paused['epoch']==1 and paused['outputs']==seal['outputs']
 for f,h in paused['outputs'].items():assert file_sha(run/f)==h
 old=next(x for x in public['rows'] if x['variant']==name)
 for f,h in old['artifact_sha256'].items():assert file_sha(run/f)==h
 assert file_sha(gate['boundaries'][name]['path'])==gate['boundaries'][name]['sha256']
 state=load_trusted(run/'last.pt');assert state['epoch']==1 and len(state['history'])==1 and state['history'][0]['optimizer_steps']==4978
 assert state['signature']==seal['signature']
 assert len(state['rng']['cuda'])==1 and state['optimizer']['state']
 for entry in state['optimizer']['state'].values():assert int(entry['step'])==4978
 identities[name]=identity;initial_hashes[name]=dict(seal['outputs'])
 del model,state
packet={'status':'PASS_AUTHORIZED_GATE_B_PRERESUME','actual_gate_b_authorization_sha256':file_sha(AUTH),
 'proposal_sha256':expected,'gate_a_receipt_sha256':file_sha(RUNS/'gate_a.json'),'initial_epoch_one_outputs':initial_hashes,
 'identities':identities,'implementation':code,'config_sha256':digest(c),'environment':pre['environment'],
 'source_proofs':proofs,'grid':grid,'budget_gpu_hours':gate['maximum_gpu_hours'],'supervisor_sha256':file_sha(__file__)}
atomic_json(packet,BASE/'gate_b_preresume_private_v1.json')
print('PASS: all source/environment/recipe proofs and six exact epoch-one states verified; starting Gate B',flush=True)
# The frozen runner consumes the original true Gate A review for its immutable
# launch identity; the additional actual Gate B authorization was enforced above.
command=[sys.executable,'-u','scripts/run_mist_cove_phase3.py','--authorize-source-train-val','--mode','gate-b',
 '--proposal',str(PROPOSAL),'--review',str(REVIEW),'--output',str(RUNS)]
# Release this verifier's CUDA context before the child checks GPU exclusivity.
# Exec into a CPU-only operational supervisor subprocess; no training state changes.
monitor=BASE/'monitor_gate_b_v1.py'
with (BASE/'gate_b_launch_ready_v1.json').open('x') as f:json.dump({'command':command,'packet_sha256':file_sha(BASE/'gate_b_preresume_private_v1.json'),'gate_a_seconds':public['total_measured_epoch_seconds']},f)
os.execv(sys.executable,[sys.executable,str(monitor)])

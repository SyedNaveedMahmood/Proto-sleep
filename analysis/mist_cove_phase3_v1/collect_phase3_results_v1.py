"""Read-only final verification and privacy-preserving aggregate projection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from dataclasses import asdict
import csv,json,time,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from mist_cove.phase3.source import load_sources,prepared,reconciled_environment,require_idle_gpu,validate_review
from mist_cove.phase2.source import training_config,VARIANTS
from mist_cove.phase2.training import configure_precision,verify_seal,view_identity
from mist_cove.phase2.provenance import digest,file_sha,implementation_snapshot,environment
from mist_cove.phase2.adapter import build_adapter
from mist_evidence.runtime import tensor_digest,load_trusted,atomic_json
from mist_transfer.metrics import classification_metrics
BASE=ROOT/'mist_transfer_runs/mist_cove/phase3_source_v1';RUNS=BASE/'training_v1'
SCALARS=['mean_subject_macro_f1','macro_f1','accuracy','kappa','balanced_accuracy','five_class_balanced_accuracy_zero_absent','nll','brier_score','ece_15_equal_width_bins']

def runtime_identity(model,cfg,seed,train,val,bank,provenance,code):
 return digest({'version':'mist-cove-sealed-training-v1','model':model.cfg.dictionary(),'train':asdict(training_config(prepared(cfg),seed)),
  'sources':[view_identity(r) for r in train+val],'provenance':provenance,'code':code,'environment':environment('cuda'),
  'bank_waveform_digest':tensor_digest({str(i):v for i,v in enumerate(bank['waveforms'])}),
  'bank_metadata_digest':digest({k:v for k,v in bank.items() if k!='waveforms'})})

def finite_tree(obj):
 if torch.is_tensor(obj):return bool(torch.isfinite(obj).all())
 if isinstance(obj,dict):return all(finite_tree(v) for v in obj.values())
 if isinstance(obj,(list,tuple)):return all(finite_tree(v) for v in obj)
 return True

def verify_prediction_grid(path,val,metrics):
 lookup={r.recording_id:r for r in val};counts={rid:0 for rid in lookup}
 truth=[];predicted=[];subjects=[];masks=[]
 with path.open() as f:
  for row in csv.DictReader(f):
   assert row['recording_id'] in lookup
   r=lookup[row['recording_id']];i=int(row['original_epoch_index'])
   assert i==counts[r.recording_id] and row['subject_id']==r.subject
   assert int(row['start_seconds'])==i*30
   assert (row['scored']=='True')==bool(r.scoring_mask[i])
   assert row['truth']==('' if r.y[i]<0 else str(r.y[i]))
   p=int(row['prediction']);assert 0<=p<5
   truth.append(int(r.y[i]));predicted.append(p);subjects.append(r.subject);masks.append(bool(r.scoring_mask[i]))
   counts[r.recording_id]+=1
 assert all(counts[r.recording_id]==len(r.y) for r in val) and sum(counts.values())==21796
 recalculated=classification_metrics(truth,predicted,subjects,masks)
 assert recalculated['confusion_matrix']==metrics['confusion_matrix']
 assert recalculated['mean_subject_macro_f1']==metrics['mean_subject_macro_f1']
 assert recalculated['subject_macro_f1']==metrics['subject_macro_f1']
 return {'physical_predictions':len(truth),'scored_predictions':sum(masks),'original_grid_verified':True,'confusion_and_participant_f1_recomputed':True}

def collect():
 started=time.perf_counter();configure_precision('cuda');torch.set_num_threads(2);require_idle_gpu()
 execution=json.loads((BASE/'gate_b_execution_v1.json').read_text());assert execution['status']=='CHILD_EXITED_OK'
 auth=json.loads((BASE/'gate_b_actual_user_authorization_v1.json').read_text());assert auth['decision']=='APPROVE_GATE_B' and auth['gate_b_authorized'] is True
 assert file_sha(auth['authorization_message_path'])==auth['authorization_message_sha256']
 launch=json.loads((BASE/'gate_b_preresume_private_v1.json').read_text())
 assert file_sha(BASE/'gate_b_actual_user_authorization_v1.json')==launch['actual_gate_b_authorization_sha256']
 proposal=validate_review(BASE/'launch_proposal_v1.json',BASE/'gate_a_actual_user_review_v1.json')
 c=json.loads((ROOT/'configs/mist_cove_phase3_source_v1.json').read_text())
 train,val,norm,banks,grid,proofs=load_sources(c)
 assert proofs==launch['source_proofs'] and grid==launch['grid']
 assert reconciled_environment()==launch['environment']
 code=implementation_snapshot();assert code==launch['implementation']
 provenance={'scope':'SC_TRAIN_VAL_ONLY','config':c,'reviewed_proposal_sha256':auth['proposal_sha256'],'grid':grid,'proofs':proofs}
 raw=json.loads((RUNS/'results.json').read_text());assert raw['status']=='ALL_18_SOURCE_RUNS_COMPLETE' and len(raw['runs'])==18
 assert {(v['seed'],v['variant']) for v in raw['runs']}=={(seed,name) for seed in [123,456,789] for name in VARIANTS}
 safe=[];private=[]
 for seed in [123,456,789]:
  for name in VARIANTS:
   run=RUNS/f'seed_{seed}'/name;bank=banks[seed];model=build_adapter(name,norm,bank)
   identity=runtime_identity(model,c,seed,train,val,bank,provenance,code)
   seal=verify_seal(run,identity);assert (run/'COMPLETE_SEAL.json').exists() and not (run/'PAUSED.json').exists()
   last=load_trusted(run/'last.pt');best=load_trusted(run/'best.pt');done=json.loads((run/'COMPLETE.json').read_text())
   assert last['signature']==best['signature']==done['signature']
   assert tensor_digest(best['state_dict'])==done['best_state_digest']
   assert finite_tree(last['state_dict']) and finite_tree(last['optimizer']) and finite_tree(best['state_dict'])
   history=last['history'];assert len(history)==last['epoch']<=60 and len(history)>=1
   selected=-float('inf');best_epoch=0;stale=0
   for i,h in enumerate(history,1):
    assert h['epoch']==i and h['optimizer_steps']==4978 and h['scored_training_epochs']==79025
    assert h['finite_loss_logits_gradients_state'] is True
    if h['val_subject_macro_f1']>selected+1e-8:selected=h['val_subject_macro_f1'];best_epoch=i;stale=0
    else:stale+=1
    assert h['best_epoch']==best_epoch and h['stale']==stale
   assert last['epoch']==60 or stale>=12
   assert last['best_epoch']==best_epoch==best['best_epoch']==done['best_epoch']
   assert last['best_score']==selected and last['stale']==stale
   assert all(int(state['step'])==4978*last['epoch'] for state in last['optimizer']['state'].values())
   if seed==123:
    # The original epoch-one resource/selection row and RNG-bearing output
    # hashes were verified before resuming; retaining row one proves no replay.
    frozen=json.loads((run/'gate_a_boundary.json').read_text())
    assert history[0]==frozen['resources']
    assert file_sha(run/'gate_a_boundary.json')==json.loads((RUNS/'gate_a.json').read_text())['boundaries'][name]['sha256']
   metrics=done['metrics'];assert metrics==best['metrics']
   predicted=verify_prediction_grid(run/'validation_predictions.csv',val,metrics)
   row=next(v for v in raw['runs'] if v['seed']==seed and v['variant']==name)
   assert row['metrics']==metrics and row['actual_epochs']==last['epoch'] and row['optimizer_steps']==4978*last['epoch']
   for field,key in [('train_seconds','train_seconds'),('validation_seconds','validation_seconds'),('train_io_seconds','train_io_seconds'),('val_io_seconds','val_io_seconds'),('epoch_wall_seconds_excluding_checkpoint_writes','seconds')]:
    assert row[field]==sum(h[key] for h in history)
   for field in ['peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','peak_process_rss_KiB']:
    assert row[field]==max(h[field] for h in history)
   assert row['parameters']==sum(p.numel() for p in model.parameters())
   assert row['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
   public={k:v for k,v in row.items() if k not in {'metrics','complete_seal'}}
   public['metrics']={k:metrics[k] for k in SCALARS+['per_stage_f1','per_stage_recall','confusion_matrix','support','n_epochs','unscored_epochs','absent_class_policy']}
   public['complete_seal_sha256']=file_sha(run/'COMPLETE_SEAL.json')
   public['prediction_verification']=predicted
   public['best_checkpoint_absent_prediction_stages']=np.flatnonzero(np.asarray(metrics['confusion_matrix']).sum(0)==0).tolist()
   public['near_collapse_98_percent']=public['largest_class_fraction']>=.98
   public['selection_reverified']=True;public['seed123_original_epoch_one_retained']=seed==123
   safe.append(public);private.append({'seed':seed,'variant':name,'metrics':metrics,'identity':identity,'outputs':seal['outputs']})
   del model,last,best
 aggregate={}
 for name in VARIANTS:
  rows=[r for r in safe if r['variant']==name]
  summary={key:{'mean':float(np.mean([r['metrics'][key] for r in rows])),'sample_sd_across_optimization_seeds':float(np.std([r['metrics'][key] for r in rows],ddof=1))} for key in SCALARS}
  summary['per_stage_f1_mean']=np.mean([r['metrics']['per_stage_f1'] for r in rows],axis=0).tolist()
  summary['per_stage_f1_sample_sd_across_seeds']=np.std([r['metrics']['per_stage_f1'] for r in rows],axis=0,ddof=1).tolist()
  summary['per_stage_recall_mean']=np.mean([r['metrics']['per_stage_recall'] for r in rows],axis=0).tolist()
  summary['confusion_matrix_sum_across_seeds']=np.sum([r['metrics']['confusion_matrix'] for r in rows],axis=0).tolist()
  subject_means=np.mean([list(r['metrics']['subject_macro_f1'].values()) for r in private if r['variant']==name],axis=0)
  summary['between_participant_variation_of_seed_means']={'n_participants':4,'sample_sd':float(np.std(subject_means,ddof=1)),'minimum':float(subject_means.min()),'maximum':float(subject_means.max()),'caveat':'Descriptive four-person source-VAL sample; optimization seeds are not participants.'}
  summary['realized_epochs']=sum(r['actual_epochs'] for r in rows);summary['optimizer_updates']=sum(r['optimizer_steps'] for r in rows)
  summary['epoch_wall_seconds_excluding_checkpoint_writes']=sum(r['epoch_wall_seconds_excluding_checkpoint_writes'] for r in rows)
  aggregate[name]=summary
 q1=[r for r in safe if r['variant']=='q1'];bg=[r for r in safe if r['variant']=='background']
 utility=[a['metrics']['mean_subject_macro_f1']-b['metrics']['mean_subject_macro_f1'] for a,b in zip(q1,bg)]
 paired={}
 for other in [v for v in VARIANTS if v!='q1']:
  differences=[]
  for seed in [123,456,789]:
   a=next(r['metrics']['subject_macro_f1'] for r in private if r['seed']==seed and r['variant']=='q1')
   b=next(r['metrics']['subject_macro_f1'] for r in private if r['seed']==seed and r['variant']==other)
   differences.append([a[s]-b[s] for s in sorted(a)])
  participant=np.mean(differences,axis=0)
  paired[other]={'mean_participant_difference_q1_minus_control':float(participant.mean()),'sd_between_four_participant_seed_mean_differences':float(participant.std(ddof=1)),'minimum_difference':float(participant.min()),'maximum_difference':float(participant.max()),'per_seed_mean_difference':np.mean(differences,axis=1).tolist()}
 summary={'status':'VERIFIED_ALL_18_COMPLETE_SOURCE_ONLY','authorization_message_sha256':auth['authorization_message_sha256'],
 'actual_gate_b_authorization_sha256':file_sha(BASE/'gate_b_actual_user_authorization_v1.json'),
 'proposal_sha256':auth['proposal_sha256'],'implementation_sha256':code['sha256'],'config_sha256':digest(c),
 'split_fingerprint':None,
 'grid':grid,'rows':safe,'aggregate':aggregate,'paired_participant_aggregate_differences':paired,
 'prespecified_decisions':{'q1_exceeds_background_all_three_seeds':all(v>0 for v in utility),'q1_minus_background_by_seed':utility,
 'q1_mean_reaches_both_new_baselines':all(aggregate['q1']['mean_subject_macro_f1']['mean']>=aggregate[v]['mean_subject_macro_f1']['mean'] for v in ['raw_context','evidence']),
 'interpretation':'Descriptive preregistered source decisions, not statistical population or transfer claims.'},
 'total_realized_epochs':sum(r['actual_epochs'] for r in safe),'total_optimizer_updates':sum(r['optimizer_steps'] for r in safe),
 'single_gpu_gate_b_elapsed_wall_seconds':execution['gate_b_elapsed_wall_seconds'],
 'gate_a_summed_epoch_seconds':execution['gate_a_epoch_seconds'],
 'peak_gpu_allocated_bytes':max(r['peak_gpu_allocated_bytes'] for r in safe),
 'peak_gpu_reserved_bytes':max(r['peak_gpu_reserved_bytes'] for r in safe),
 'all_seals_and_original_prediction_grids_verified':True,'source_normalization_banks_environment_unchanged':True,
 'historical_p3b_reference_means':{'raw_context':0.723511,'evidence':0.718031,'note':'Historical source-VAL selected checkpoints; not new runs or independent validation.'},
 'audit_wall_seconds':time.perf_counter()-started,'analysis_code_sha256':file_sha(__file__),
 'reserved_access':False,'st_access':False,'shhs_access':False}
 # Record the actual frozen split value, not the preflight artifact fingerprint.
 pre=json.loads(Path(proposal['artifacts']['preflight']['path']).read_text())
 summary['split_fingerprint']=pre['split_fingerprint'];summary['split_file_sha256']=pre['split_file_sha256']
 atomic_json({'aggregate':summary,'private_runs':private,'source_proofs':proofs},BASE/'phase3_completed_private_v1.json')
 with (BASE/'phase3_source_safe_results_v1.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':summary['status'],'epochs':summary['total_realized_epochs'],'updates':summary['total_optimizer_updates'],
  'participant_macro_f1':{k:v['mean_subject_macro_f1'] for k,v in aggregate.items()}},indent=2))
 return summary
if __name__=='__main__':collect()

"""Postfit waveform audit under the frozen Phase III analysis plan.

No training or fitting. Real data and plots always remain in the ignored root.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from mist_cove.phase2.provenance import file_sha,implementation_snapshot,digest
from mist_cove.phase2.adapter import build_adapter
from mist_cove.phase2.report import local_evidence_report
from mist_cove.phase2.traces import EpochIdentity,source_ledger
from mist_cove.phase2.training import configure_precision,verify_seal
from mist_cove.model import spectral_features
from mist_cove.phase3.source import load_sources,require_idle_gpu,reconciled_environment
from mist_evidence.runtime import load_trusted,atomic_json
BASE=ROOT/'mist_transfer_runs/mist_cove/phase3_source_v1';RUNS=BASE/'training_v1'


def safe_mean(values):
 values=[v for v in values if v is not None]
 return float(np.mean(values)) if values else None


def choose_donor(physical,intervals):
 candidates=[]
 for lo in range(0,2801,100):
  hi=lo+200
  if any(lo<b and hi>a for a,b in intervals):continue
  rms=float(physical[lo:hi].std(unbiased=False));candidates.append((rms,lo,hi))
 if not candidates:return None,{'supported':False,'reason':'no disjoint 200-sample background candidate'}
 rms,lo,hi=min(candidates)
 if rms>3:return None,{'supported':False,'reason':'minimum disjoint RMS exceeds unchanged 3-uV synthetic compatibility cap','minimum_rms_uV':rms}
 return (lo,hi),{'supported':True,'centered_rms_uV':rms,'interval':[lo,hi]}


def declared_groups(model,x,evidence):
 wave=model.physical_uV(x)[0,0]
 intervals=[tuple(int(v) for v in interval) for interval in evidence.event_intervals[0,evidence.valid[0]].tolist()]
 groups={'all_selected':intervals} if intervals else {}
 dominance=[];fractions=[]
 for lo,hi in intervals:
  bands,_,_=spectral_features(wave[lo:hi][None]);band=bands[0]
  dominance.append(int(band.argmax()));fractions.append(band)
 for b in range(5):
  members=[v for v,k in zip(intervals,dominance) if k==b]
  if len(members)>=2:groups[f'selected_same_dominant_band_{b}']=members
 similarities=[]
 for i in range(len(fractions)):
  for j in range(i):
   similarities.append(float(torch.nn.functional.cosine_similarity(fractions[i][None],fractions[j][None])[0]))
 return groups,intervals,dominance,similarities


def transformations(model,x,seed):
 physical=model.physical_uV(x)
 g=torch.Generator().manual_seed(seed)
 noise=torch.randn(physical.shape,generator=g).to(x.device)*.01
 changed={'tiny_noise_0.01uV':physical+noise,'physical_gain_1.01':physical*1.01,
          'within_epoch_shift_10_samples':torch.cat((torch.zeros_like(physical[...,:10]),physical[...,:-10]),dim=-1)}
 return {k:(v-model.mean_uV)/model.std_uV for k,v in changed.items()}


def make_plot(path,physical,evidence,truth,predicted,stage_names):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 y=physical.detach().cpu().numpy();t=np.arange(3000)/100
 fig,axes=plt.subplots(3,1,figsize=(12,7),layout='constrained')
 axes[0].plot(t,y,lw=.6,color='black')
 for lo,hi in evidence.event_intervals[0,evidence.valid[0]].tolist():axes[0].axvspan(lo/100,hi/100,color='tab:red',alpha=.2)
 axes[0].set(xlabel='Seconds within original physical epoch',ylabel='Calibrated uV',title=f'Scored {stage_names[truth]}; model {stage_names[predicted]}; selected intervals are model evidence, not confirmed clinical events')
 axes[1].specgram(y,NFFT=200,Fs=100,noverlap=100,cmap='magma');axes[1].set(ylim=(0,30),xlabel='Seconds',ylabel='Hz')
 events=evidence.contributions[0,evidence.valid[0]].detach().cpu().numpy()
 image=axes[2].imshow(events.T,aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
 axes[2].set(yticks=range(5),yticklabels=stage_names,xlabel='Selected witness rank',title='Signed additive event scores (not removal effects)');fig.colorbar(image,ax=axes[2])
 fig.savefig(path,dpi=130);plt.close(fig)


def analyze():
 started=time.perf_counter();configure_precision('cuda');torch.set_num_threads(2);require_idle_gpu()
 assert (BASE/'phase3_source_safe_results_v1.json').exists(),'Complete/audit all 18 runs before waveform analysis'
 completed=json.loads((BASE/'phase3_completed_private_v1.json').read_text())
 plan_path=BASE/'waveform_analysis_plan_v1.json';plan=json.loads(plan_path.read_text())
 plan_receipt=json.loads((BASE/'waveform_analysis_plan_receipt_v1.json').read_text())
 assert file_sha(plan_path)==plan_receipt['analysis_plan_sha256']
 assert file_sha(ROOT/plan['parent_protocol'])==plan_receipt['parent_protocol_sha256']
 launch=json.loads((BASE/'gate_b_preresume_private_v1.json').read_text())
 c=json.loads((ROOT/'configs/mist_cove_phase3_source_v1.json').read_text())
 train,val,norm,banks,grid,proofs=load_sources(c)
 assert proofs==launch['source_proofs'] and grid==launch['grid']
 assert reconciled_environment()==launch['environment'] and implementation_snapshot()==launch['implementation']
 out=BASE/'waveform_audit_v1';assert not out.exists();out.mkdir();(out/'plots').mkdir()
 samples=[];physical_arrays=[]
 for r in val:
  for stage in range(5):
   eligible=np.flatnonzero((r.y==stage)&r.scoring_mask)[:5]
   for ordinal,index in enumerate(eligible):
    x=torch.from_numpy(r.x[int(index)][None])
    samples.append({'recording':r,'stage':stage,'epoch':int(r.indices[index]),'index':int(index),'ordinal':ordinal,'x':x})
 assert samples
 private_sample_plan=[{'recording_id':v['recording'].recording_id,'subject_id':v['recording'].subject,'stage':v['stage'],'original_epoch':v['epoch'],'ordinal':v['ordinal']} for v in samples]
 frozen_plan=BASE/'waveform_sample_plan_frozen_private_v1.json'
 sample_receipt=json.loads((BASE/'waveform_sample_plan_receipt_v1.json').read_text())
 assert file_sha(frozen_plan)==sample_receipt['sample_plan_sha256']
 assert private_sample_plan==json.loads(frozen_plan.read_text())
 atomic_json(private_sample_plan,out/'sample_plan_private.json')
 stats={};details=[];stage_names=['Wake','N1','N2','N3','REM']
 for seed in plan['seeds']:
  for name in plan['methods']:
   print('Auditing',seed,name,'examples',len(samples),flush=True)
   run=RUNS/f'seed_{seed}'/name
   identity=next(v['identity'] for v in completed['private_runs'] if v['seed']==seed and v['variant']==name)
   verify_seal(run,identity)
   checkpoint=load_trusted(run/'best.pt')
   adapter=build_adapter(name,norm,banks[seed]).cuda();adapter.load_state_dict(checkpoint['state_dict']);adapter.eval();model=adapter.network
   rows=[];detail_path=out/f'{seed}_{name}_private.jsonl'
   with detail_path.open('x') as stream:
    for sample in samples:
     r=sample['recording'];x=sample['x'].cuda();truth=sample['stage']
     with torch.no_grad():
      ev=model(x,return_evidence=True);prob=ev.logits[0].softmax(-1);pred=int(prob.argmax())
      groups,intervals,dominance,similarities=declared_groups(model,x,ev)
      record={'seed':seed,'variant':name,'subject_id':r.subject,'recording_id':r.recording_id,'night':r.descriptor['night'],
       'original_epoch':sample['epoch'],'truth':truth,'prediction':pred,'probabilities':prob.tolist(),
       'groups_declared_before_effects':groups,'dominant_bands':dominance,'spectral_similarities':similarities,
       'interpretation':'No confirmed clinical event identity or physiological causal counterfactual is claimed.'}
      identity_meta=EpochIdentity(r.subject,r.recording_id,str(r.descriptor['night']),'val',sample['epoch'],r.file_hash,ev.physical_input_sha256[0],True)
      record['ledger']=source_ledger(ev,[identity_meta])
      row={'truth':truth,'prediction':pred,'confidence':float(prob.max()),'availability_mean':float(ev.availability[ev.valid].mean()) if ev.valid.any() else None,
       'events':len(intervals),'spectral_similarity_mean':float(np.mean(similarities)) if similarities else None,'groups':{}}
      if not intervals:
       record['status']='EVENTLESS_NO_INTERVAL_ATTRIBUTION';rows.append(row);stream.write(json.dumps(record,allow_nan=False)+'\n');continue
      physical=model.physical_uV(x)[0,0]
      seed_noise=20261018+int(hashlib.sha256(r.recording_id.encode()).hexdigest()[:6],16)+sample['epoch']
      transformed=transformations(model,x,seed_noise)
      bridge=local_evidence_report(model,x,groups,donor_specs=[{'strategy':'bridge'}],transformations=transformed)
      donor,donor_status=choose_donor(physical,intervals);record['background_donor_status']=donor_status
      background=None
      if donor is not None:
       background=local_evidence_report(model,x,groups,donor_specs=[{'strategy':'background','donor_interval':donor}],transformations=None)
      record['bridge']=bridge;record['background']=background
      row['background_donor_supported']=donor is not None
      row['stability']={k:v['witness_set_jaccard_distance'] for k,v in bridge['D_attribution_stability'].items()}
      for group_name,group in bridge['groups'].items():
       d=group['donors'][0];full=np.asarray(d['C_group_counterfactual_effect']);individual=np.asarray(d['B_individual_counterfactual_effects']);add=np.asarray(d['A_additive_contribution'])
       residual=max(abs(v) for v in d['reconciliation_residual'])
       if residual>1e-5:raise ValueError('Real counterfactual decomposition fails frozen 1e-5 tolerance')
       info={'group_effect_true_stage':float(full[truth]),'group_effect_predicted_stage':float(full[pred]),
        'individual_abs_effect_mean_true_stage':float(np.abs(individual[:,truth]).mean()),
        'group_abs_effect_true_stage':float(abs(full[truth])),
        'additive_score_predicted_stage':float(add[pred]),'additive_score_true_stage':float(add[truth]),
        'individual_vs_group_nonadditivity_l1':float(np.abs(full-individual.sum(0)).sum()),
        'additive_vs_full_l1':float(np.abs(add-full).sum()),'background_effect_l1':float(np.abs(d['background_change']).sum()),
        'fixed_event_path_l1':float(np.abs(d['internal_event_path_effect']).sum()),'reselection_l1':float(np.abs(d['reselection_change']).sum()),
        'reselection_material':d['flags']['reselection_material'],'unsupported_bridge':d['flags']['replacement_unsupported'],
        'reconciliation_max_abs':residual,'donor_spread_max':None,
        'replacement_confidence':float(max(d['diagnostics']['probabilities_after'])),
        'replacement_original_stage_probability':float(d['diagnostics']['probabilities_after'][truth]),
        'boundary_jump_change_max_uV':max(abs(v['boundary_jump_after_uV']-v['boundary_jump_before_uV']) for v in d['diagnostics']['replacements']),
        'energy_mismatch_mean':float(np.mean([v['energy_mismatch_relative'] for v in d['diagnostics']['replacements']])),
        'spectral_distortion_mean':float(np.mean([v['spectral_fraction_L1'] for v in d['diagnostics']['replacements']]))}
       if background is not None:
        other=background['groups'][group_name]['donors'][0]
        info['donor_spread_max']=float(np.max(np.abs(full-np.asarray(other['C_group_counterfactual_effect']))))
        info['background_donor_elementary_unsupported']=other['flags']['replacement_unsupported']
        residual2=max(abs(v) for v in other['reconciliation_residual'])
        if residual2>1e-5:raise ValueError('Background donor decomposition fails frozen tolerance')
        info['reconciliation_max_abs']=max(residual,residual2)
       row['groups'][group_name]=info
      if seed==123 and name=='q1' and sample['ordinal']==0:
       make_plot(out/'plots'/f'{r.recording_id}_{sample["epoch"]}_{stage_names[truth]}.png',physical,ev,truth,pred,stage_names)
      rows.append(row);stream.write(json.dumps(record,allow_nan=False)+'\n')
   summary={'examples':len(rows),'stage_counts':np.bincount([v['truth'] for v in rows],minlength=5).tolist(),
    'accuracy_on_selected_nonrepresentative_examples':float(np.mean([v['truth']==v['prediction'] for v in rows])),
    'confidence_on_selected_examples_mean':float(np.mean([v['confidence'] for v in rows])),
    'background_donor_supported_fraction':float(np.mean([v.get('background_donor_supported',False) for v in rows])),
    'eventless_examples':sum(v['events']==0 for v in rows),'by_group':{},'stability':{},
    'availability_mean':safe_mean([v['availability_mean'] for v in rows]),
    'selected_pair_spectral_similarity_mean':safe_mean([v['spectral_similarity_mean'] for v in rows])}
   for transform in ['tiny_noise_0.01uV','physical_gain_1.01','within_epoch_shift_10_samples']:
    values=[v['stability'][transform] for v in rows if 'stability' in v]
    summary['stability'][transform]={'mean_jaccard_distance':float(np.mean(values)) if values else None,'n':len(values)}
   for group_name in sorted({k for v in rows for k in v['groups']}):
    values=[v['groups'][group_name] for v in rows if group_name in v['groups']]
    gs={'n_examples':len(values)}
    for field in values[0]:
     good=[v[field] for v in values if v[field] is not None]
     gs[field+'_mean']=float(np.mean(good)) if good else None
     gs[field+'_n']=len(good)
    gs['reconciliation_max_abs_over_examples']=max(v['reconciliation_max_abs'] for v in values)
    spreads=[v['donor_spread_max'] for v in values if v['donor_spread_max'] is not None]
    gs['donor_spread_over_0.25_fraction_among_supported']=float(np.mean(np.asarray(spreads)>.25)) if spreads else None
    summary['by_group'][group_name]=gs
   stats[f'{seed}_{name}']=summary
   details.append({'seed':seed,'variant':name,'private_details_sha256':file_sha(detail_path)})
   atomic_json(rows,out/f'{seed}_{name}_safe_intermediate_rows.json')
   del model,adapter,checkpoint;torch.cuda.empty_cache()
 assert implementation_snapshot()==launch['implementation']
 _,_,_,_,_,after=load_sources(c);assert after==proofs
 report={'status':'POSTFIT_WAVEFORM_AUDIT_COMPLETE','analysis_plan_sha256':file_sha(plan_path),'analysis_code_sha256':file_sha(__file__),
  'parent_training_implementation_sha256':launch['implementation']['sha256'],'sample_plan_sha256':file_sha(frozen_plan),'examples_per_model':len(samples),'methods':plan['methods'],'seeds':plan['seeds'],
  'models':stats,'private_detail_sha256':details,'wall_seconds':time.perf_counter()-started,
  'clinical_event_identity':'NOT_ESTABLISHED','physiological_counterfactual_validity':'NOT_ESTABLISHED',
  'selection_caveat':'Earliest label-stratified original epochs; not representative full-grid performance or independent participants.',
  'donor_caveat':'Unchanged synthetic 3-uV background cap may reject real EEG; no relaxation or physiological validity assumption.',
  'source_proofs_unchanged':True,'training_executed':False,'reserved_access':False,'st_access':False,'shhs_access':False}
 with (BASE/'phase3_waveform_safe_results_v1.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print('Waveform audit complete; details and plots remain private',flush=True)
 return report
if __name__=='__main__':analyze()

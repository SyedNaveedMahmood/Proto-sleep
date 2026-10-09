"""P3B-only full-data source baselines; frozen gates and sealed epoch boundaries."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mist_evidence.data import blocks
from mist_evidence.runtime import (TrainConfig,train,seed_all,load_trusted,atomic_torch_save,
    tensor_digest,EpochPaused,pause_epoch)
from .protocol import read_frozen_split,load_development,P2_COMPLETION
from .provenance import atomic_json,atomic_text,sha256_file,fingerprint,git_info,environment
from .source_data import SourceRecording,training_bank,validate_bank
from .experiment import evaluate_source,reused_implementation_hash,source_input_proof,verify_outputs
from .baselines import make_baseline

SPLIT='3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84'
VERSION='p3b-sc-full-v1'
VARIANTS=['attnsleep','evidence','summary_only','raw_context']
SEEDS=[123,456,789]
DEFAULT_OUTPUT=Path('mist_transfer_runs/p3b/v1')


def validate_configuration(config):
    expected={'purpose':'P3B_AUTHORIZED_FULL_SC_TRAIN_VAL_V1','version':VERSION,
        'split_fingerprint':SPLIT,'variants':VARIANTS,'seeds':SEEDS,'threads':2,
        'amp':False,'deterministic':True,'optimizer':'AdamW','max_epochs':60,'patience':12,
        'lr':.0003,'weight_decay':.0001,'core_epochs':16,'encode_batch':16,'grad_clip':5.,
        'temporal_radius':0,'crf':False,'smoke':False,'reserved_access':False,
        'st_access':False,'shhs_access':False,'optional_augmentation':False,
        'gate_a_maximum_estimated_hours':24}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('P3B frozen full-data configuration violated')


def training_config(config,seed):
    validate_configuration(config)
    if seed not in SEEDS:raise ValueError('unapproved seed')
    return TrainConfig(epochs=config['max_epochs'],patience=config['patience'],lr=config['lr'],
        weight_decay=config['weight_decay'],core_epochs=config['core_epochs'],encode_batch=config['encode_batch'],
        grad_clip=config['grad_clip'],seed=seed,deterministic=True)


def verify_grid(train_records,val_records):
    for rows,role,people,nights,physical,scored in [
        (train_records,'train',15,29,79984,79025),(val_records,'val',4,8,21796,21789)]:
        if len(rows)!=nights or len({r.subject for r in rows})!=people or sum(len(r.y) for r in rows)!=physical or sum(int(r.scoring_mask.sum()) for r in rows)!=scored:
            raise ValueError('full-data subject/recording/physical/scored counts disagree')
        for r in rows:
            if r.split!=role or not np.array_equal(r.indices,np.arange(len(r.y))) or r.x.shape!=(len(r.y),1,3000) or not np.array_equal(r.scoring_mask,r.y>=0):
                raise ValueError('full-data geometry/mask/role contract violated')
    if {r.subject for r in train_records}&{r.subject for r in val_records}:raise ValueError('development participant leakage')
    items=blocks(train_records,16,0)
    if sum(hi-lo for _,_,_,lo,hi in items)!=79025:raise ValueError('scored training epoch loss coverage violated')
    return {'smoke':False,'train_subjects':15,'train_recordings':29,'train_physical_epochs':79984,
        'train_scored_epochs':79025,'val_subjects':4,'val_recordings':8,'val_physical_epochs':21796,
        'val_scored_epochs':21789,'optimizer_steps_per_epoch':len(items),'reserved_access':False}


def cuda_identity():
    if not torch.cuda.is_available():raise ValueError('P3B requires usable CUDA')
    import os
    return {'name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
        'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}


def check_implementation(identity):
    current=git_info();current.pop('dirty')
    if current!=identity['git'] or reused_implementation_hash()!=identity['reused_implementation_hash']:
        raise ValueError('active experiment implementation changed; stop and use a new versioned identity')


def verify_correctness(path,identity):
    proof=json.loads(Path(path).read_text())
    if proof['git_sha']!=identity['git']['sha'] or proof['implementation_hash']!=identity['git']['implementation_hash'] or proof['reused_implementation_hash']!=identity['reused_implementation_hash']:
        raise ValueError('correctness checks do not cover frozen implementation')
    if proof['gate']!='PASS' or not proof['cuda_exact_resume_all_four']:raise ValueError('Gate A correctness not passed')
    for name,digest in proof['logs'].items():
        if sha256_file(Path(name))!=digest:raise ValueError('correctness log digest changed')
    return proof


def verify_boundary(run):
    """Reject changed partial checkpoint state before trusted pickle loading."""
    files=sorted((run/'epochs').glob('epoch_*.json')) if (run/'epochs').exists() else []
    if (run/'last.pt').exists():
        if not files:raise ValueError('checkpoint without sealed epoch boundary')
        marker=json.loads(files[-1].read_text())
        for name,digest in marker['outputs'].items():
            if sha256_file(run/name)!=digest:raise ValueError('epoch-boundary artifact digest mismatch: '+name)
    return files


def measured_benchmark(run,variant,wall=None):
    boundary=json.loads((run/'epochs/epoch_001.json').read_text())
    boundary.update(variant=variant,seed=123,**json.loads((run/'parameter_counts.json').read_text()),
        call_wall_seconds=wall if wall is not None else boundary['resources']['epoch_wall_seconds'],
        call_wall_recovered_from_epoch_boundary=wall is None)
    atomic_json(boundary,run/'benchmark.json');return boundary


class EpochObserver:
    def __init__(self,identity,grid,output):self.identity=identity;self.grid=grid;self.output=output
    def __call__(self,run,signature,row,metrics,checkpoint_seconds):
        check_implementation(self.identity)
        if row['scored_training_epochs']!=self.grid['train_scored_epochs'] or row['optimizer_steps']!=self.grid['optimizer_steps_per_epoch'] or metrics['n_epochs']!=self.grid['val_scored_epochs'] or metrics['unscored_epochs']!=7:
            raise ValueError('epoch/validation coverage invariant failed')
        predicted=np.asarray(metrics['confusion_matrix']).sum(0)
        diagnostics={'predicted_class_counts':predicted.tolist(),'one_predicted_class':int((predicted>0).sum())==1,
            'largest_class_fraction':float(predicted.max()/predicted.sum()),'near_collapse_98_percent':bool(predicted.max()/predicted.sum()>=.98)}
        measured={**row,'last_and_history_checkpoint_seconds':checkpoint_seconds,
            'train_scored_epochs_per_second':row['scored_training_epochs']/row['train_seconds'],
            'val_physical_epochs_per_second':self.grid['val_physical_epochs']/row['validation_seconds']}
        measured['epoch_wall_seconds']=row['seconds']+checkpoint_seconds
        atomic_json({'signature':signature,'epoch':row['epoch'],'resources':measured,'metrics':metrics,
            'class_diagnostics':diagnostics,'reserved_access':False,
            'outputs':{name:sha256_file(run/name) for name in ['last.pt','best.pt','history.csv','provenance.json']}},
            run/'epochs'/f"epoch_{row['epoch']:03d}.json")
        if diagnostics['one_predicted_class'] or diagnostics['near_collapse_98_percent']:
            print('CLASS COLLAPSE DIAGNOSTIC (retained valid result): '+json.dumps(diagnostics),flush=True)
        if (self.output/'PAUSE_REQUESTED').exists():pause_epoch(run,signature,row['epoch'])


def seal_result(run,identity,seed,variant,counts,metrics):
    verify_boundary(run);last=load_trusted(run/'last.pt');best=load_trusted(run/'best.pt')
    done=json.loads((run/'COMPLETE.json').read_text())
    if tensor_digest(best['state_dict'])!=done['best_state_digest'] or done['signature']!=last['signature']:
        raise ValueError('completion/checkpoint state disagreement')
    if not all(torch.isfinite(v).all().item() for state in [last['state_dict'],best['state_dict']] for v in state.values()):
        raise FloatingPointError('nonfinite completed checkpoint')
    epochs=[json.loads(p.read_text()) for p in sorted((run/'epochs').glob('epoch_*.json'))]
    if len(epochs)!=last['epoch']:raise ValueError('missing measured epoch boundary')
    result={'variant':variant,'seed':seed,'run_fingerprint':identity['run_fingerprint'],
        **counts,'metrics':metrics,'best_epoch':done['best_epoch'],'actual_epochs':last['epoch'],
        'optimizer_steps':sum(h['optimizer_steps'] for h in last['history']),
        'training_scored_epoch_exposures':79025*last['epoch'],
        'epoch_wall_seconds':sum(e['resources']['epoch_wall_seconds'] for e in epochs),
        'train_seconds':sum(e['resources']['train_seconds'] for e in epochs),
        'validation_seconds':sum(e['resources']['validation_seconds'] for e in epochs),
        'train_io_seconds':sum(e['resources']['train_io_seconds'] for e in epochs),
        'val_io_seconds':sum(e['resources']['val_io_seconds'] for e in epochs),
        'peak_gpu_allocated_bytes':max(e['resources']['peak_gpu_allocated_bytes'] for e in epochs),
        'peak_gpu_reserved_bytes':max(e['resources']['peak_gpu_reserved_bytes'] for e in epochs),
        'peak_process_rss_KiB':max(e['resources']['peak_process_rss_KiB'] for e in epochs),
        'class_diagnostics_by_epoch':[e['class_diagnostics'] for e in epochs],
        'best_state_digest':done['best_state_digest'],'last_state_digest':tensor_digest(last['state_dict']),
        'finite':True,'reserved_access':False}
    result['train_scored_epochs_per_second']=result['training_scored_epoch_exposures']/result['train_seconds']
    atomic_json(result,run/'result.json')
    outputs=['result.json','last.pt','best.pt','COMPLETE.json','history.csv','validation_predictions.csv','provenance.json']
    outputs.extend(str(p.relative_to(run)) for p in sorted((run/'epochs').glob('epoch_*.json')))
    atomic_json({'run_fingerprint':identity['run_fingerprint'],'seed':seed,'variant':variant,
        'status':'VALIDATED_COMPLETE','outputs':{name:sha256_file(run/name) for name in outputs}},run/'VALIDATED_COMPLETE.json')
    return result


def prepare_sources(output,identity,frozen):
    train_rows=load_development(frozen,'train');val_rows=load_development(frozen,'val')
    proof=source_input_proof(train_rows+val_rows);p3a=Path('mist_transfer_runs/p3a/cuda_smoke')
    prior=json.loads((p3a/'COMPLETE.json').read_text());norm_path=output/'normalization.json'
    if not norm_path.exists():
        if sha256_file(p3a/'normalization.json')!=prior['outputs']['normalization.json']:raise ValueError('verified P3A normalization changed')
        saved=json.loads((p3a/'normalization.json').read_text())
        if saved['input_proof']!=proof:raise ValueError('normalization input proofs differ')
        atomic_json(saved,norm_path)
    saved=json.loads(norm_path.read_text());normalization=saved['statistics']
    if saved['input_proof']!=proof or normalization['physical_epoch_count']!=79984 or normalization['fit_role']!='train' or normalization['label_access'] or normalization['fit_subjects']!=sorted({r['subject_id'] for r in train_rows}) or normalization['source_hashes']!={r['recording_id']:r['source_sha256'] for r in train_rows}:
        raise ValueError('normalization TRAIN-only provenance violated')
    fitted=output/'fit_artifacts.receipt.json'
    if fitted.exists():verify_outputs(output,fitted,identity['run_fingerprint'])
    banks={}
    for seed in SEEDS:
        path=output/f'seed_{seed}'/'training_waveform_bank.pt'
        if path.exists():bank=load_trusted(path)
        elif seed==123:
            if sha256_file(p3a/'training_waveform_bank.pt')!=prior['outputs']['training_waveform_bank.pt']:raise ValueError('P3A TRAIN bank changed')
            bank=load_trusted(p3a/'training_waveform_bank.pt');atomic_torch_save(bank,path)
        else:
            with threadpool_limits(limits=2):bank=training_bank(train_rows,normalization,seed)
            atomic_torch_save(bank,path)
        if bank['seed']!=seed:raise ValueError('anchor seed mismatch')
        validate_bank(bank,train_rows,normalization,verify_waveforms=not fitted.exists());banks[seed]=bank
    if not fitted.exists():
        names=['normalization.json']+[f'seed_{seed}/training_waveform_bank.pt' for seed in SEEDS]
        atomic_json({'run_fingerprint':identity['run_fingerprint'],'outputs':{name:sha256_file(output/name) for name in names},'normalization_reused_from':'P3A','per_seed_banks_frozen_before_full_scores':True},fitted)
    train_records=[SourceRecording(r,normalization,smoke=False) for r in train_rows]
    val_records=[SourceRecording(r,normalization,smoke=False) for r in val_rows]
    grid=verify_grid(train_records,val_records);atomic_json(grid,output/'full_grid.json')
    return train_records,val_records,normalization,banks,grid,proof


def aggregate_results(results):
    aggregate={}
    scalar=['mean_subject_macro_f1','macro_f1','accuracy','kappa','balanced_accuracy','nll','brier_score','ece_15_equal_width_bins']
    for name in VARIANTS:
        rows=[r for r in results if r['variant']==name]
        if len(rows)!=3 or {r['seed'] for r in rows}!=set(SEEDS):raise ValueError('aggregate requires all three seeds')
        summary={key:{'mean':float(np.mean([r['metrics'][key] for r in rows])),
            'sd_across_seeds':float(np.std([r['metrics'][key] for r in rows],ddof=1))} for key in scalar}
        summary['per_stage_f1_mean']=np.mean([r['metrics']['per_stage_f1'] for r in rows],axis=0).tolist()
        summary['subject_f1_mean_across_seeds']={sid:float(np.mean([r['metrics']['subject_macro_f1'][sid] for r in rows])) for sid in rows[0]['metrics']['subject_macro_f1']}
        v=list(summary['subject_f1_mean_across_seeds'].values())
        summary['subject_variability']={'n_subjects':4,'sd_between_subject_seed_means':float(np.std(v,ddof=1)),
            'min':min(v),'max':max(v),'caveat':'Four VAL participants only; seeds are repeated optimization, not independent subjects or external transfer.'}
        aggregate[name]=summary
    return aggregate


def run_full(split_path,config_path,output=DEFAULT_OUTPUT,mode='gate-a',resume=False,
             authorize_full=False,correctness=None,dry_run=False):
    if not authorize_full and not dry_run:raise ValueError('explicit --authorize-full-sc-train-val required')
    output=Path(output);config=json.loads(Path(config_path).read_text());validate_configuration(config)
    frozen=read_frozen_split(split_path)
    if frozen['split_fingerprint']!=SPLIT or frozen['p2_completion_sha256']!=P2_COMPLETION:raise ValueError('reviewed split/P2 fingerprint differs')
    identity={'phase':'P3B','version':VERSION,'config':config,'split_fingerprint':SPLIT,
        'split_file_sha256':sha256_file(Path(split_path)),'git':git_info(),
        'reused_implementation_hash':reused_implementation_hash(),'environment':environment(),
        'cuda':cuda_identity(),'reserved_access':False,'data_scope':'all physical source TRAIN/VAL; never reserved descriptors'}
    dirty=identity['git'].pop('dirty');signature=fingerprint(identity);identity['run_fingerprint']=signature
    if dry_run:print(json.dumps({k:v for k,v in identity.items() if k!='environment'},indent=2));return 0
    if dirty:raise ValueError('commit/freeze P3B implementation before execution')
    if mode not in {'gate-a','gate-b'}:raise ValueError('unknown execution gate')
    correctness=Path(correctness or 'mist_transfer_runs/p3b/correctness.json');verified=verify_correctness(correctness,identity)
    canonical=Path(frozen['records'][0]['descriptor']).parent.parent
    if sha256_file(canonical/'COMPLETE.json')!=P2_COMPLETION or sha256_file(canonical/'manifest.json')!=frozen['canonical_manifest_sha256']:
        raise ValueError('canonical P2 metadata changed')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('P3B already running; one writer/GPU process only')
        provenance=output/'provenance.json'
        if provenance.exists():
            if not resume:raise FileExistsError('P3B output exists; use --resume')
            if json.loads(provenance.read_text())['run_fingerprint']!=signature:raise ValueError('full-run resume fingerprint mismatch')
        else:atomic_json({**identity,'correctness_receipt_sha256':sha256_file(correctness),'policy_frozen_before_full_scores':True},provenance)
        torch.set_num_threads(2);seed_all(123,True)
        tr,va,norm,banks,grid,input_proof=prepare_sources(output,identity,frozen)
        observer=EpochObserver(identity,grid,output)
        if (output/'COMPLETE.json').exists():
            verify_outputs(output,output/'COMPLETE.json',signature);print('VERIFIED COMPLETE P3B: all outputs unchanged',flush=True);return 0
        if mode=='gate-b':
            gate=json.loads((output/'gate_a.json').read_text())
            if gate['status']!='PASS' or gate['run_fingerprint']!=signature:raise ValueError('Gate B blocked until Gate A passes')
            verify_outputs(output,output/'gate_a.receipt.json',signature)
        results=[];benchmarks=[]
        for seed in ([123] if mode=='gate-a' else SEEDS):
            for name in VARIANTS:
                run=output/f'seed_{seed}'/name;check_implementation(identity)
                if (run/'VALIDATED_COMPLETE.json').exists():
                    verify_outputs(run,run/'VALIDATED_COMPLETE.json',signature)
                    results.append(json.loads((run/'result.json').read_text()));continue
                boundaries=verify_boundary(run)
                if mode=='gate-a' and boundaries:
                    if len(boundaries)!=1 or (run/'COMPLETE.json').exists():raise ValueError('benchmark must be a paused first epoch')
                    if not (run/'PAUSED.json').exists():
                        try:pause_epoch(run,json.loads(boundaries[0].read_text())['signature'],1)
                        except EpochPaused:pass
                    if not (run/'benchmark.json').exists():measured_benchmark(run,name)
                    benchmarks.append(json.loads((run/'benchmark.json').read_text()));continue
                seed_all(seed,True);model=make_baseline(name,banks[seed],norm)
                counts={'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
                atomic_json(counts,run/'parameter_counts.json');cfg=training_config(config,seed)
                print(f'P3B {mode}: seed={seed} model={name} full TRAIN={grid["train_physical_epochs"]} VAL={grid["val_physical_epochs"]} max=60 patience=12',flush=True)
                torch.cuda.empty_cache();torch.cuda.synchronize();started=time.perf_counter()
                try:
                    model,metrics,_=train(model,tr,va,cfg,run,banks[seed],
                        {'root_identity':identity,'normalization_fingerprint':fingerprint(norm),
                         'source_inputs':input_proof,'grid':grid,'seed':seed,'variant':name},torch.device('cuda'),
                        resume=bool(boundaries),evaluate_fn=evaluate_source,pause_after_epoch=1 if mode=='gate-a' else None,
                        epoch_callback=observer,monitor=True)
                except EpochPaused:
                    if mode=='gate-a':
                        benchmark=measured_benchmark(run,name,time.perf_counter()-started);benchmarks.append(benchmark)
                    if mode!='gate-a' or (output/'PAUSE_REQUESTED').exists():
                        print('P3B paused by operational request; NOT COMPLETE',flush=True);return 0
                else:
                    if mode=='gate-a':raise ValueError('initial epoch was incorrectly marked complete')
                    result=seal_result(run,identity,seed,name,counts,metrics);results.append(result)
                    print('VALIDATED COMPLETE '+json.dumps({k:result[k] for k in ['variant','seed','actual_epochs','best_epoch','optimizer_steps','epoch_wall_seconds']}),flush=True)
                del model;torch.cuda.empty_cache()
        if mode=='gate-a':
            # Resume reuses the originally measured costs rather than rerunning.
            benchmarks=[json.loads((output/f'seed_123/{name}/benchmark.json').read_text()) for name in VARIANTS]
            estimate=3*60*sum(b['resources']['epoch_wall_seconds'] for b in benchmarks)/3600
            gate={'status':'PASS' if estimate<=config['gate_a_maximum_estimated_hours'] else 'STOP_IMPRACTICAL_COST',
                'run_fingerprint':signature,'grid':grid,'benchmarks':benchmarks,
                'estimated_maximum_training_hours':estimate,'operational_ceiling_hours':config['gate_a_maximum_estimated_hours'],
                'correctness':verified,'configuration_unchanged':True,'initial_epochs_complete':False,'initial_epochs_resumable':True,
                'reserved_access':False,'note':'Cost estimate assumes all 60 epochs for all 12 runs; actual early stopping may be shorter.'}
            atomic_json(gate,output/'gate_a.json')
            names=['gate_a.json','full_grid.json','fit_artifacts.receipt.json']+[f'seed_123/{name}/benchmark.json' for name in VARIANTS]
            atomic_json({'run_fingerprint':signature,'outputs':{name:sha256_file(output/name) for name in names}},output/'gate_a.receipt.json')
            print('GATE A '+gate['status']+f'; estimated maximum {estimate:.2f} GPU hours',flush=True)
            return 0 if gate['status']=='PASS' else 3
        report={'phase':'P3B','status':'ALL_12_VALIDATED_COMPLETE','run_fingerprint':signature,'split_fingerprint':SPLIT,
            'implementation_commit':identity['git']['sha'],'grid':grid,'per_seed':results,'aggregate':aggregate_results(results),
            'reserved_access':False,'external_transfer_claim':False,'st_access':False,'shhs_access':False,'mcr_trained':False,
            'note':'Source VAL only, four participants. Models share maximum budget and early stopping policy; realized compute may differ.'}
        atomic_json(report,output/'results.json')
        names=['results.json','gate_a.json','gate_a.receipt.json','normalization.json','fit_artifacts.receipt.json','full_grid.json','provenance.json']
        for seed in SEEDS:
            names.append(f'seed_{seed}/training_waveform_bank.pt')
            for name in VARIANTS:
                run=output/f'seed_{seed}'/name;done=json.loads((run/'VALIDATED_COMPLETE.json').read_text())
                names.append(str((run/'VALIDATED_COMPLETE.json').relative_to(output)))
                names.extend(str((run/p).relative_to(output)) for p in done['outputs'])
        atomic_json({'run_fingerprint':signature,'status':report['status'],'outputs':{name:sha256_file(output/name) for name in names}},output/'COMPLETE.json')
        print('P3B ALL 12 VALIDATED COMPLETE; reserved/st/shhs access=False',flush=True)
        return 0

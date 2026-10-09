"""Matched source-only integration using v1 training/checkpoint machinery."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from mist_evidence.runtime import (TrainConfig, train, seed_all, atomic_torch_save,
    load_trusted, tensor_digest)
from .protocol import read_frozen_split, load_development
from .source_data import fit_normalization, training_bank, SourceRecording, validate_bank
from .preprocessing import SourceProofs
from .baselines import make_baseline
from .metrics import classification_metrics
from .provenance import atomic_json, atomic_text, fingerprint, sha256_file, environment, git_info


@torch.no_grad()
def evaluate_source(model,recordings,device,batch):
    if any(r.split!='val' for r in recordings):raise ValueError('source validation only; no TRAIN/TEST selection metric')
    if model.cfg.radius!=0:raise ValueError('streamed P3A evaluation requires epoch-only model')
    model.eval();truth=[];predictions=[];people=[];masks=[];rows=[];loss_sum=0.;count=0;probabilities=[]
    for r in recordings:
        for lo in range(0,len(r.y),batch):
            hi=min(len(r.y),lo+batch);x=torch.from_numpy(r.x[lo:hi]).to(device)
            logits=model.emissions(model.encode(x)[None])[0]
            if logits.shape!=(hi-lo,5) or not torch.isfinite(logits).all():raise FloatingPointError('invalid validation logits')
            pred,prob=model.predict(logits)
            if not torch.isfinite(prob).all() or not torch.allclose(prob.sum(-1),torch.ones(hi-lo,device=device),atol=1e-5):
                raise FloatingPointError('invalid validation probabilities')
            y=torch.from_numpy(r.y[lo:hi]).to(device);mask=torch.from_numpy(r.scoring_mask[lo:hi]).to(device)
            if mask.any():
                loss=F.cross_entropy(logits[mask].float(),y[mask],reduction='sum')
                if not torch.isfinite(loss):raise FloatingPointError('invalid validation loss')
                loss_sum+=float(loss);count+=int(mask.sum())
            p=pred.cpu().numpy();pr=prob.cpu().numpy()
            truth.extend(r.y[lo:hi].tolist());predictions.extend(p.tolist());people.extend([r.subject]*(hi-lo));masks.extend(r.scoring_mask[lo:hi].tolist())
            probabilities.extend(pr.tolist())
            for j in range(hi-lo):
                rows.append({'recording_id':r.recording_id,'subject_id':r.subject,'original_epoch_index':int(r.indices[lo+j]),
                    'start_seconds':int(r.indices[lo+j])*30,'scored':bool(r.scoring_mask[lo+j]),
                    'truth':int(r.y[lo+j]) if r.scoring_mask[lo+j] else None,'prediction':int(p[j])})
    metrics=classification_metrics(truth,predictions,people,masks);metrics['nll']=loss_sum/count
    selected=np.asarray(masks);y=np.asarray(truth)[selected];pr=np.asarray(probabilities)[selected]
    metrics['brier_score']=float(((pr-np.eye(5)[y])**2).sum(1).mean())
    confidence=pr.max(1);correct=pr.argmax(1)==y;ece=0.
    for a,b in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        idx=(confidence>=a)&(confidence<b if b<1 else confidence<=b)
        if idx.any():ece+=float(idx.mean()*abs(confidence[idx].mean()-correct[idx].mean()))
    metrics['ece_15_equal_width_bins']=ece
    return metrics,rows


def reused_implementation_hash():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'src/mist_evidence').glob('*.py'))+[root/'src/protosleep'/name for name in ['attnsleep.py','config.py','utils.py']]
    return fingerprint({str(p.relative_to(root)):sha256_file(p) for p in sorted(files)})


def source_input_proof(records):
    return {r['recording_id']:{'source_sha256':r['source_sha256'],'annotation_sha256':r['annotation_sha256'],
             'stat':r['source_stat_proof'],'annotation_stat':r['annotation_stat_proof']} for r in records}


def verify_outputs(output,marker,signature):
    done=json.loads(Path(marker).read_text())
    if done['run_fingerprint']!=signature:raise ValueError('completion fingerprint changed')
    for name,digest in done['outputs'].items():
        if sha256_file(Path(output)/name)!=digest:raise ValueError('completed output digest changed: '+name)


def run_integration(split_path,config_path,output,device='cuda',resume=False,dry_run=False):
    start=time.perf_counter();output=Path(output);frozen=read_frozen_split(split_path)
    config=json.loads(Path(config_path).read_text())
    if config['purpose']!='P3A_DEVELOPMENT_ONLY_SMOKE' or config['training']['epochs'] not in {1,2}:
        raise ValueError('P3A runner permits only one/two-epoch development integration, not P3B')
    if config['variants']!=['attnsleep','evidence','summary_only','raw_context'] or config['amp']:
        raise ValueError('four matched FP32 integration baselines required; AMP unvalidated')
    cfg=TrainConfig(**config['training'])
    if not cfg.deterministic or config['threads']<1:raise ValueError('deterministic development integration required')
    if config['split_seed']!=frozen['seed']:raise ValueError('frozen split seed differs')
    identity={'phase':'P3A','split_fingerprint':frozen['split_fingerprint'],
        'split_file_sha256':sha256_file(Path(split_path)),'config':config,'git':git_info(),
        'reused_implementation_hash':reused_implementation_hash(),'environment':environment(),
        'device':device,'reserved_test_access':False,'training_scope':'source TRAIN/VAL geometry-only smoke windows'}
    identity['git'].pop('dirty');signature=fingerprint(identity)
    if dry_run:
        print(json.dumps({**identity,'run_fingerprint':signature,'subject_roles':dict(Counter(frozen['subject_roles'].values())),
            'recording_roles':dict(Counter(r['split'] for r in frozen['records'])),'source_data_opened':False},indent=2));return 0
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('real P3A integration requires usable CUDA; no silent CPU fallback')
    output.mkdir(parents=True,exist_ok=True);provenance=output/'provenance.json'
    if provenance.exists():
        if not resume:raise FileExistsError('integration output exists; use --resume')
        if json.loads(provenance.read_text())['run_fingerprint']!=signature:raise ValueError('integration resume fingerprint mismatch')
    else:atomic_json({**identity,'run_fingerprint':signature},provenance)
    torch.set_num_threads(config['threads']);seed_all(cfg.seed,True)
    train_rows=load_development(frozen,'train');val_rows=load_development(frozen,'val')
    # Completed resume verifies unchanged TRAIN/VAL sources and outputs only;
    # it does not rerun training or overwrite measured runtime/throughput.
    if (output/'COMPLETE.json').exists():
        verify_outputs(output,output/'COMPLETE.json',signature)
        proofs=SourceProofs()
        for r in train_rows+val_rows:
            for path,sha,stat in [('path','source_sha256','source_stat_proof'),('annotation_source','annotation_sha256','annotation_stat_proof')]:
                proofs.checked[str(Path(r[path]).resolve())]=r[stat];proofs.verify(r[path],r[sha])
        print('VERIFIED COMPLETE P3A: output digests and development sources unchanged; no training/test access',flush=True)
        return 0
    input_proof=source_input_proof(train_rows+val_rows);artifact=output/'normalization.json'
    fitted=output/'fit_artifacts.receipt.json'
    if fitted.exists():verify_outputs(output,fitted,signature)
    if artifact.exists():
        normalization=json.loads(artifact.read_text())
        if normalization['input_proof']!=input_proof:raise ValueError('normalization source proof changed')
        normalization=normalization['statistics']
    else:
        normalization=fit_normalization(train_rows)
        atomic_json({'input_proof':input_proof,'statistics':normalization},artifact)
    if normalization['fit_subjects']!=sorted({r['subject_id'] for r in train_rows}) or normalization['source_hashes']!={r['recording_id']:r['source_sha256'] for r in train_rows}:
        raise ValueError('normalization includes non-TRAIN source')
    bank_path=output/'training_waveform_bank.pt'
    if bank_path.exists():bank=load_trusted(bank_path)
    else:
        with threadpool_limits(limits=config['threads']):bank=training_bank(train_rows,normalization,cfg.seed)
        atomic_torch_save(bank,bank_path)
    validate_bank(bank,train_rows,normalization,verify_waveforms=not fitted.exists())
    if not fitted.exists():atomic_json({'run_fingerprint':signature,'outputs':{name:sha256_file(output/name) for name in ['normalization.json','training_waveform_bank.pt']}},fitted)
    train_records=[SourceRecording(r,normalization,smoke=True) for r in train_rows]
    val_records=[SourceRecording(r,normalization,smoke=True) for r in val_rows]
    input_identity={**identity,'run_fingerprint':signature,'normalization':normalization,
        'source_input_proof':input_proof,'smoke_original_indices':{r.recording_id:r.indices.tolist() for r in train_records+val_records},
        'data_mode':'bounded smoke windows only; same windows/steps all four; no final performance'}
    results=[]
    for name in config['variants']:
        seed_all(cfg.seed,True);model=make_baseline(name,bank,normalization)
        total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
        model,metrics,_=train(model,train_records,val_records,cfg,output/name,bank,
            {**input_identity,'baseline':name},torch.device('cuda'),resume=resume,evaluate_fn=evaluate_source)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0
        last=load_trusted(output/name/'last.pt');steps=sum(h['optimizer_steps'] for h in last['history'])
        processed=sum(r.scoring_mask.sum() for r in train_records)*len(last['history'])
        result={'baseline':name,'parameter_count':total,'trainable_parameters':trainable,
            'metrics':metrics,'seconds':elapsed,'optimizer_steps':steps,'trained_scored_epochs':int(processed),
            'scored_training_epochs_per_second':float(processed/elapsed),
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
            'last_state_digest':tensor_digest(last['state_dict']),'finite_state':all(torch.isfinite(p).all().item() for p in model.state_dict().values()),
            'amp':False,'development_only':True,'best_epoch':last['best_epoch']}
        if not result['finite_state']:raise FloatingPointError('nonfinite model checkpoint')
        results.append(result);print(json.dumps(result),flush=True)
        del model;torch.cuda.empty_cache()
    if len({r['optimizer_steps'] for r in results})!=1 or len({r['trained_scored_epochs'] for r in results})!=1:
        raise ValueError('baseline training budgets differ')
    report={'gate':'PASS_DEVELOPMENT_INTEGRATION_ONLY','run_fingerprint':signature,
        'split_fingerprint':frozen['split_fingerprint'],'results':results,'normalization':normalization,
        'gpu':torch.cuda.get_device_name(),'cuda_version':torch.version.cuda,'elapsed_seconds':time.perf_counter()-start,
        'train_subjects':len({r.subject for r in train_records}),'val_subjects':len({r.subject for r in val_records}),
        'reserved_subjects':55,'reserved_test_access':False,'physical_train_smoke_epochs':sum(len(r.y) for r in train_records),
        'physical_val_smoke_epochs':sum(len(r.y) for r in val_records),'anchor_metadata':bank['metadata'],
        'note':'one smoke epoch over geometry-selected development windows; scores are not final performance',
        'p3b_data_and_adapter_eligibility':'VERIFIED_CORE_FOUR; full training still requires separate authorization',
        'st_preprocessed':False,'shhs_accessed':False,'mcr_trained':False}
    atomic_json(report,output/'validation.json')
    outputs=['validation.json','normalization.json','training_waveform_bank.pt','fit_artifacts.receipt.json']
    for name in config['variants']:outputs.extend(name+'/'+f for f in ['last.pt','best.pt','COMPLETE.json','history.csv','validation_predictions.csv','provenance.json'])
    atomic_json({'run_fingerprint':signature,'outputs':{name:sha256_file(output/name) for name in outputs},
                'gate':report['gate'],'reserved_test_access':False},output/'COMPLETE.json')
    atomic_text('P3A CUDA development-only integration PASS; scores are not final performance.\n'+
        'Split: '+frozen['split_fingerprint']+'\n'+json.dumps([{k:v for k,v in r.items() if k!='metrics'} for r in results],indent=2)+'\n',output/'FINAL_COPY_PASTE.txt')
    return 0

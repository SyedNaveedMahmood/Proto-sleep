"""Prepared source adapter and independently reviewed future launch gates.

Default CLI is metadata-only. This phase does not create real approval receipts.
"""
from dataclasses import asdict,dataclass
import json
from pathlib import Path

import torch
from torch import nn

from mist_cove.phase2.adapter import SourceAdapter
from mist_cove.phase2.provenance import digest,file_sha,implementation_snapshot
from mist_cove.phase2.training import sealed_train,verify_seal
from mist_evidence.runtime import TrainConfig,seed_all,atomic_json,EpochPaused
from .model import make_model


@dataclass(frozen=True)
class SourceConfig:
    method:str
    mean_uV:float
    std_uV:float
    radius:int=0
    use_crf:bool=False
    def dictionary(self):return {'version':'mist-cove-phase4-source-adapter-v1',**asdict(self)}


class Adapter(nn.Module):
    def __init__(self,method,normalization):
        super().__init__();self.cfg=SourceConfig(method,normalization['mean_uV'],normalization['std_uV'])
        self.network=make_model(method,mean_uV=self.cfg.mean_uV,std_uV=self.cfg.std_uV)
    def encode(self,x):return self.network(x)
    def emissions(self,features):return features
    loss=SourceAdapter.loss
    predict=SourceAdapter.predict


def factory(method,normalization,bank=None):
    if method in {'raw_context','evidence'}:
        from mist_transfer.baselines import make_baseline
        if bank is None:raise ValueError('verified TRAIN bank required')
        return make_baseline(method,bank,normalization)
    return Adapter(method,normalization)


def validate_config(c):
    expected={'seeds':[123,456,789],'variants':['M0','M1','M2','M3','M2_rms','raw_context','evidence'],
      'training':{'learning_rate':.0003,'weight_decay':.0001,'epochs':60,'patience':12,'core_epochs':16,'encode_batch':16,'grad_clip':5.,'radius':0,'use_crf':False,'amp':False,'tf32':False}}
    if any(c.get(k)!=v for k,v in expected.items()) or c.get('scope')!='PREPARED_SC_TRAIN_VAL_NOT_AUTHORIZED':
        raise ValueError('frozen future source comparison changed')
    if c.get('split_file_sha256')!='1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5' or c.get('split_fingerprint')!='3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84':
        raise ValueError('frozen split changed')
    if c.get('input')!={'channel':'EEG Fpz-Cz','units':'uV','sample_rate':100,'samples':3000,'physical_grid':'COMPLETE_ORIGINAL','mask_unscored_only_in_loss_metrics':True} or c.get('maximum_gpu_hours')!=24 or c.get('real_data_authorized') is not False:
        raise ValueError('source physical contract/authorization/budget changed')
    return c


def launch_gate(c,proposal_path,review_path,*,authorized=False,gate='a'):
    if not authorized:raise ValueError('separate real-source user authorization required before file access')
    validate_config(c)
    review=json.loads(Path(review_path).read_text());proposal=json.loads(Path(proposal_path).read_text())
    if gate not in {'a','b'} or review.get('decision')!=f'APPROVE_PHASE_IV_GATE_{gate.upper()}' or review.get('scope')!='SC_TRAIN_VAL_ONLY_PHASE_IV' or not review.get('actual_user_reference'):
        raise ValueError('actual separate Phase IV gate review required')
    if review.get('proposal_sha256')!=file_sha(proposal_path) or proposal.get('config_sha256')!=digest(c) or proposal.get('implementation_sha256')!=implementation_snapshot()['sha256']:
        raise ValueError('review/config/implementation mismatch')
    if proposal.get('status')!='REAL_PHASE_IV_PREFLIGHT_PASS' or not proposal.get('source_proofs') or not proposal.get('grid'):
        raise ValueError('fresh real preflight with source proofs required')
    for name in ['cpu','cuda','exact_resume']:
        item=proposal.get('correctness',{}).get(name,{})
        if item.get('status')!='PASS' or item.get('implementation_sha256')!=implementation_snapshot()['sha256'] or not item.get('path') or file_sha(item['path'])!=item.get('sha256'):
            raise ValueError('environment-bound correctness evidence missing')
    return proposal


def _run_reviewed(c,proposal_path,review_path,output,*,authorized=False,gate='a',loader=None,trainer=None):
    proposal=launch_gate(c,proposal_path,review_path,authorized=authorized,gate=gate)
    output=Path(output).resolve();private=(Path.cwd()/'mist_transfer_runs/mist_cove').resolve()
    if not output.is_relative_to(private):raise ValueError('real outputs must remain ignored and private')
    # The default canonical loader is reached only after all human/code gates.
    if loader is None:
        from mist_cove.phase3.source import load_sources,require_idle_gpu,reconciled_environment
        require_idle_gpu()
        from mist_cove.phase2.training import configure_precision
        configure_precision('cuda');torch.set_num_threads(2)
        if proposal.get('environment')!=reconciled_environment():raise ValueError('source environment changed')
        prior=json.loads(Path('configs/mist_cove_phase3_source_v1.json').read_text())
        loader=lambda:load_sources(prior)
    trainer=sealed_train if trainer is None else trainer
    if gate=='b':
        boundary=json.loads((output/'gate_a.json').read_text())
        if boundary['maximum_gpu_hours']>24 or boundary['proposal_sha256']!=file_sha(proposal_path):raise ValueError('Gate A budget/identity failed')
        for name,proof in boundary['boundaries'].items():
            if file_sha(output/'seed_123'/name/'gate_a_boundary.json')!=proof:raise ValueError('Gate A boundary changed')
    train,val,norm,banks,grid,proofs=loader()
    if grid!=proposal['grid'] or proofs!=proposal['source_proofs']:raise ValueError('source preflight proofs changed')
    results=[];boundaries={};seconds=[]
    for seed in ([123] if gate=='a' else c['seeds']):
        for name in c['variants']:
            seed_all(seed,True);model=factory(name,norm,banks.get(seed));run=output/f'seed_{seed}'/name
            cfg=TrainConfig(epochs=60,patience=12,core_epochs=16,encode_batch=16,lr=.0003,weight_decay=.0001,grad_clip=5.,seed=seed)
            provenance={'phase':'IV','config':c,'proposal_sha256':file_sha(proposal_path),'source_proofs':proofs,'grid':grid}
            if gate=='a' and (run/'EPOCH_SEAL.json').exists():raise ValueError('preserve Gate A pause; do not replay')
            try:
                result=trainer(model,train,val,cfg,run,provenance=provenance,device='cuda',bank=banks.get(seed),resume=(run/'EPOCH_SEAL.json').exists(),pause_after_epoch=1 if gate=='a' else None)
                results.append({'seed':seed,'method':name,'result':result})
            except EpochPaused:
                if gate!='a':raise
            if gate=='a':
                seal=json.loads((run/'EPOCH_SEAL.json').read_text())
                atomic_json(seal,run/'gate_a_boundary.json');boundaries[name]=file_sha(run/'gate_a_boundary.json')
                seconds.append(seal['resources']['seconds']+seal['checkpoint_write_seconds'])
    if gate=='a':
        maximum=180*sum(seconds)/3600
        atomic_json({'proposal_sha256':file_sha(proposal_path),'maximum_gpu_hours':maximum,'boundaries':boundaries,'separate_gate_b_review_required':True},output/'gate_a.json')
        if maximum>24:raise ValueError('maximum-budget estimate exceeds ceiling; stop')
    return results


def run_reviewed(c,proposal_path,review_path,output,*,authorized=False,gate='a',loader=None,trainer=None):
    # Fail before creating even a lock file when authorization is absent.
    launch_gate(c,proposal_path,review_path,authorized=authorized,gate=gate)
    kwargs={'authorized':authorized,'gate':gate,'loader':loader,'trainer':trainer}
    if loader is not None: # Explicit synthetic integration injection only.
        return _run_reviewed(c,proposal_path,review_path,output,**kwargs)
    import fcntl
    from mist_cove.phase3.source import PRIVATE
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'gpu.writer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_reviewed(c,proposal_path,review_path,output,**kwargs)

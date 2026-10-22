"""Exact P3B candidate only. Public results contain statuses and digests, no values."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .security import authenticated_snapshot,restricted_load,IntegrityError,sha256
from mist_cove.phase11a.provenance import parse_publication_manifest

ROOT=Path(__file__).resolve().parents[3]
P3B=ROOT/'mist_transfer_runs/p3b/v1'
ROOT_SHA='905ba78cf8326594a431ce7a71ce9624c35814d2e76c94f57abda8afbe7c7e7d'
CHILD_SHA='6029425769c825cc621832efa33b86ff3610a6446fe1801f82fd1d9fbe8707bf'
CHECKPOINT_SHA='f41b768c0ba81547872399a000f4a56338cc34f802bd49522fabb6bd7a56d147'
STATE_SHA='1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d'
RUN='a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b'
IMPL='661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23'
CHILD='seed_123/raw_context/VALIDATED_COMPLETE.json'
BEST='seed_123/raw_context/best.pt'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def named_state_digest(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_source():
    checked={}
    for name in ['src/mist_evidence/model.py','src/mist_evidence/cli.py','src/mist_transfer/baselines.py']:
        original=subprocess.check_output(['git','--no-replace-objects','show',IMPL+':'+name],cwd=ROOT)
        actual=(ROOT/name).read_bytes()
        if actual!=original:raise IntegrityError('original model construction source differs')
        checked[name]=sha256(actual)
    return checked


def authenticated_receipts():
    root_path=P3B/'COMPLETE.json';child_path=P3B/CHILD
    with authenticated_snapshot(root_path,ROOT_SHA,allowlist=[root_path],limit=2*1024*1024) as snap:
        root=parse_publication_manifest(snap.data,'p3b-root-completion-v1')
    if root['run_fingerprint']!=RUN or root['outputs'].get(CHILD)!=CHILD_SHA:raise IntegrityError('root selected run/child identity differs')
    with authenticated_snapshot(child_path,CHILD_SHA,allowlist=[child_path],limit=2*1024*1024) as snap:
        child=parse_publication_manifest(snap.data,'p3b-validated-completion-v1')
    if child['run_fingerprint']!=RUN or child['outputs'].get('best.pt')!=CHECKPOINT_SHA or root['outputs'].get(BEST)!=CHECKPOINT_SHA:
        raise IntegrityError('selected original checkpoint reference differs')
    return root,child


def authenticate_model():
    import torch
    from mist_evidence.model import EvidenceModel,ModelConfig,STAGES
    start=time.perf_counter();sources=verify_source();root,child=authenticated_receipts();path=P3B/BEST
    with authenticated_snapshot(path,CHECKPOINT_SHA,allowlist=[path]) as snap:payload=restricted_load(snap)
    if type(payload) is not dict or set(payload)!={'state_dict','model_config','bank','signature','best_epoch','metrics'}:
        raise IntegrityError('original best checkpoint schema differs')
    expected=ModelConfig(radius=0,use_crf=False,raw_control=True).dictionary()
    config=payload['model_config']
    if type(config) is not dict or {**config,'scales':tuple(config.get('scales',()))}!=expected:
        raise IntegrityError('original model configuration differs')
    bank=payload['bank'];state=payload['state_dict']
    if not isinstance(bank,dict) or bank.get('seed')!=123:raise IntegrityError('selected bank seed differs')
    if bank['config']!=ModelConfig(radius=0,use_crf=False).dictionary():raise IntegrityError('bank construction config differs')
    if any(type(t) is not torch.Tensor or t.dtype!=torch.float32 or t.device.type!='cpu' or not torch.isfinite(t).all() for t in state.values()):
        raise IntegrityError('state tensor type/dtype/device/finiteness')
    if named_state_digest(state)!=STATE_SHA:raise IntegrityError('original named-state digest mismatch')
    with torch.random.fork_rng(devices=[]):
        model=EvidenceModel(ModelConfig(**config),bank['waveforms'],tuple(bank['amplitude_stats'])).eval()
    reference=model.state_dict()
    if list(state)!=list(reference) or any(state[k].shape!=reference[k].shape or state[k].dtype!=reference[k].dtype for k in state):
        raise IntegrityError('state keys/shapes/dtypes differ from exact constructor')
    model.load_state_dict(state,strict=True)
    if named_state_digest(model.state_dict())!=STATE_SHA:raise IntegrityError('loaded state altered')
    if any(not torch.equal(w,b.waveforms) for w,b in zip(bank['waveforms'],model.banks)):
        raise IntegrityError('embedded bank/state buffer disagreement')
    if not torch.equal(model.amplitude_stats,torch.tensor(bank['amplitude_stats'],dtype=torch.float32)):
        raise IntegrityError('amplitude metadata/buffer disagreement')
    if tuple(STAGES)!=('Wake','N1','N2','N3','REM') or len(model.feature_names)!=209 or model.crf is not None or model.cfg.radius!=0:
        raise IntegrityError('stage/feature/temporal identity differs')
    counts=(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.parameters() if p.requires_grad))
    if counts!=(34197,33957):raise IntegrityError('actual parameter count differs')
    if type(payload['best_epoch']) is not int or payload['best_epoch']!=28:raise IntegrityError('selected epoch differs')
    receipt={'status':'PASS','checkpoint_byte_sha256':CHECKPOINT_SHA,'named_state_digest':STATE_SHA,'restricted_load':True,'custom_globals':False,'torch_version':torch.__version__,'source_hashes':sources,'parameters':counts[0],'originally_trainable_parameters':counts[1],'features':len(model.feature_names),'stages':list(STAGES),'best_epoch':payload['best_epoch'],'bank_waveforms_digest':named_state_digest({str(i):w for i,w in enumerate(bank['waveforms'])}),'bank_metadata_digest':sha256(canonical({k:v for k,v in bank.items() if k!='waveforms'})),'state_unchanged':True,'wall_seconds':time.perf_counter()-start,'real_EEG_accessed':False}
    return model,payload,root,child,receipt


def lineage(payload,root,child):
    """Read only selected root-bound metadata; never invoke a source/EDF loader."""
    from mist_cove.phase11a.provenance import strict_json
    from mist_transfer.protocol import validate_role_records
    allowed=['provenance.json','normalization.json','fit_artifacts.receipt.json','seed_123/raw_context/provenance.json','seed_123/raw_context/COMPLETE.json']
    artifacts={};digests={}
    for key in allowed:
        expected=root['outputs'].get(key)
        if expected is None:raise IntegrityError('required root-bound selected metadata missing')
        path=P3B/key
        with authenticated_snapshot(path,expected,allowlist=[path],limit=2*1024*1024) as snap:
            artifacts[key]=strict_json(snap.data)
        digests[key]=expected
    split_path=ROOT/'mist_transfer_runs/p3a/frozen_source_split.json'
    split_sha='1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5'
    with authenticated_snapshot(split_path,split_sha,allowlist=[split_path],limit=2*1024*1024) as snap:
        frozen=strict_json(snap.data)
    fp=frozen['split_fingerprint']
    if fp!='3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84' or sha256(canonical({k:v for k,v in frozen.items() if k!='split_fingerprint'}))!=fp:
        raise IntegrityError('frozen split fingerprint differs')
    validate_role_records(frozen['records'])
    if any(r['split']!=frozen['subject_roles'][r['subject_id']] for r in frozen['records']):raise IntegrityError('role linkage differs')
    training=[r for r in frozen['records'] if r['split']=='train'];validation=[r for r in frozen['records'] if r['split']=='val']
    if len(training)!=29 or len(validation)!=8 or len({r['subject_id'] for r in training})!=15 or len({r['subject_id'] for r in validation})!=4:
        raise IntegrityError('frozen development cohort differs')
    norm=artifacts['normalization.json']['statistics'];runprov=artifacts['seed_123/raw_context/provenance.json'];rootprov=artifacts['provenance.json'];fit=artifacts['fit_artifacts.receipt.json'];complete=artifacts['seed_123/raw_context/COMPLETE.json']
    if rootprov['run_fingerprint']!=RUN or rootprov['split_fingerprint']!=fp or rootprov['split_file_sha256']!=split_sha:
        raise IntegrityError('root provenance run/split differs')
    if rootprov['git']['sha']!=IMPL:raise IntegrityError('historical implementation differs')
    if sha256(canonical({k:v for k,v in rootprov.items() if k not in {'run_fingerprint','correctness_receipt_sha256','policy_frozen_before_full_scores'}}))!=RUN:
        raise IntegrityError('original run fingerprint reconstruction differs')
    if norm['fit_role']!='train' or norm['label_access'] or norm['scored_epoch_selection'] or norm['physical_epoch_count']!=79984 or norm['fit_subjects']!=sorted({r['subject_id'] for r in training}) or norm['source_hashes']!={r['recording_id']:r['source_sha256'] for r in training}:
        raise IntegrityError('normalization TRAIN fitting lineage differs')
    if fit['run_fingerprint']!=RUN or fit['normalization_reused_from']!='P3A' or not fit['per_seed_banks_frozen_before_full_scores'] or fit['outputs']['normalization.json']!=digests['normalization.json']:
        raise IntegrityError('fit receipt lineage differs')
    if fit['outputs']['seed_123/training_waveform_bank.pt']!=root['outputs']['seed_123/training_waveform_bank.pt']:
        raise IntegrityError('selected original bank reference differs')
    signature=sha256(canonical({k:v for k,v in runprov.items() if k!='signature'}))
    if signature!=runprov['signature'] or signature!=payload['signature'] or complete['signature']!=signature or complete['best_state_digest']!=STATE_SHA or complete['best_epoch']!=payload['best_epoch']:
        raise IntegrityError('selected model creation/completion signature differs')
    bank=payload['bank']
    if runprov['bank_digest']!=named_state_digest({str(i):w for i,w in enumerate(bank['waveforms'])}) or runprov['bank_metadata_digest']!=sha256(canonical({k:v for k,v in bank.items() if k!='waveforms'})):
        raise IntegrityError('embedded bank original provenance differs')
    if bank['normalization_fingerprint']!=sha256(canonical(norm)) or bank['training_subjects']!=norm['fit_subjects']:
        raise IntegrityError('embedded bank TRAIN/normalization binding differs')
    by_path={r['path']:r for r in training}
    for group in bank['metadata']:
        for anchor in group:
            ref=by_path.get(anchor['source_path'])
            if ref is None or anchor['split']!='train' or anchor['subject']!=ref['subject_id'] or anchor['source_sha256']!=ref['source_sha256'] or anchor['annotation_sha256']!=ref['annotation_sha256']:
                raise IntegrityError('embedded bank source role/proof differs')
    result={'status':'PASS_METADATA_LINEAGE','root_linked_metadata_hashes':digests,'split_sha256':split_sha,'split_fingerprint':fp,'TRAIN_participants':15,'VAL_participants':4,'TRAIN_recordings':29,'VAL_recordings':8,'normalization_fit_role':'train','normalization_physical_epochs':79984,'normalization_label_access':False,'P3A_reuse':'HASH_BOUND_P3B_RECEIPT_ASSERTION','bank_metadata_and_waveforms_match_original_run':True,'original_run_fingerprint_reconstructed':True,'original_model_signature_reconstructed':True,'environment_compatibility':'STATE_IDENTITY_PASS; numerical behavior checked separately under patched CPU runtime','creation_time_independent_signer':'NOT ESTABLISHED','source_crop_waveform_recomparison':'NOT PERFORMED','real_EEG_accessed':False}
    return frozen,norm,result

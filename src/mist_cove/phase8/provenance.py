"""Git-anchored prospective-only loading and fail-closed Phase VIII freezes."""
import json
from pathlib import Path
import subprocess
import time
import torch
from mist_cove.phase2.provenance import file_sha,digest,environment
from mist_cove.phase7.integrity import confined_file, verified_bytes, require_scope, ROOT
from mist_cove.phase7.prospective import restore, optimizer
from mist_cove.phase7.robustness import ClippedModel
from mist_cove.phase4.model import make_model
from mist_evidence.runtime import tensor_digest

BASE='4262a1a9680f9e8c1747363553204aaeb1ce976b'
PRIVATE=ROOT/'mist_transfer_runs/mist_cove/phase8_v1'
CONFIG=ROOT/'configs/mist_cove_phase8_synthetic_v1.json'


def anchored(relative):
    raw=subprocess.check_output(['git','show',f'{BASE}:{relative}'],cwd=ROOT)
    import hashlib
    return verified_bytes(ROOT,relative,hashlib.sha256(raw).hexdigest())


def output_directory(path,scope='SYNTHETIC_ONLY'):
    require_scope(scope)
    path=Path(path)
    if not path.resolve().is_relative_to(PRIVATE.resolve()) or path.resolve()==PRIVATE.resolve():
        raise ValueError('ignored Phase VIII child output required')
    confined_file(ROOT,str(path.resolve().relative_to(ROOT)))
    path.mkdir(parents=True,exist_ok=False)
    return path


def freeze_proof(stage):
    if stage not in {'diagnostic','confirmatory'}:raise ValueError('unknown protocol stage')
    p=ROOT/f'experiments/mist_cove/phase8_{stage}_freeze_v1.json'
    # A present-day JSON cannot replace a committed preregistration.
    committed=subprocess.check_output(['git','show',f'HEAD:{p.relative_to(ROOT)}'],cwd=ROOT)
    if committed!=p.read_bytes():raise ValueError('freeze must be committed before execution')
    r=json.loads(committed)
    for relative,sha in r['files'].items():verified_bytes(ROOT,relative,sha)
    return {'sha256':file_sha(p),'inventory_sha256':digest(r['files'])}


def trusted_reference(task,seed,method,*,scope='SYNTHETIC_ONLY'):
    require_scope(scope)
    if task not in {'single','multiple'} or seed not in {123,456,789} or method not in {'P0','R1','R2'}:
        raise ValueError('only sealed prospective P0/R1/R2 identities allowed')
    diagnostics=json.loads(anchored('reports/mist_cove/phase7_v1/prospective_diagnostics_v2.json'))
    if method=='P0':
        row=next(r for r in diagnostics['rows'] if (r['task'],r['seed'])==(task,seed))
        creation=row['training']['creation']
        folder='diagnostic_v1' if (task,seed)==('single',123) else 'diagnostic_corrected_v2'
    else:
        rows=[json.loads(v) for v in anchored('reports/mist_cove/phase7_v1/robustness_runs_v1.jsonl').splitlines()]
        row=next(r for r in rows if (r['task'],r['seed'],r['method'])==(task,seed,method))
        creation=row['training']['creation'];folder='robustness_v1'
    identity=creation['identity']
    if identity['task']!=task or identity['seed']!=seed or identity['environment']!=environment('cpu'):
        raise ValueError('prospective architecture/environment identity mismatch')
    config_name='configs/mist_cove_phase7_prospective_v1.json' if method=='P0' else 'configs/mist_cove_phase7_robustness_v1.json'
    config=json.loads(anchored(config_name))
    if file_sha(ROOT/config_name)!=identity['config_sha256']:raise ValueError('generator/optimizer config mismatch')
    implementation=identity['implementation']
    manifest='experiments/mist_cove/phase7_prospective_freeze_v1.json' if method=='P0' and (task,seed)==('single',123) else (
        'experiments/mist_cove/phase7_prospective_freeze_v2.json' if method=='P0' else 'experiments/mist_cove/phase7_robustness_freeze_v1.json')
    raw=anchored(manifest);m=json.loads(raw)
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=implementation.get('manifest_sha256',implementation.get('sha256')):
        raise ValueError('source freeze mismatch')
    if digest(m['files'])!=implementation['inventory_sha256']:raise ValueError('implementation inventory mismatch')
    archive=('48c36f6' if method=='P0' and (task,seed)==('single',123) else
             'a3d0502453a4713c0d1a3a12f298a6f76e3a52b2' if method=='P0' else
             'e3fb36e7f8cf955cea30a9f3b4e85f266ca7d5d6')
    archived_differences=[]
    for relative,sha in m['files'].items():
        historical=subprocess.check_output(['git','show',f'{archive}:{relative}'],cwd=ROOT)
        if hashlib.sha256(historical).hexdigest()!=sha:raise ValueError('archived source identity mismatch')
        if file_sha(ROOT/relative)!=sha:
            # The first sealed state predates the documented localization-only
            # correction. Validate its actual source, then compare executable
            # training/restore functions structurally; never reseal that state.
            if method!='P0' or (task,seed)!=('single',123) or relative not in {
                'src/mist_cove/phase7/prospective.py','tests/mist_cove/test_phase7_prospective.py'}:
                raise ValueError('unexpected historical source divergence')
            archived_differences.append(relative)
            if relative.endswith('/prospective.py'):
                import ast
                live=ast.parse((ROOT/relative).read_text());old=ast.parse(historical)
                for name in ('optimizer','steps','seal','restore','train_checkpoint'):
                    left=next(n for n in live.body if isinstance(n,ast.FunctionDef) and n.name==name)
                    right=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
                    if ast.dump(left)!=ast.dump(right):raise ValueError('historical training/restore implementation changed')
    path=confined_file(ROOT,f'mist_transfer_runs/mist_cove/phase7_prospective_v1/{folder}/{task}_{seed}_{method}.pt')
    if json.loads(path.with_suffix('.creation.json').read_text())!=creation:raise ValueError('creation receipt mismatch')
    model=ClippedModel(torch.zeros(18),torch.ones(18)) if method=='R2' else make_model('M1')
    opt=optimizer(model,config);g=torch.Generator();started=time.perf_counter()
    step=restore(path,creation,identity,model,opt,g)
    if step!=300 or len(opt.state)!=len(list(model.parameters())):raise ValueError('optimizer provenance incomplete')
    # Restore verifies payload hashes before loading; assert state-level fields too.
    model.eval()
    receipt={'task':task,'seed':seed,'method':method,'checkpoint_sha256':creation['checkpoint_sha256'],
        'creation_sha256':file_sha(path.with_suffix('.creation.json')),'model_sha256':tensor_digest(model.state_dict()),
        'encoder_sha256':tensor_digest(model.encoder.state_dict()),'source_inventory_sha256':implementation['inventory_sha256'],
        'restored_updates':step,'optimizer_entries':len(opt.state),'python_numpy_torch_batch_rng_restored':True,
        'environment_verified':True,'checkpoint_read_restore_seconds':time.perf_counter()-started,
        'historical_phase6_deserializations':0,'archived_source_commit':archive,
        'documented_reporting_only_source_differences':archived_differences}
    return model,receipt

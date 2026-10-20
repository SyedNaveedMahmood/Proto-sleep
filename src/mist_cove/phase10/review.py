"""Metadata/publication verification without a source-data release path."""
import hashlib
import json
from pathlib import PurePosixPath
import subprocess
from mist_evidence.model import ModelConfig
from .dependency_graph import ROOT,BASE,support_map
from .safety import PUBLIC,no_sensitive_exports
from .verification import verify_frozen_mock_core,require_gates


def public_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts:raise ValueError('metadata path traversal')
    allowed=(name=='next_steps.md' or name=='configs/mist_cove_phase10_v1.json'
             or (name.startswith('docs/mist_cove/PHASE_X_') and name.endswith('.md'))
             or (name.startswith('experiments/mist_cove/phase10_') and name.endswith('.json'))
             or (name.startswith('reports/mist_cove/phase10_v1/') and name.endswith('.json'))
             or (name.startswith('src/mist_cove/phase10/') and name.endswith('.py'))
             or (name.startswith('tests/mist_cove/test_phase10_') and name.endswith('.py'))
             or name in ('scripts/check_mist_cove_phase10.py','scripts/benchmark_mist_cove_phase10_mock.py'))
    if not allowed:raise PermissionError('not a Phase X safe code/aggregate artifact')
    path=ROOT
    for part in p.parts:
        path=path/part
        if path.is_symlink():raise ValueError('symlink metadata forbidden')
    return path


def verify_publication():
    manifest=json.loads(public_name('reports/mist_cove/phase10_v1/publication_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256(public_name(name).read_bytes()).hexdigest()!=expected:raise ValueError('publication tamper: '+name)
    for name in manifest['files']:
        if name.startswith('reports/mist_cove/phase10_v1/') and name.endswith('.json'):
            no_sensitive_exports(json.loads(public_name(name).read_text()))
    return {'status':'PASS','verified_files':len(manifest['files']),'private_source_exports':0}


def metadata_verify():
    freeze=verify_frozen_mock_core()
    c=json.loads(public_name('configs/mist_cove_phase10_v1.json').read_text())
    cfg=ModelConfig(**{k:(tuple(c[k]) if k=='scales' else c[k]) for k in ('fs','samples','scales','stride','embedding_dim','prototypes_per_scale','raw_control','use_summary','use_amplitude','radius','use_crf')})
    g=support_map(cfg)
    recorded=json.loads(public_name('reports/mist_cove/phase10_v1/dependency_summary.json').read_text())
    if recorded['identity']!=g['identity']:raise ValueError('dependency map/config identity mismatch')
    for name,sha in recorded['source_sha256'].items():
        # Only the fixed source paths in this code-produced inventory are permitted.
        if name not in ('src/mist_evidence/model.py','src/mist_evidence/cli.py','src/mist_transfer/baselines.py','src/mist_transfer/source_data.py','src/mist_cove/phase2/adapter.py','configs/p3b_sc_matched_v1.json'):
            raise PermissionError('unknown historical source path')
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('historical source dependency changed')
    correctness=json.loads(public_name('reports/mist_cove/phase10_v1/correctness_receipt.json').read_text());require_gates(correctness,['identity','noninterference'])
    changes=subprocess.check_output(['git','diff','--name-status',BASE,'--'],cwd=ROOT,text=True).splitlines()
    if any(not line.startswith('A\t') for line in changes):raise ValueError('historical file changed')
    plan=subprocess.check_output(['git','hash-object','next_steps.md'],cwd=ROOT,text=True).strip()
    if plan!='e40e646cb027de163d43220e47e51a491316bce2':raise ValueError('Phase IX plan changed')
    publication=verify_publication() if (PUBLIC/'publication_manifest.json').exists() else {'status':'NOT_YET_PUBLISHED'}
    return {'status':'MOCK_INTERFACE_VERIFIED_SOURCE_NOT_AUTHORIZED','frozen_core':freeze,'publication':publication,
            'native_features':g['feature_count'],'windows':sum(len(s['windows']) for s in g['scales']),
            'source_artifact_authentication':'BLOCKED','real_data_reader_available':False,'real_artifacts_opened':0}

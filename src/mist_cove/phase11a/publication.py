"""Owned aggregate IO and reproducibility checks; no historical private roots."""
import json
import os
from pathlib import PurePosixPath
import tempfile
from .provenance import ROOT,BASE,sha256,strict_json,git,InvalidEvidence

PUBLIC=ROOT/'reports/mist_cove/phase11a_v1'


def safe_name(name):
    if not isinstance(name,str) or PurePosixPath(name).is_absolute() or any(v in ('','.','..') for v in name.split('/')) or '\\' in name:raise InvalidEvidence('unsafe publication name')
    valid=(name=='next_steps.md' or name=='configs/mist_cove_phase11a_v1.json'
           or name.startswith('src/mist_cove/phase11a/') and name.endswith('.py')
           or name.startswith('tests/mist_cove/test_phase11a_') and name.endswith('.py')
           or name.startswith('docs/mist_cove/PHASE_XIA_') and name.endswith('.md')
           or name.startswith('reports/mist_cove/phase11a_v1/') and name.endswith('.json')
           or name=='scripts/check_mist_cove_phase11a.py')
    if not valid:raise PermissionError('not an owned Phase XI-A source/aggregate artifact')
    p=ROOT
    for part in PurePosixPath(name).parts:
        p=p/part
        if p.is_symlink():raise InvalidEvidence('symlink publication file/parent')
    return p


def privacy(value):
    if isinstance(value,dict):
        if set(value)&{'checkpoint_bytes','state_dict','normalization_values','waveform_values','private_path','source_recording_path'}:raise InvalidEvidence('sensitive publication content')
        for item in value.values():privacy(item)
    elif isinstance(value,(list,tuple)):
        for item in value:privacy(item)
    elif isinstance(value,str):
        if value.startswith(('/', '~','file:')) or 'mist_transfer_runs/' in value:raise InvalidEvidence('private path publication')


def read_report(name):
    if name not in {'evidence_inventory.json','provenance_graph.json','gate_status.json','correctness_receipt.json','publication_manifest.json','progress.json'}:
        raise PermissionError('fixed report allowlist only')
    return strict_json(safe_name('reports/mist_cove/phase11a_v1/'+name).read_bytes())


def write_report(name,value,*,replace=False):
    if name not in {'gate_status.json','correctness_receipt.json','publication_manifest.json','progress.json'}:raise PermissionError('owned output allowlist only')
    privacy(value);path=safe_name('reports/mist_cove/phase11a_v1/'+name);data=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.xia-')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        if replace:os.replace(tmp,path)
        else:os.link(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def verify_publication():
    manifest=read_report('publication_manifest.json')
    for name,expected in manifest['files'].items():
        path=safe_name(name)
        if sha256(path.read_bytes())!=expected:raise InvalidEvidence('publication content mismatch')
        if name.startswith('reports/mist_cove/phase11a_v1/'):privacy(strict_json(path.read_bytes()))
    return {'status':'PASS','verified_files':len(manifest['files']),'sensitive_artifact_exports':0}


def metadata_check():
    from .provenance import verify_inventory
    from .evidence_graph import validate_graph
    from .gates import historical_gate
    inventory=read_report('evidence_inventory.json');graph=read_report('provenance_graph.json')
    proof=verify_inventory(inventory);shape=validate_graph(graph);gates=historical_gate(inventory,graph)
    git('merge-base','--is-ancestor',BASE,'HEAD')
    changes=git('diff','--name-status',BASE,'--').decode().splitlines()
    if any(not row.startswith('A\t') for row in changes):raise InvalidEvidence('historical tracked file changed')
    if git('hash-object','next_steps.md').decode().strip()!='e40e646cb027de163d43220e47e51a491316bce2':raise InvalidEvidence('historical Phase IX plan changed')
    publication=verify_publication() if (PUBLIC/'publication_manifest.json').exists() else {'status':'NOT YET PUBLISHED'}
    return {'status':'METADATA_VERIFIED_AUTHENTICATION_BLOCKED','metadata':proof,'graph':shape,
            'historical_authentication_status':gates['overall_status'],'publication':publication,'real_artifact_reader_available':False,
            'real_checkpoint_bytes_hashed':0,'real_checkpoints_deserialized':0,'real_EEG_accessed':False,'GPU_queries':0}

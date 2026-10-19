"""Frozen synthetic identities; forbid role/path expansion before filesystem IO."""
import json
from pathlib import Path
from mist_cove.phase2.provenance import file_sha,digest
from mist_cove.phase5.runtime import ROOT,historical_proofs,verify_files


def historical():
    original=historical_proofs()
    for name in ['experiments/mist_cove/phase5_preregistration_v1.json','experiments/mist_cove/phase5_execution_manifest_v1.json','reports/mist_cove/phase5_v1/publication_manifest_v1.json']:
        r=json.loads((ROOT/name).read_text());verify_files(r.get('files') or r['sha256'])
    return {'phase4':original,'phase5_publication_sha256':file_sha(ROOT/'reports/mist_cove/phase5_v1/publication_manifest_v1.json')}


def frozen():
    result={}
    for name in ['preregistration','execution_manifest']:
        p=ROOT/f'experiments/mist_cove/phase6_{name}_v1.json'
        r=json.loads(p.read_text());verify_files(r['files'])
        result[name]={'sha256':file_sha(p),'files':r['files'],'inventory_sha256':digest(r['files'])}
    return result


def private_output(path,scope):
    if scope!='SYNTHETIC_ONLY':raise ValueError('synthetic-only role rejected before filesystem access')
    path=Path(path);allowed=(ROOT/'mist_transfer_runs/mist_cove/phase6_v1').resolve()
    if allowed not in path.resolve().parents:raise ValueError('ignored Phase VI output required')
    if path.exists():raise FileExistsError('previous output preserved')
    path.mkdir(parents=True);return path

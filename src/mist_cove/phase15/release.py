"""Deterministic public release manifest; private caches are never enumerated."""
import json

from .governance import ROOT,PUBLIC,LABEL,read,sha,verify_plan_commit


def paths():
    candidates=[]
    candidates.extend((ROOT/'src/mist_cove/phase15').glob('*.py'))
    candidates.extend((ROOT/'reports/mist_cove/phase15_v1').rglob('*'))
    candidates.extend((ROOT/'docs/mist_cove').glob('PHASE_XV*.md'))
    candidates.extend((ROOT/'docs/mist_cove').glob('PHASE_XVI*.md'))
    candidates.append(ROOT/'configs/mist_cove_phase15_v1.json')
    candidates.extend((ROOT/'tests').glob('test_phase15*.py'))
    exclude={'progress.json','publication_manifest.json'}
    result=sorted(p for p in set(candidates) if p.is_file() and p.name not in exclude)
    if any(p.is_symlink() for p in result):raise ValueError('symlink in release')
    return result


def create():
    verify_plan_commit()
    p=PUBLIC/'publication_manifest.json'
    files={str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in paths()}
    value={'status':LABEL,'files_sha256':files,'file_count':len(files),
           'exclusions':['progress.json: final PR/CI status may update without changing scientific assets',
                         'publication_manifest.json: self-reference'],
           'no_private_cache_or_waveform_file':True,'private_digest_references_only':True,
           'registered_phase14_sigma_gate':'FAIL_UNCHANGED',
           'new_ST_independent_confirmation':False}
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return value


def verify():
    value=read(PUBLIC/'publication_manifest.json')
    expected=value['files_sha256']
    if set(expected)!={str(f.relative_to(ROOT)) for f in paths()}:
        raise ValueError('public release file set differs')
    for name,digest in expected.items():
        if sha((ROOT/name).read_bytes())!=digest:raise ValueError('public release hash differs: '+name)
    return len(expected)


if __name__=='__main__':
    print(json.dumps({'files':create()['file_count']}))

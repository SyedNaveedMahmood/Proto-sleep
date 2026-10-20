"""Typed references, strict manifests and content-verified approved Git objects."""
import hashlib
import hmac
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE='0ce6d4235661c6bda57392040954abb506abb7b8'
P3B_IMPL='661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23'
P3B_PUBLICATION='15995b31821ee8db717f3d6edfa41166de65d4a3'
PHASE3_IMPL='a601e8e004b9e0c2942e83d4517fb8a47856de72'
PHASE3_PUBLICATION='4095c805cfd48fbb1e9ef938dbc46da28ed1dfb1'
COMMITS={BASE,P3B_IMPL,P3B_PUBLICATION,PHASE3_IMPL,PHASE3_PUBLICATION}
HEX256=re.compile(r'[0-9a-f]{64}\Z');HEX160=re.compile(r'[0-9a-f]{40}\Z')
MAX_MANIFEST=2*1024*1024
IDENTITY_KEYS={'experiment','variant','seed','implementation_commit','split_fingerprint',
               'normalization_fingerprint','bank_sha256','config_sha256','environment_sha256','run_fingerprint'}


class InvalidEvidence(ValueError):pass


def sha256(data):return hashlib.sha256(data).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest_matches(actual,expected):
    if not isinstance(actual,str) or not isinstance(expected,str) or not HEX256.fullmatch(actual) or not HEX256.fullmatch(expected):
        raise InvalidEvidence('malformed SHA256')
    return hmac.compare_digest(actual,expected)


def typed_digest_equal(reference,kind,experiment,actual):
    """A typed equality check, not a declaration of historical authenticity."""
    kinds={'MODEL_STATE_DIGEST','CHECKPOINT_BYTES_SHA256','COMPLETION_MARKER_BYTES_SHA256','SPLIT_FILE_SHA256','SPLIT_FINGERPRINT'}
    if not isinstance(reference,dict) or set(reference)!={'kind','value','experiment'} or kind not in kinds:
        raise InvalidEvidence('typed digest reference required')
    if reference['kind']!=kind or reference['experiment']!=experiment:raise InvalidEvidence('wrong digest kind or experiment lineage')
    return digest_matches(actual,reference['value'])


def strict_json(data,*,limit=MAX_MANIFEST):
    if not isinstance(data,bytes) or len(data)>limit:raise InvalidEvidence('bounded byte JSON required')
    def pairs(items):
        value={}
        for key,v in items:
            if key in value:raise InvalidEvidence('duplicate JSON key')
            value[key]=v
        return value
    def nonfinite(value):raise InvalidEvidence('nonfinite JSON')
    try:return json.loads(data,object_pairs_hook=pairs,parse_constant=nonfinite)
    except (UnicodeError,ValueError,RecursionError) as error:raise InvalidEvidence('malformed or duplicate JSON') from error


def validate_identity(value,*,synthetic=False):
    if not isinstance(value,dict) or set(value)!=IDENTITY_KEYS:raise InvalidEvidence('exact experiment identity required')
    experiments={'SYNTHETIC_P3B','SYNTHETIC_PHASE_III'} if synthetic else {'P3B','MIST_COVE_PHASE_III'}
    if value['experiment'] not in experiments or value['variant'] not in {'raw_context','evidence'} or type(value['seed']) is not int or value['seed'] not in (123,456,789):
        raise InvalidEvidence('ambiguous experiment/variant/seed')
    if not isinstance(value['implementation_commit'],str) or not HEX160.fullmatch(value['implementation_commit']):raise InvalidEvidence('implementation commit identity required')
    for key in IDENTITY_KEYS-{'experiment','variant','seed','implementation_commit'}:
        if not isinstance(value[key],str) or not HEX256.fullmatch(value[key]):raise InvalidEvidence('missing typed lineage hash: '+key)
    return value


def logical_name(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '\x00' in name:raise InvalidEvidence('unsafe logical artifact name')
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ('','.','..') for x in name.split('/')):raise InvalidEvidence('absolute/traversal artifact path')
    return name


def output_map(value):
    if not isinstance(value,dict) or not value:raise InvalidEvidence('complete output map required')
    for name,digest in value.items():
        logical_name(name)
        if not isinstance(digest,str) or not HEX256.fullmatch(digest):raise InvalidEvidence('output SHA256 required')
    return value


def parse_publication_manifest(data,schema):
    """Pure bytes only. Caller must authenticate a historical JSON before parsing.

    There is no file/root-directory argument and no original-artifact reader.
    """
    value=strict_json(data)
    if not isinstance(value,dict):raise InvalidEvidence('manifest object required')
    if schema=='p3b-root-completion-v1':
        if set(value)!={'run_fingerprint','status','outputs'} or value['status']!='ALL_12_VALIDATED_COMPLETE':raise InvalidEvidence('P3B root schema')
        digest_matches(value['run_fingerprint'],value['run_fingerprint']);output_map(value['outputs'])
    elif schema=='p3b-validated-completion-v1':
        if set(value)!={'run_fingerprint','seed','variant','status','outputs'} or value['status']!='VALIDATED_COMPLETE' or value['variant']!='raw_context' or type(value['seed']) is not int or value['seed']!=123:raise InvalidEvidence('selected P3B marker schema/identity')
        digest_matches(value['run_fingerprint'],value['run_fingerprint']);output_map(value['outputs'])
    elif schema=='phase3-complete-seal-v1':
        if set(value)!={'identity','status','outputs'} or value['status']!='COMPLETE':raise InvalidEvidence('Phase III seal schema')
        digest_matches(value['identity'],value['identity']);output_map(value['outputs'])
        if set(value['outputs'])!={'last.pt','best.pt','history.csv','provenance.json','COMPLETE.json','validation_predictions.csv','EPOCH_SEAL.json'}:raise InvalidEvidence('Phase III seal output set')
    else:raise InvalidEvidence('unapproved manifest schema')
    return value


def approved_git_path(name):
    logical_name(name)
    p=PurePosixPath(name)
    source=(name.endswith('.py') and (str(p.parent) in {'src/mist_transfer','src/mist_evidence','scripts','src/mist_cove/phase2','src/mist_cove/phase3','analysis/mist_cove_phase3_v1'} or name in {'src/protosleep/attnsleep.py','src/protosleep/config.py','src/protosleep/utils.py'}))
    public=name in {'reports/p3b/results.json','reports/p3b/FINAL_COPY_PASTE.txt','reports/p3b/validation_summary.json','reports/p3b/gate_a_summary.json','reports/mist_cove/phase3_source_v1/source_results.json','reports/mist_cove/phase3_source_v1/execution_receipt.json','configs/p3b_sc_matched_v1.json','configs/mist_cove_phase3_source_v1.json'}
    if not source and not public:raise PermissionError('not an approved historical source/publication path')
    return name


def git(*args):
    env={k:v for k,v in os.environ.items() if k not in {'GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES'}}
    env['GIT_NO_REPLACE_OBJECTS']='1'
    return subprocess.check_output(['git','--no-replace-objects','-C',str(ROOT),*args],env=env,stderr=subprocess.PIPE,timeout=10)


def verified_object(kind,oid):
    if kind not in {'blob','commit'} or not isinstance(oid,str) or not HEX160.fullmatch(oid):raise InvalidEvidence('Git object identity required')
    raw=git('cat-file',kind,oid)
    if hashlib.sha1(kind.encode()+b' '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=oid:raise InvalidEvidence('Git content identity mismatch')
    return raw


def historical_blob(commit,name,*,expected_blob=None):
    if commit not in COMMITS:raise PermissionError('unapproved historical commit')
    approved_git_path(name)  # before Git/filesystem access
    verified_object('commit',commit)
    row=git('ls-tree','-z',commit,'--',name).split(b'\0')[0]
    if not row:raise InvalidEvidence('historical record missing')
    header,path=row.split(b'\t');mode,kind,oid=header.decode().split()
    if mode!='100644' or kind!='blob' or path.decode()!=name:raise InvalidEvidence('nonregular/symlink Git record')
    if expected_blob is not None and oid!=expected_blob:raise InvalidEvidence('historical blob differs from pinned reference')
    return verified_object('blob',oid),{'commit':commit,'path':name,'blob_oid':oid}


def verify_inventory(inventory):
    if inventory.get('base_commit')!=BASE or inventory.get('historical_private_artifacts_opened')!=0 or inventory.get('real_checkpoint_bytes_hashed')!=0:
        raise InvalidEvidence('metadata-only inventory scope/identity mismatch')
    raw={};seen=set()
    for row in inventory['records']:
        key=(row['commit'],row['path'])
        if key in seen:raise InvalidEvidence('duplicate provenance record')
        seen.add(key)
        data,_=historical_blob(*key,expected_blob=row['blob_oid'])
        if not digest_matches(sha256(data),row['blob_sha256']):raise InvalidEvidence('published record content changed')
        raw[key]=data
    summary=strict_json(raw[(P3B_PUBLICATION,'reports/p3b/validation_summary.json')])
    result=strict_json(raw[(P3B_PUBLICATION,'reports/p3b/results.json')])
    selected=[r for r in result['per_seed'] if r['variant']=='raw_context' and r['seed']==123]
    if len(selected)!=1:raise InvalidEvidence('ambiguous P3B candidate')
    c=inventory['candidate']
    if (c['experiment'],c['variant'],c['seed'],c['implementation_commit'])!=('P3B','raw_context',123,P3B_IMPL) or c['run_fingerprint']!=result['run_fingerprint'] or c['model_state_digest']!=selected[0]['best_state_digest']:
        raise InvalidEvidence('selected run/variant/seed/state mismatch')
    if c['original_serialized_checkpoint_sha256'] is not None:
        raise InvalidEvidence('unestablished original checkpoint byte reference')
    config=strict_json(raw[(P3B_IMPL,'configs/p3b_sc_matched_v1.json')])
    phase3=strict_json(raw[(PHASE3_PUBLICATION,'reports/mist_cove/phase3_source_v1/source_results.json')])
    receipt=strict_json(raw[(PHASE3_PUBLICATION,'reports/mist_cove/phase3_source_v1/execution_receipt.json')])
    expected_refs={
        'MODEL_STATE_DIGEST':selected[0]['best_state_digest'],
        'COMPLETION_MARKER_BYTES_SHA256':summary['root_completion_sha256'],
        'SPLIT_FINGERPRINT':summary['split_fingerprint'],
        'SPLIT_FILE_SHA256':phase3['split_file_sha256'],
    }
    refs=inventory['typed_references']
    if len(refs)!=len(expected_refs) or {r['kind'] for r in refs}!=set(expected_refs):
        raise InvalidEvidence('duplicate or missing typed historical reference')
    for ref in refs:
        if not digest_matches(ref['value'],expected_refs[ref['kind']]):
            raise InvalidEvidence('typed historical reference mismatch')
    if not (summary['run_fingerprint']==result['run_fingerprint'] and
            summary['implementation_commit']==result['implementation_commit']==P3B_IMPL and
            summary['split_fingerprint']==result['split_fingerprint']==config['split_fingerprint']==phase3['split_fingerprint']==receipt['split_fingerprint'] and
            phase3['split_file_sha256']==receipt['split_file_sha256']):
        raise InvalidEvidence('published source/run/split references disagree')
    seals=inventory['phase3_seal_references']
    if len(seals)!=3 or {r['seed'] for r in seals}!={123,456,789}:
        raise InvalidEvidence('ambiguous Phase III seal reference set')
    for ref in seals:
        rows=[r for r in phase3['rows'] if r['variant']=='raw_context' and r['seed']==ref['seed']]
        if (ref['experiment']!='MIST_COVE_PHASE_III' or ref['variant']!='raw_context' or
                ref['source_commit']!=PHASE3_PUBLICATION or ref['checkpoint_byte_sha256'] is not None or
                len(rows)!=1 or not digest_matches(ref['complete_seal_sha256'],rows[0]['complete_seal_sha256'])):
            raise InvalidEvidence('Phase III seal identity/reference mismatch')
    for name,reference_field in [('implementation','implementation_hash'),('reused','reused_implementation_hash')]:
        manifest=inventory['source_manifests'][name]
        for path,expected in manifest['files'].items():
            if sha256(raw[(P3B_IMPL,path)])!=expected:raise InvalidEvidence('source manifest file mismatch')
        if sha256(canonical(manifest['files']))!=summary[reference_field] or manifest['canonical_SHA256']!=summary[reference_field]:raise InvalidEvidence('original implementation fingerprint mismatch')
    root=[r for r in inventory['typed_references'] if r['kind']=='COMPLETION_MARKER_BYTES_SHA256']
    if len(root)!=1 or root[0]['experiment']!='P3B' or root[0]['value']!=summary['root_completion_sha256']:
        raise InvalidEvidence('root marker reference mismatch')
    for commit in (P3B_IMPL,P3B_PUBLICATION,PHASE3_IMPL,PHASE3_PUBLICATION):git('merge-base','--is-ancestor',commit,BASE)
    return {'status':'PASS','verified_historical_records':len(seen),
            'original_source_files':len(inventory['source_manifests']['implementation']['files']),
            'reused_source_files':len(inventory['source_manifests']['reused']['files']),
            'root_completion_reference_sha256':root[0]['value'],'trust_level':'LEVEL_2_HISTORICAL_REFERENCES_ONLY',
            'original_checkpoint_byte_value_materialized':False,'actual_artifact_authenticated':False}


def real_artifact_access(*args,**kwargs):
    raise PermissionError('Phase XI-A has no real artifact reader or deserializer; separate reviewed entrypoint required')

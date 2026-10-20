"""Fixture-only authentication and immutable-byte safe parsing. No real loader."""
from dataclasses import dataclass
import math
import os
from .provenance import canonical,sha256,strict_json,validate_identity,digest_matches,InvalidEvidence
from .fixtures import FixtureCase,IndependentFixtureReference,_CAP,MAGIC,DEFAULT_STATE,default_identity

MAX_BYTES=1048576


@dataclass(frozen=True)
class Decision:
    status:str
    code:str
    stage:str
    def aggregate(self):return {'status':self.status,'code':self.code,'stage':self.stage,'domain':'GENERATED_FIXTURE_ONLY','unsafe_deserializer_calls':0}


def stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def read_bounded(fd,limit):
    info=os.fstat(fd)
    if info.st_size>limit:raise InvalidEvidence('size ceiling')
    chunks=[];offset=0
    while offset<=limit:
        block=os.pread(fd,min(65536,limit+1-offset),offset)
        if not block:break
        chunks.append(block);offset+=len(block)
    data=b''.join(chunks)
    if len(data)>limit or stamp(info)!=stamp(os.fstat(fd)):raise InvalidEvidence('changed or oversized bytes')
    return data,stamp(info)


def receipt(raw,kind):
    value=strict_json(raw,limit=65536)
    keys={'schema','identity','checkpoint_sha256','checkpoint_size','model_state_digest','format'}
    if kind=='completion':keys|={'creation_sha256'}
    if not isinstance(value,dict) or set(value)!=keys or value['schema']!=f'xia-fixture-{kind}-v1':raise InvalidEvidence('receipt schema')
    validate_identity(value['identity'],synthetic=True)
    for name in ('checkpoint_sha256','model_state_digest'):
        digest_matches(value[name],value[name])
    if kind=='completion':digest_matches(value['creation_sha256'],value['creation_sha256'])
    if type(value['checkpoint_size']) is not int or not 0<value['checkpoint_size']<=MAX_BYTES:raise InvalidEvidence('receipt size')
    if value['format'] not in {'GENERATED_JSON','UNSAFE_PICKLE_FIXTURE'}:raise InvalidEvidence('receipt format')
    return value


class VerifiedFixtureSnapshot:
    def __init__(self,capability,case,fd,data,metadata,creation):
        if capability is not _CAP:raise PermissionError('verified fixture snapshot required')
        self._capability=capability;self._case=case;self._fd=fd;self._bytes=data;self._stamp=metadata;self._creation=creation;self._closed=False
    def close(self):
        if not self._closed:os.close(self._fd);self._closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def verify_fixture(case):
    """No pathname accepted. Return decision and optional verified owned snapshot."""
    if type(case) is not FixtureCase or case._capability is not _CAP:
        return Decision('BLOCKED','UNAUTHORIZED_ARTIFACT_INTERFACE','CAPABILITY'),None
    try:case.valid()
    except PermissionError:return Decision('BLOCKED','EXPIRED_FIXTURE_CAPABILITY','CAPABILITY'),None
    reference=case.reference
    if type(reference) is not IndependentFixtureReference or reference.capability is not _CAP or reference.case_number!=case._number:
        return Decision('BLOCKED','INDEPENDENT_REFERENCE_MISSING_OR_UNTRUSTED','REFERENCE'),None
    if reference.reference_kind!='CHECKPOINT_BYTES_SHA256':return Decision('FAIL','STATE_VERSUS_BYTE_REFERENCE_KIND','REFERENCE'),None
    if reference.identity_bytes!=canonical(default_identity()):return Decision('FAIL','PROSPECTIVE_EXPERIMENT_IDENTITY_MISMATCH','REFERENCE'),None
    if reference.model_state_digest!=sha256(canonical(DEFAULT_STATE)):return Decision('FAIL','PUBLISHED_MODEL_STATE_REFERENCE_MISMATCH','REFERENCE'),None
    parsed={}
    for kind,name,expected in [('creation','creation.json',reference.creation_sha256),('completion','completion.json',reference.completion_sha256)]:
        try:
            fd=case.open_owned(name)
            try:data,_=read_bounded(fd,65536)
            finally:os.close(fd)
        except FileNotFoundError:return Decision('BLOCKED',kind.upper()+'_RECEIPT_MISSING','RECEIPTS'),None
        except (OSError,InvalidEvidence):return Decision('FAIL','RECEIPT_PATH_OR_SIZE_REJECTED','RECEIPTS'),None
        if not digest_matches(sha256(data),expected):return Decision('FAIL',kind.upper()+'_HISTORICAL_REFERENCE_MISMATCH','RECEIPTS'),None
        try:parsed[kind]=receipt(data,kind)
        except InvalidEvidence:return Decision('FAIL','MALFORMED_DUPLICATE_OR_AMBIGUOUS_RECEIPT','RECEIPTS'),None
    first,last=parsed['creation'],parsed['completion']
    common=('identity','checkpoint_sha256','checkpoint_size','model_state_digest','format')
    if any(first[k]!=last[k] for k in common) or last['creation_sha256']!=reference.creation_sha256:
        return Decision('FAIL','CONTRADICTORY_CREATION_COMPLETION_CHAIN','RECEIPTS'),None
    if canonical(first['identity'])!=reference.identity_bytes or first['checkpoint_sha256']!=reference.checkpoint_sha256 or first['model_state_digest']!=reference.model_state_digest:
        return Decision('FAIL','RECEIPTS_DISAGREE_WITH_INDEPENDENT_SELECTION','RECEIPTS'),None
    fd=None
    try:
        fd=case.open_owned('candidate.bin');data,metadata=read_bounded(fd,MAX_BYTES)
        if len(data)<first['checkpoint_size']:decision=Decision('FAIL','INCOMPLETE_CHECKPOINT_WRITE','BYTES')
        elif len(data)!=first['checkpoint_size']:decision=Decision('FAIL','CHECKPOINT_SIZE_MISMATCH','BYTES')
        elif not digest_matches(sha256(data),reference.checkpoint_sha256):decision=Decision('FAIL','CHECKPOINT_BYTE_DIGEST_MISMATCH','BYTES')
        else:
            snapshot=VerifiedFixtureSnapshot(_CAP,case,fd,data,metadata,first);fd=None
            return Decision('PASS','SYNTHETIC_ARTIFACT_BYTES_AUTHENTICATED','BYTES'),snapshot
        return decision,None
    except FileNotFoundError:return Decision('BLOCKED','ANCHORED_REFERENCE_WITHOUT_CANDIDATE','BYTES'),None
    except (OSError,InvalidEvidence):return Decision('FAIL','CANDIDATE_SYMLINK_NONREGULAR_OR_CHANGED','BYTES'),None
    finally:
        if fd is not None:os.close(fd)


def load_generated_json(snapshot):
    """Only known generated JSON, from immutable verified bytes; never pickle."""
    if type(snapshot) is not VerifiedFixtureSnapshot or snapshot._capability is not _CAP or snapshot._closed:
        return Decision('BLOCKED','VERIFIED_FIXTURE_SNAPSHOT_REQUIRED','LOAD')
    try:
        snapshot._case.valid()
        current=os.stat('candidate.bin',dir_fd=snapshot._case._fd,follow_symlinks=False)
        data,metadata=read_bounded(snapshot._fd,MAX_BYTES)
        if stamp(current)!=snapshot._stamp or metadata!=snapshot._stamp or sha256(data)!=sha256(snapshot._bytes):
            return Decision('FAIL','TOCTOU_CANDIDATE_CHANGED','LOAD')
    except (OSError,InvalidEvidence,PermissionError):return Decision('FAIL','TOCTOU_CANDIDATE_UNAVAILABLE','LOAD')
    if snapshot._creation['format']!='GENERATED_JSON' or not snapshot._bytes.startswith(MAGIC):
        return Decision('BLOCKED','UNSAFE_OR_UNSUPPORTED_SERIALIZATION_NOT_EXECUTED','LOAD')
    try:
        value=strict_json(snapshot._bytes[len(MAGIC):],limit=MAX_BYTES)
        if set(value)!={'schema','identity','state','model_state_digest','metrics'} or value['schema']!='xia-generated-envelope-v1':raise InvalidEvidence('payload schema')
        if canonical(value['identity'])!=canonical(default_identity()):raise InvalidEvidence('payload identity')
        state=value['state']
        if not isinstance(state,dict) or not 1<=len(state)<=64:raise InvalidEvidence('state schema')
        for name,numbers in state.items():
            if not isinstance(name,str) or not isinstance(numbers,list) or not 1<=len(numbers)<=64 or any(type(x) not in (int,float) or not math.isfinite(x) for x in numbers):raise InvalidEvidence('numeric generated state only')
        if sha256(canonical(state))!=snapshot._creation['model_state_digest'] or value['model_state_digest']!=snapshot._creation['model_state_digest']:raise InvalidEvidence('model-state digest')
        if value['metrics']!={'score':.5}:raise InvalidEvidence('generated metrics schema')
    except (InvalidEvidence,TypeError,KeyError):return Decision('FAIL','GENERATED_PAYLOAD_SCHEMA_OR_STATE_MISMATCH','LOAD')
    return Decision('PASS','VERIFIED_GENERATED_JSON_ONLY','LOAD')


SCENARIOS=(
    'correct_chain','modified_bytes','rewritten_local_sidecar','state_digest_wrong_serialization',
    'wrong_seed','wrong_variant','same_metrics_different_weights','wrong_normalization','wrong_bank',
    'missing_creation','forged_present_checksum','missing_completion','different_run_marker',
    'wrong_implementation','outside_symlink','changed_before_loading','incomplete_write',
    'duplicate_or_conflicting_receipts','unsafe_pickle','anchor_without_candidate')


def run_scenario(name):
    from .fixtures import FixtureLab
    if name not in SCENARIOS:raise ValueError('unknown generated scenario')
    changes={
        'wrong_seed':{'seed':456},'wrong_variant':{'variant':'evidence'},
        'wrong_normalization':{'normalization_fingerprint':'b'*64},
        'wrong_bank':{'bank_sha256':'b'*64},'wrong_implementation':{'implementation_commit':'b'*40}}
    expected_status='FAIL'
    expected_code={
        'correct_chain':'VERIFIED_GENERATED_JSON_ONLY','modified_bytes':'CHECKPOINT_BYTE_DIGEST_MISMATCH',
        'rewritten_local_sidecar':'CHECKPOINT_BYTE_DIGEST_MISMATCH','state_digest_wrong_serialization':'CHECKPOINT_BYTE_DIGEST_MISMATCH',
        'same_metrics_different_weights':'PUBLISHED_MODEL_STATE_REFERENCE_MISMATCH',
        'missing_creation':'CREATION_RECEIPT_MISSING','forged_present_checksum':'INDEPENDENT_REFERENCE_MISSING_OR_UNTRUSTED',
        'missing_completion':'COMPLETION_RECEIPT_MISSING','different_run_marker':'COMPLETION_HISTORICAL_REFERENCE_MISMATCH',
        'outside_symlink':'CANDIDATE_SYMLINK_NONREGULAR_OR_CHANGED','changed_before_loading':'TOCTOU_CANDIDATE_CHANGED',
        'incomplete_write':'INCOMPLETE_CHECKPOINT_WRITE','duplicate_or_conflicting_receipts':'MALFORMED_DUPLICATE_OR_AMBIGUOUS_RECEIPT',
        'unsafe_pickle':'UNSAFE_OR_UNSUPPORTED_SERIALIZATION_NOT_EXECUTED','anchor_without_candidate':'ANCHORED_REFERENCE_WITHOUT_CANDIDATE'}
    if name in changes:expected_code[name]='PROSPECTIVE_EXPERIMENT_IDENTITY_MISMATCH'
    if name=='correct_chain':expected_status='PASS'
    if name in {'missing_creation','forged_present_checksum','missing_completion','unsafe_pickle','anchor_without_candidate'}:expected_status='BLOCKED'
    with FixtureLab() as lab:
        kwargs={'identity_changes':changes.get(name)}
        if name=='same_metrics_different_weights':kwargs['state']={'synthetic_weight':[7.,8.,9.],'synthetic_buffer':[1.,2.]}
        if name=='duplicate_or_conflicting_receipts':kwargs['malformed']='duplicate'
        if name=='unsafe_pickle':kwargs['unsafe']=True
        case=lab.new_case(**kwargs)
        if name in {'modified_bytes','rewritten_local_sidecar'}:
            original=case.bytes('candidate.bin');case.overwrite('candidate.bin',original[:-1]+b'!')
            if name=='rewritten_local_sidecar':case.overwrite('local_sidecar.json',canonical({'today_sha256':sha256(case.bytes('candidate.bin')),'claimed_historical':True}))
        elif name=='state_digest_wrong_serialization':
            data=case.bytes('candidate.bin');changed=data.replace(b'"score":0.5',b'"score":0.6');assert changed!=data
            case.overwrite('candidate.bin',changed)
        elif name=='missing_creation':case.remove('creation.json')
        elif name=='forged_present_checksum':
            case.reference={'trusted':True,'today_sha256':sha256(case.bytes('candidate.bin')),'claimed_historical':True}
        elif name=='missing_completion':case.remove('completion.json')
        elif name=='different_run_marker':
            other=lab.new_case(identity_changes={'seed':456});case.overwrite('completion.json',other.bytes('completion.json'))
        elif name=='outside_symlink':case.outside_symlink()
        elif name=='incomplete_write':case.overwrite('candidate.bin',case.bytes('candidate.bin')[:17])
        elif name=='anchor_without_candidate':case.remove('candidate.bin')
        byte_decision,snapshot=verify_fixture(case);decision=byte_decision
        try:
            if snapshot is not None:
                if name=='changed_before_loading':case.overwrite('candidate.bin',case.bytes('candidate.bin')[:-1]+b'!')
                decision=load_generated_json(snapshot)
        finally:
            if snapshot is not None:snapshot.close()
        extra=None
        if name=='duplicate_or_conflicting_receipts':
            conflict=lab.new_case(malformed='contradiction');bad,_=verify_fixture(conflict)
            extra=bad.aggregate()
            if bad.code!='CONTRADICTORY_CREATION_COMPLETION_CHAIN':raise AssertionError('conflicting record not rejected')
        if lab.unsafe_executed():raise AssertionError('unsafe serialized object executed')
        correct=(decision.status==expected_status and decision.code==expected_code[name])
        if not correct:raise AssertionError('unexpected generated scenario decision: '+name+' '+decision.code)
        return {'case_id':SCENARIOS.index(name)+1,'case':name,**decision.aggregate(),
                'expected_decision_observed':correct,'byte_verification_status':byte_decision.status,
                'local_sidecar_used_as_trust':False,'unsafe_execution_observed':False,
                'additional_conflict_decision':extra}


def run_all_scenarios():
    import time
    import resource
    start=time.perf_counter();rows=[]
    for name in SCENARIOS:
        if time.perf_counter()-start>10:raise TimeoutError('generated fixture verification budget exceeded')
        rows.append(run_scenario(name))
    return {'status':'PASS','domain':'GENERATED_FIXTURE_ONLY','required_cases':20,'rows':rows,
            'wall_seconds':time.perf_counter()-start,'peak_process_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'real_artifact_access':False,'GPU_queries':0,'neural_fits':0,'optimizer_updates':0,'unsafe_deserializer_calls':0}

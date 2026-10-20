import dataclasses
import pickle
import pytest
from mist_cove.phase11a.fixtures import FixtureLab
from mist_cove.phase11a.verification import verify_fixture,load_generated_json,run_scenario,SCENARIOS


def test_original_linked_generated_chain_and_safe_snapshot():
    with FixtureLab() as lab:
        case=lab.new_case();decision,snapshot=verify_fixture(case)
        assert decision.status=='PASS'
        with snapshot:assert load_generated_json(snapshot).status=='PASS'
        assert load_generated_json(snapshot).status=='BLOCKED'


def test_tampering_rejected_before_payload_parse(monkeypatch):
    import mist_cove.phase11a.verification as module
    with FixtureLab() as lab:
        case=lab.new_case();case.overwrite('candidate.bin',case.bytes('candidate.bin')[:-1]+b'!')
        monkeypatch.setattr(module,'load_generated_json',lambda *a:pytest.fail('must not load bad bytes'))
        decision,snapshot=verify_fixture(case)
        assert decision.code=='CHECKPOINT_BYTE_DIGEST_MISMATCH' and snapshot is None


@pytest.mark.parametrize('name',SCENARIOS)
def test_twenty_required_adversarial_fixture_decisions(name):
    row=run_scenario(name)
    assert row['expected_decision_observed'] and not row['unsafe_execution_observed']
    assert not row['local_sidecar_used_as_trust']


def test_state_reference_cannot_be_used_as_byte_reference():
    with FixtureLab() as lab:
        case=lab.new_case();case.reference=dataclasses.replace(case.reference,reference_kind='MODEL_STATE_DIGEST')
        result,snapshot=verify_fixture(case)
        assert result.code=='STATE_VERSUS_BYTE_REFERENCE_KIND' and snapshot is None


def test_unsafe_pickle_never_reaches_unrestricted_deserializer(monkeypatch):
    monkeypatch.setattr(pickle,'loads',lambda *a,**kw:pytest.fail('pickle must never execute'))
    with FixtureLab() as lab:
        case=lab.new_case(unsafe=True);result,snapshot=verify_fixture(case)
        assert result.status=='PASS'  # byte authenticity does not mean safe parsing
        with snapshot:assert load_generated_json(snapshot).code=='UNSAFE_OR_UNSUPPORTED_SERIALIZATION_NOT_EXECUTED'
        assert not lab.unsafe_executed()


def test_no_caller_path_or_expired_capability_can_open_source():
    class NeverInspect:
        def __fspath__(self):pytest.fail('path must never be interpreted')
        def __str__(self):pytest.fail('path must never be interpreted')
    for value in ['/unopened/source/best.pt','../bank',NeverInspect(),{'authorize':True}]:
        result,snapshot=verify_fixture(value)
        assert result.code=='UNAUTHORIZED_ARTIFACT_INTERFACE' and snapshot is None
    with FixtureLab() as lab:case=lab.new_case()
    assert verify_fixture(case)[0].code=='EXPIRED_FIXTURE_CAPABILITY'

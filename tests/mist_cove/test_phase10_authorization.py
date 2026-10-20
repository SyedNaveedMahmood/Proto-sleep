import importlib.util
import json
from pathlib import Path
import pytest
import torch
from mist_cove.phase10.safety import source_access,mock_request,no_sensitive_exports
from mist_cove.phase10.verification import require_gates


@pytest.mark.parametrize('role',['train','val','reserved','test','ST','SHHS','synthetic_mock','artifact_only'])
def test_all_source_roles_rejected_before_any_path_operation(role):
    class NeverInspect:
        def __str__(self):pytest.fail('path must not be inspected')
        def __fspath__(self):pytest.fail('path must not be inspected')
    with pytest.raises(PermissionError):source_access(role,NeverInspect())


@pytest.mark.parametrize('role',['train','val','reserved','ST','SHHS'])
def test_mock_role_cannot_be_released_by_path_or_flag(role):
    with pytest.raises(PermissionError):mock_request(role)


def test_absolute_and_symlink_paths_never_resolved(tmp_path,monkeypatch):
    target=tmp_path/'fake_train.edf';target.write_text('synthetic sentinel')
    link=tmp_path/'mock_link';link.symlink_to(target)
    def forbidden(*a,**kw):pytest.fail('unauthorized path resolved or opened')
    monkeypatch.setattr(Path,'resolve',forbidden);monkeypatch.setattr(Path,'stat',forbidden);monkeypatch.setattr(Path,'open',forbidden)
    for path in (str(target),link,Path('/unopened/mock/VAL/record.edf')):
        with pytest.raises(PermissionError):source_access('train',path)
        with pytest.raises(PermissionError):mock_request(input_path=path)
        with pytest.raises(PermissionError):mock_request(checkpoint_path=path)
        with pytest.raises(PermissionError):mock_request(bank_path=path)


def test_mock_factory_and_adapter_never_call_source_readers(monkeypatch):
    import mist_transfer.source_data as source
    import mist_transfer.protocol as protocol
    import numpy as np
    from mist_cove.phase10.mock_models import make_mock,mock_input
    from mist_cove.phase10.source_adapter import SourceAdapter
    def forbidden(*a,**kw):pytest.fail('real data or checkpoint reader invoked')
    for name in ('SourceRecording','training_bank','fit_normalization'):monkeypatch.setattr(source,name,forbidden)
    for name in ('load_development','read_frozen_split'):monkeypatch.setattr(protocol,name,forbidden)
    monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(np,'load',forbidden)
    adapter=SourceAdapter(make_mock().network);cache=adapter.cache(mock_input(n=1))
    assert adapter.logits(cache,torch.ones(1,15,dtype=torch.bool)).shape==(1,1,5)


def test_cli_cannot_enable_real_access(monkeypatch):
    root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('phase10_cli',root/'scripts/check_mist_cove_phase10.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for args in [['--authorize-real-source'],['--role','train'],['--checkpoint','fake.pt'],['real'],['--input','fake.edf']]:
        with pytest.raises(SystemExit) as error:module.main(args)
        assert error.value.code==2


def test_gates_fail_closed_and_public_export_rejects_sensitive_fields():
    for r in ({},{'identity':{'status':'FAIL'}},{'identity':{'status':'PASS'}}):
        with pytest.raises(ValueError):require_gates(r,['identity','noninterference'])
    for r in ({'source_path':'fake'},{'rows':[{'state_dict':{}}]},{'where':'/private/fake'},{'where':'mist_transfer_runs/fake'}):
        with pytest.raises(ValueError):no_sensitive_exports(r)


def test_output_atomicity_existing_nonfinite_and_symlink_rejected(tmp_path,monkeypatch):
    import mist_cove.phase10.safety as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'PUBLIC',tmp_path/'public');monkeypatch.setattr(module,'PRIVATE',tmp_path/'private')
    p=module.PUBLIC/'receipt.json';module.write_json(p,{'aggregate':.2});before=p.read_bytes()
    with pytest.raises(FileExistsError):module.write_json(p,{'aggregate':.4})
    assert p.read_bytes()==before
    with pytest.raises(ValueError):module.write_json(module.PUBLIC/'bad.json',{'value':float('nan')})
    link=module.PUBLIC/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):module.write_json(link/'bad.json',{})
    assert not list(module.PUBLIC.glob('.phase10-*'))


def test_publication_manifest_tamper_and_path_injection_rejected(tmp_path,monkeypatch):
    from mist_cove.phase10 import review
    monkeypatch.setattr(review,'ROOT',tmp_path)
    public=tmp_path/'reports/mist_cove/phase10_v1';public.mkdir(parents=True)
    p=public/'aggregate.json';p.write_text('{}');manifest=public/'publication_manifest.json'
    import hashlib
    manifest.write_text(json.dumps({'files':{'reports/mist_cove/phase10_v1/aggregate.json':hashlib.sha256(p.read_bytes()).hexdigest()}}))
    assert review.verify_publication()['status']=='PASS'
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='tamper'):review.verify_publication()
    for name in ['/unopened/fake.edf','../reserved.edf','mist_transfer_runs/fake.pt']:
        with pytest.raises((ValueError,PermissionError)):review.public_name(name)


def test_occupied_cuda_gate_prevents_device_calls(monkeypatch):
    from mist_cove.phase10 import benchmark
    monkeypatch.setattr(benchmark.subprocess,'check_output',lambda *a,**kw:'other process')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('must not reach CUDA runtime'))
    result=benchmark.cuda_smoke();assert result['status']=='BLOCKED' and result['new_cuda_model_operations']==0

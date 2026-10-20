import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('phase8_publication',ROOT/'analysis/mist_cove_phase8_v1/collect.py')
PUBLICATION=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PUBLICATION)

@pytest.mark.parametrize('private_key',['waveform','weights','optimizer','rng','logits','labels','probabilities'])
def test_private_payload_rejected_recursively(private_key,tmp_path):
    target=tmp_path/'public.json'
    with pytest.raises(ValueError):PUBLICATION.publish(target,{'rows':[{'nested':{private_key:[1,2]}}]})
    assert not target.exists()

def test_atomic_publication_never_overwrites_and_rejects_nonfinite(tmp_path):
    target=tmp_path/'public.json';PUBLICATION.publish(target,{'aggregate_mean':.5})
    before=target.read_bytes()
    with pytest.raises(FileExistsError):PUBLICATION.publish(target,{'aggregate_mean':.6})
    assert target.read_bytes()==before
    with pytest.raises(ValueError):PUBLICATION.publish(tmp_path/'nan.json',{'aggregate_mean':float('nan')})
    assert not list(tmp_path.glob('.phase8-*'))

def test_tampered_and_incomplete_receipt_rejected_before_publication(tmp_path):
    p=tmp_path/'receipt.json';p.write_text('{"status":"COMPLETE"}')
    with pytest.raises(ValueError,match='mismatch'):PUBLICATION.load_verified(p,'0'*64)
    p.write_text('{"status":"STOPPED"}')
    with pytest.raises(ValueError,match='complete'):PUBLICATION.load_verified(p)

def test_no_real_loader_or_dataset_discovery_in_phase8_modules():
    import ast
    banned={'load_development','SourceRecording','read_frozen_split','read_raw_edf','read_edf'}
    for p in (ROOT/'src/mist_cove/phase8').glob('*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                assert name not in banned

def test_gpu_occupied_gate_does_not_import_torch_or_allocate(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('phase8_cuda',ROOT/'scripts/check_mist_cove_phase8_cuda.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess,'check_output',lambda args,**kwargs:'some other compute process')
    result=module.availability()
    assert result['status']=='BLOCKED' and result['model_operations']==0

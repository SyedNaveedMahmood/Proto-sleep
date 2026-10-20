import json
import importlib.util
from pathlib import Path
import pytest
from mist_cove.phase9.reporting import privacy_check,publish_json

@pytest.mark.parametrize('key',['state_dict','optimizer','rng','margins','probabilities','nodes'])
def test_nested_private_state_or_per_mask_archive_rejected(key,tmp_path):
    p=tmp_path/'public.json'
    with pytest.raises(ValueError):publish_json(p,{'rows':[{'hidden':{key:[1,2]}}]})
    assert not p.exists()


def test_atomic_publication_refuses_overwrite_and_nonfinite(tmp_path):
    p=tmp_path/'public.json';publish_json(p,{'aggregate':.2});before=p.read_bytes()
    with pytest.raises(FileExistsError):publish_json(p,{'aggregate':.3})
    assert p.read_bytes()==before
    with pytest.raises(ValueError):publish_json(tmp_path/'bad.json',{'aggregate':float('nan')})
    assert not list(tmp_path.glob('.wic-*'))


def test_cuda_occupied_gate_without_device_operations(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('phase9_cuda_fixture',root/'scripts/check_mist_cove_phase9_cuda.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:'other compute process')
    result=module.availability();assert result['status']=='BLOCKED' and result['model_operations']==0

def test_publication_manifest_tamper_is_rejected(tmp_path,monkeypatch):
    from mist_cove.phase9 import reporting
    from mist_cove.phase2.provenance import file_sha
    monkeypatch.setattr(reporting,'ROOT',tmp_path);monkeypatch.setattr(reporting,'PUBLIC',tmp_path)
    value=tmp_path/'aggregate.txt';value.write_text('safe aggregate')
    (tmp_path/'publication_manifest.json').write_text(json.dumps({'files':{'aggregate.txt':file_sha(value)},'figure_sha256':{}}))
    assert reporting.verify_publication(check_git=False)['status']=='PASS'
    value.write_text('changed')
    with pytest.raises(ValueError,match='mismatch'):reporting.verify_publication(check_git=False)

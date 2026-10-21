import pytest
from mist_cove.phase11b.source import development_records,SourceAccess,validate_geometry,sealed_memfile
from mist_cove.phase11b.security import IntegrityError


@pytest.mark.parametrize('role',['test','reserved','ST','SHHS','TRAIN','../train'])
def test_role_rejected_before_any_descriptor_access(role):
    with pytest.raises(IntegrityError):development_records({},role)
    access=SourceAccess({}, {}, {'status':'PASS','state_unchanged':True})
    with pytest.raises(IntegrityError):access.records(role)


def test_missing_adapter_gate_blocks_access():
    with pytest.raises(IntegrityError):SourceAccess({}, {}, {'status':'PASS'})


def test_unissued_approved_role_record_rejected_without_descriptor_access(monkeypatch):
    access=SourceAccess({}, {}, {'status':'PASS','state_unchanged':True})
    monkeypatch.setattr(access,'records',lambda role:pytest.fail('unissued record caused descriptor read'))
    with pytest.raises(IntegrityError,match='unissued'):access.require_record({'split':'val'})


def test_sealed_memory_snapshot_is_readonly():
    import os
    with sealed_memfile(b'generated') as path:
        with open(path,'rb') as stream:assert stream.read()==b'generated'
        with pytest.raises(OSError):
            fd=os.open(path,os.O_WRONLY)
            try:os.write(fd,b'attack')
            finally:os.close(fd)


def test_original_geometry_and_gap_contract():
    r={'split':'train','study':'SC','signal_contract':{'montage':'Fpz-Cz','fs_native':100,'fs_output':100,'raw_units':'uV'},'epoch_grid':{'n_epochs':3,'samples_per_epoch':3000,'original_epoch_index':{'start':0,'stop':3,'step':1}}}
    validate_geometry(r);r['epoch_grid']['original_epoch_index']['step']=2
    with pytest.raises(IntegrityError):validate_geometry(r)

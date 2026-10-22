import io
import pytest
import torch
from mist_cove.phase11b.development import ReliabilityLedger
from mist_cove.phase11b.security import IntegrityError
from mist_cove.phase13.provenance import TrainOnlyAccess,isolated_rng
from mist_cove.phase13.model_adequacy import WidthLedger
from mist_cove.phase13.seals import save_sealed,load_sealed

@pytest.mark.parametrize('role',['val','reserved','ST','SHHS','TRAIN','../train'])
def test_train_only_before_descriptor_access(role,monkeypatch):
    access=TrainOnlyAccess({}, {}, {'status':'PASS','state_unchanged':True})
    from mist_cove.phase11b import source
    monkeypatch.setattr(source,'development_records',lambda *args:pytest.fail('forbidden descriptor read'))
    with pytest.raises(IntegrityError):access.records(role)
    with pytest.raises(IntegrityError):access.require_record({'split':role})

@pytest.mark.parametrize('routing',[False,True])
def test_small_is_exact_previous_architecture(routing):
    torch.manual_seed(123);old=ReliabilityLedger(routing)
    torch.manual_seed(123);new=WidthLedger('S',routing)
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key])
    x=torch.randn(2,1,3000);a=old(x);b=new(x)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

@pytest.mark.parametrize('size,routing,active,stored',[('S',False,1029,1078),('S',True,1078,1078),('L',False,11781,11974),('L',True,11974,11974)])
def test_capacity_counts_and_accounting(size,routing,active,stored):
    m=WidthLedger(size,routing);c=m.counts();assert (c['trainable'],c['stored'])==(active,stored)
    out=m(torch.randn(3,1,3000));torch.testing.assert_close(out['logits'],m.bias+out['global']+out['contributions'].sum(1),atol=0,rtol=0)

def test_rng_restored():
    state=torch.get_rng_state().clone()
    with isolated_rng(9):WidthLedger('L')
    assert torch.equal(state,torch.get_rng_state())

def test_sealed_state_identity_and_corruption_before_parse(tmp_path):
    import mist_cove.phase13.seals as seals
    model=WidthLedger();path=tmp_path/'generated.pt';identity={'run':'generated','step':3}
    save_sealed(path,{'state_dict':model.state_dict()},identity)
    if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,10):
        payload,_=load_sealed(path,identity);assert torch.equal(payload['state_dict']['bias'],model.bias)
    with pytest.raises(IntegrityError):load_sealed(path,{'run':'wrong'})
    path.write_bytes(b'mutated')
    with pytest.raises(IntegrityError):load_sealed(path,identity)

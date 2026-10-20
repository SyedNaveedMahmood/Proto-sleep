import copy
import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import steps,optimizer,seal,restore
from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_cove.phase8.provenance import output_directory,trusted_reference
from mist_cove.phase8.pooling import TrimmedModel
from mist_evidence.runtime import rng_state,restore_rng

C={'updates':6,'batch_size':5,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':5}

@pytest.mark.parametrize('factory',[TrimmedModel,lambda:make_model('M1')])
def test_exact_model_optimizer_all_rng_resume(factory,tmp_path):
    from mist_cove.phase2.training import configure_precision
    configure_precision('cpu');torch.manual_seed(928);torch.set_num_threads(2)
    m=factory();initial=copy.deepcopy(m.state_dict());rng=rng_state();d=generate(10,928)
    def fresh():
        m=factory();m.load_state_dict(initial);o=optimizer(m,C);g=torch.Generator().manual_seed(928);restore_rng(rng);return m,o,g
    m,o,g=fresh();steps(m,o,g,d.x,d.y,C,0,6,Budget(60));expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g.get_state(),rng_state())
    a,b,c=fresh();steps(a,b,c,d.x,d.y,C,0,3,Budget(60));identity={'scope':'GENERATED_ENGINEERING_FIXTURE'}
    proof=seal(tmp_path/'new.pt',a,b,c,3,identity);x,y,z=fresh();assert restore(tmp_path/'new.pt',proof,identity,x,y,z)==3
    steps(x,y,z,d.x,d.y,C,3,6,Budget(60));assert_exact(expected,(x.state_dict(),y.state_dict(),z.get_state(),rng_state()))
    (tmp_path/'new.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='before checkpoint deserialization'):restore(tmp_path/'new.pt',proof,identity,x,y,z)

def test_scope_and_original_untrusted_identity_rejected_before_io(tmp_path):
    class Forbidden:
        def __fspath__(self):pytest.fail('scope checked too late')
    with pytest.raises(ValueError):output_directory(Forbidden(),'RESERVED')
    with pytest.raises(ValueError):output_directory(tmp_path)
    with pytest.raises(ValueError):trusted_reference('single',123,'historical_M1')

import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate
from mist_cove.phase5.runtime import Budget
from mist_cove.phase8.evidence import matched_heads
from mist_cove.phase8.interventions import distortions
from mist_evidence.runtime import tensor_digest

C={'updates':2,'head_updates':2,'batch_size':5,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':5}

def test_failed_prerequisite_prevents_all_fitting():
    with pytest.raises(ValueError,match='prerequisite'):matched_heads(None,None,None,C,123,Budget(10),gate={'status':'FAIL'})
    with pytest.raises(ValueError,match='fixture'):matched_heads(None,None,None,C,123,Budget(10),gate={},engineering=True)

def test_complete_fixture_pipeline_shared_encoder_signed_accounting_and_eventless():
    m=make_model('M1');encoder=m.encoder.state_dict();d=generate(10,9381)
    ms,rs=matched_heads(encoder,d.x,d.y,C,123,Budget(60),gate={'scope':'GENERATED_ENGINEERING_FIXTURE'},engineering=True)
    assert set(ms)=={'E0','E1','E2','E3'} and len({r['encoder_sha256'] for r in rs.values()})==1
    for k,m in ms.items():
        assert tensor_digest(m.encoder.state_dict())==tensor_digest(encoder)
        ev=m(d.x,return_evidence=True);torch.testing.assert_close(ev.logits,ev.background_logits+ev.contributions.sum(1))
        zero=m(torch.zeros(5,1,3000),return_evidence=True);assert not zero.valid.any()
        if k=='E2':assert not ev.contributions.any()
        if k in {'E1','E3'}:
            from types import SimpleNamespace
            changed=m(d.base,return_evidence=True);fixed=m(d.base,return_evidence=True,witnesses=SimpleNamespace(intervals=ev.event_intervals,valid=ev.valid))
            assert torch.isfinite(changed.logits).all() and torch.isfinite(fixed.logits).all()

def test_donor_distortions_identity_and_energy_rejection():
    x=torch.randn(200);assert distortions(x,x)=={'energy_relative':0.,'spectral_l1':0.,'boundary_max':0.}
    assert distortions(x,x*10)['energy_relative']>90

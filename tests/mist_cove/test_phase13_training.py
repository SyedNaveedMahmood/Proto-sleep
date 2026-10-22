import copy
import pytest
import torch
from mist_cove.phase13.model_adequacy import WidthLedger
from mist_cove.phase13.provenance import isolated_rng
from mist_cove.phase13.optimization import sampler,one_update,training_payload,restore_payload
from mist_cove.phase13.seals import save_sealed,load_sealed
from mist_cove.phase13.support_operator import LedgerOperator
from mist_cove.phase13.gates import absolute_adequacy,routing_gain


def test_matched_exposures_across_width_and_route():
    x=torch.randn(20,1,3000);y=torch.arange(20)%5;hashes=[]
    for size in ('S','L'):
        for route in (False,True):
            with isolated_rng(123):
                m=WidthLedger(size,route);o=torch.optim.Adam(m.parameters(),lr=.001);g=sampler(123)
                hashes.append([one_update(m,o,x,y,g)['draw_sha256'] for _ in range(3)])
    assert all(h==hashes[0] for h in hashes)


def test_exact_sealed_resume_optimizer_and_sampler(tmp_path):
    if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))<(2,10):pytest.skip('positive restricted resume requires patched runtime')
    with isolated_rng(88):
        x=torch.randn(20,1,3000);y=torch.arange(20)%5;m=WidthLedger();o=torch.optim.Adam(m.parameters(),lr=.001);g=sampler(88);history=[]
        for _ in range(3):history.append(one_update(m,o,x,y,g))
        identity={'generated':True};p=tmp_path/'generated.pt';save_sealed(p,training_payload(m,o,g,3,history,identity),identity)
        expected=[one_update(m,o,x,y,g) for _ in range(4)]
        payload,_=load_sealed(p,identity);m2=WidthLedger();o2=torch.optim.Adam(m2.parameters(),lr=.001);g2=sampler(88)
        restore_payload(payload,m2,o2,g2,identity);actual=[one_update(m2,o2,x,y,g2) for _ in range(4)]
        assert expected==actual and torch.equal(g.get_state(),g2.get_state())
        assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
        for a,b in zip(o.state.values(),o2.state.values()):
            for key in a:assert torch.equal(a[key],b[key])


@pytest.mark.parametrize('size,route',[('S',False),('S',True),('L',False),('L',True)])
def test_operator_full_identity_hidden_noninterference_and_mutation(size,route):
    m=WidthLedger(size,route).eval();operator=LedgerOperator(m);x=torch.randn(1,1,3000);cache=operator.cache(x)
    full=torch.ones(1,15,dtype=torch.bool);torch.testing.assert_close(operator.logits(cache,full)[0],cache.native,atol=1e-6,rtol=1e-6)
    mask=full.clone();mask[0,3]=False;changed=x.clone();changed[:,:,600:800]+=100
    other=operator.cache(changed)
    for neutral in (0.,.01):torch.testing.assert_close(operator.logits(cache,mask,neutral),operator.logits(other,mask,neutral),atol=1e-6,rtol=1e-6)
    cache.contributions[0,0]+=1
    with pytest.raises(ValueError):operator.logits(cache,full)


def test_operator_rejects_topology_and_gate_semantic_changes():
    model=WidthLedger().eval();operator=LedgerOperator(model);cache=operator.cache(torch.randn(1,1,3000))
    model.routing=False
    with pytest.raises(ValueError):operator.logits(cache,torch.ones(1,15,dtype=torch.bool))


def test_clean_penalty_and_class_collapse_cannot_be_promoted():
    def row(clean,robust,classes=5):
        def metrics(f1):return {'mean_subject_macro_f1':f1,'per_stage_recall':[.7]*5,'per_stage_f1':[.7]*5}
        return {'development':{'clean':{'metrics':metrics(clean),'collapse':{'predicted_classes':classes,'largest_fraction':.4}},'strong_local_HF':{'metrics':metrics(robust)}}}
    assert absolute_adequacy(row(.7,.6)) and not absolute_adequacy(row(.7,.6,1))
    assert not routing_gain(row(.6,.9),row(.7,.5))

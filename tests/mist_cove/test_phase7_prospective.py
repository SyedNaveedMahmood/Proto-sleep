import copy
import json
from pathlib import Path

import pytest
import torch

from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget
from mist_cove.phase7.prospective import (CONFIG, matched, pooled, oracle_mask, head, feature_paths,
    encode, steps, optimizer, seal, restore, output_directory, analyze)
from mist_evidence.runtime import tensor_digest, rng_state

C=json.loads(CONFIG.read_text())


@pytest.mark.parametrize('task',['single','multiple'])
@pytest.mark.parametrize('strength',[0.,.5,1.,2.])
def test_exact_matched_background_diagnostic_preservation(task,strength):
    a,b=matched(10,9123,task,C,strength=strength)
    assert torch.equal(a.events,b.events) and torch.equal(a.background,b.background)
    assert torch.equal(a.noise,b.noise) and torch.equal(a.artifact,b.artifact)
    assert a.intervals==b.intervals and torch.equal(a.y,b.y)
    assert torch.equal(b.x,b.base+b.events.sum(1))
    assert torch.equal(a.x,matched(10,9123,task,C,strength=0)[1].x)
    if strength:assert not torch.equal(a.x,b.x)


@pytest.mark.parametrize('task',['single','multiple'])
def test_nuisance_independent_of_swapped_label_and_full_geometry(task):
    a,b=matched(10,9147,task,C)
    x,y=matched(10,9147,task,C,labels=(a.y+1)%5)
    assert torch.equal(a.base,x.base) and torch.equal(b.base,y.base)
    assert not torch.equal(a.events,x.events) and a.intervals==x.intervals
    assert b.x.shape==(10,1,3000)
    mask=oracle_mask(a,contain=True)
    assert mask.shape==(10,14) and mask.any(1).all()
    assert all(0<=lo<hi<=3000 and hi-lo==200 for row in a.intervals for lo,hi in row)


def test_eventless_does_not_receive_extra_nuisance():
    a,b=matched(10,9184,'single',C,condition='eventless')
    assert not a.x.any() and not b.x.any() and b.intervals==[[]]*10
    with pytest.raises(ValueError,match='nonempty'):pooled(torch.ones(10,14,18),oracle_mask(a))


def test_feature_geometry_and_signed_head_path_reconciliation():
    torch.manual_seed(9199);model=make_model('M1');a,b=matched(10,9199,'multiple',C)
    f,r=encode(model,a.x);g,_=encode(model,b.x)
    assert f.shape==(10,14,18) and r.shape==(10,14)
    assert pooled(f).shape==(10,36)
    torch.testing.assert_close(head(model,pooled(f)),model(a.x))
    result=feature_paths(model,f,g,oracle_mask(a))
    assert result['max_reconciliation_error']<1e-5
    assert result['off_manifold_internal_intervention']


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    try:
        import numpy as np
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
    except ImportError:pass
    return a==b


def test_exact_resume_model_optimizer_global_and_batch_rng(tmp_path):
    from mist_cove.phase2.training import configure_precision
    configure_precision('cpu');torch.set_num_threads(2)
    a,_=matched(10,9200,'single',C);c={**C,'batch_size':5};identity={'new_experiment':'fixture','seed':9200}
    torch.manual_seed(9200);initial=make_model('M1').state_dict();initial=copy.deepcopy(initial)
    start_rng=rng_state()
    from mist_evidence.runtime import restore_rng
    def fresh():
        m=make_model('M1');m.load_state_dict(initial);o=optimizer(m,c);g=torch.Generator().manual_seed(9200+711)
        restore_rng(start_rng);return m,o,g
    m,o,g=fresh();steps(m,o,g,a.x,a.y,c,0,6,Budget(60));expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g.get_state(),rng_state())
    m,o,g=fresh();steps(m,o,g,a.x,a.y,c,0,3,Budget(60));record=seal(tmp_path/'fresh.pt',m,o,g,3,identity)
    m2=make_model('M1');o2=optimizer(m2,c);g2=torch.Generator()
    assert restore(tmp_path/'fresh.pt',record,identity,m2,o2,g2)==3
    steps(m2,o2,g2,a.x,a.y,c,3,6,Budget(60))
    assert equal_state(expected,(m2.state_dict(),o2.state_dict(),g2.get_state(),rng_state()))


@pytest.mark.parametrize('tamper',['weights','seal','identity'])
def test_creation_proof_rejected_before_deserialization(tmp_path,monkeypatch,tamper):
    import mist_cove.phase7.prospective as module
    m=make_model('M1');o=optimizer(m,C);g=torch.Generator();identity={'new':'fixture'};p=tmp_path/'new.pt'
    record=seal(p,m,o,g,0,identity)
    if tamper=='weights':p.write_bytes(b'changed')
    elif tamper=='seal':p.with_suffix('.seal.json').write_text('{}')
    else:identity={'new':'different'}
    def forbidden(*a):pytest.fail('deserialized without authentic creation proof')
    monkeypatch.setattr(module,'load_trusted',forbidden)
    with pytest.raises(ValueError,match='before checkpoint deserialization'):restore(p,record,identity,m,o,g)


def test_output_cannot_reuse_historical_run_or_existing_output(tmp_path):
    with pytest.raises(ValueError):output_directory(tmp_path,'SYNTHETIC_ONLY')
    class ForbiddenPath:
        def __fspath__(self):pytest.fail('unauthorized path accessed')
    with pytest.raises(ValueError):output_directory(ForbiddenPath(),'reserved')


def test_complete_audit_batches_localization_per_example(tmp_path):
    torch.manual_seed(9250);model=make_model('M1');a,b=matched(10,9250,'multiple',C)
    f,_=encode(model,a.x)
    result,features=analyze(model,a,b,f,C,Budget(60),tmp_path/'generated-details.pt')
    assert 0<=result['localization_clean']['topk_hit']<=1
    assert result['paths']['max_reconciliation_error']<1e-5
    assert features['local'][0].shape==(10,18)
    assert len(result['oracle_group_effects'])==5

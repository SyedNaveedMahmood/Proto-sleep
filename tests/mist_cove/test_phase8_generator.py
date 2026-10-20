import pytest
import torch
from mist_cove.phase6.generator import generate
from mist_cove.phase8.generator import TAXONOMY,STRENGTHS,intervene,training_draw,information_loss

@pytest.mark.parametrize('task',['single','multiple'])
@pytest.mark.parametrize('kind',TAXONOMY)
def test_pairing_geometry_and_independence(task,kind):
    a=generate(10,834151,task);b=intervene(a,kind,1.)
    swapped=generate(10,834151,task,labels=(a.y+1)%5);other=intervene(swapped,kind,1.)
    assert torch.equal(b.base,other.base) and torch.equal(b.nuisance,other.nuisance)
    assert torch.equal(a.y,b.y) and a.latents==b.latents and a.intervals==b.intervals
    assert torch.equal(a.events*.2 if kind=='weak_diagnostic' else a.events,b.events)
    assert b.x.shape==(10,1,3000) and b.x.unfold(-1,400,200).shape[-2]==14
    assert torch.equal(b.x,b.base+b.events.sum(1)) and torch.isfinite(b.x).all()
    zero=intervene(a,kind,0.);assert torch.equal(zero.base,a.base)
    for iv in b.nuisance_intervals:assert all(0<=lo<hi<=3000 for lo,hi in iv)
    if kind=='disjoint':
        for iv,truth,old in zip(b.nuisance_intervals,a.intervals,a.nuisance_intervals):
            assert all(hi<=aa or lo>=bb for lo,hi in iv[len(old):] for aa,bb in truth)

@pytest.mark.parametrize('kind',TAXONOMY)
def test_strength_replays_identical_sites(kind):
    a=generate(5,834152);b=intervene(a,kind,.5);c=intervene(a,kind,1.)
    torch.testing.assert_close(b.base-a.base,(c.base-a.base)*.5,atol=2e-5,rtol=1e-5)
    assert b.nuisance_intervals==c.nuisance_intervals

@pytest.mark.parametrize('method',['B1','A1','T1'])
def test_fixed_training_distribution(method):
    a,y=training_draw(100,834153,'single',method);b,z=training_draw(100,834153,'single',method)
    assert torch.equal(a,b) and torch.equal(y,z)
    clean=generate(100,834153,'single','train');assert torch.equal(a[::4],clean.x[::4])
    assert a.shape==(100,1,3000)

def test_information_loss_and_invalid_conditions():
    a=generate(10,834154,'multiple','competitive')
    assert torch.equal(information_loss(a,'removed'),a.base)
    assert torch.equal(information_loss(a,'partial'),a.base+a.events[:,1])
    assert torch.equal(information_loss(a,'background'),a.background+a.noise)
    with pytest.raises(ValueError):intervene(a,'unknown',1.)
    with pytest.raises(ValueError):intervene(a,'impulse',.7)

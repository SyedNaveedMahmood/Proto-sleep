"""Generated correctness fixtures; no held-out scientific tuning."""
import copy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from mist_cove.phase6.generator import generate,latent_draw,compose,class_swap,conditions_for,rng
from mist_cove.phase6.features import features,signal_view
from mist_cove.phase6.runtime import private_output,historical
from mist_cove.phase6.evaluation import location
from mist_cove.phase6.experiment import head_stage


@pytest.mark.parametrize('task',['single','multiple'])
@pytest.mark.parametrize('condition',sorted({'train','clean','competitive','removed','nuisance_swap','position_shift','partial','eventless','redundant','complementary','repeated','correlated','adjacent','overlap','weak','artifact'}))
def test_geometry_reproduction_and_composition(task,condition):
    data=generate(10,901,task,condition);again=generate(10,901,task,condition,latents=data.latents)
    assert data.x.shape==(10,1,3000) and data.x.dtype==torch.float32 and torch.isfinite(data.x).all()
    assert torch.equal(data.x,again.x) and data.latents==again.latents
    assert torch.equal(data.x,compose(data.base,data.events))
    for truth in data.intervals:
        for lo,hi in truth:
            assert 0<=lo<hi<=3000 and hi-lo==200
            assert any(a<=lo and hi<=a+400 for a in range(0,2601,200))
    if condition in {'removed','eventless'}:assert all(not t for t in data.intervals)
    if condition=='eventless':assert not data.x.any() and not data.base.any()


@pytest.mark.parametrize('task',['single','multiple'])
@pytest.mark.parametrize('condition',['clean','competitive','redundant','complementary','overlap','position_shift','removed','eventless'])
def test_exact_class_pair_preserves_baseline_and_geometry(task,condition):
    data=generate(10,902,task,condition);changed=class_swap(data)
    assert torch.equal(data.base,changed.base)
    assert data.intervals==changed.intervals
    assert torch.equal(data.background,changed.background) and torch.equal(data.noise,changed.noise)
    mask=torch.zeros_like(data.x,dtype=torch.bool)
    for i,truth in enumerate(data.intervals):
        for lo,hi in truth:mask[i,0,lo:hi]=True
    assert torch.equal(data.x[~mask],changed.x[~mask])
    assert torch.equal(data.x[~mask],data.base[~mask])
    for a,b in zip(data.latents,changed.latents):assert {k:v for k,v in a.items() if k!='label'}=={k:v for k,v in b.items() if k!='label'}


@pytest.mark.parametrize('stream',range(1,9))
def test_streams_are_explicitly_independent(stream):
    assert np.array_equal(rng(903,stream,1).integers(0,100,20),rng(903,stream,1).integers(0,100,20))
    assert not np.array_equal(rng(903,stream,1).integers(0,100,20),rng(903,stream+100,1).integers(0,100,20))


@pytest.mark.parametrize('task',['single','multiple'])
def test_nuisance_pair_and_exact_group_deletion(task):
    data=generate(10,904,task);other=generate(10,904,task,'nuisance_swap',latents=data.latents)
    assert torch.equal(data.events,other.events) and not torch.equal(data.base,other.base)
    assert torch.equal(compose(data.base,data.events,torch.zeros(10,2)),data.base)
    if task=='multiple':
        a=compose(data.base,data.events,torch.tensor([[0.,1.]]).expand(10,-1))
        assert torch.equal(a,data.base+data.events[:,1])


@pytest.mark.parametrize('task',['single','multiple'])
@pytest.mark.parametrize('name',['background','nuisance','position','artifact','diagnostic_amplitude','diagnostic_spectrum','full_magnitude'])
def test_feature_geometry_and_metadata_isolation(task,name):
    a=generate(10,905,task);b=generate(10,906,task,'competitive')
    x=features(a,name);other=features(b,name)
    assert x.ndim==2 and x.shape==other.shape and torch.isfinite(x).all()
    if name in {'background','nuisance','position','artifact'}:
        swapped=class_swap(a)
        assert torch.equal(features(a,name),features(swapped,name))


def test_redundant_and_complementary_supports_differ_without_label_geometry():
    a=generate(10,907,'multiple','redundant');b=generate(10,907,'multiple','complementary')
    assert a.intervals==b.intervals and torch.equal(a.base,b.base)
    assert not torch.equal(a.events,b.events)
    assert all(len(t)==2 for t in a.intervals)
    # A remaining complete witness retains its actual waveform; no requirement
    # makes a remaining complementary factor individually identifying.
    assert torch.equal(compose(a.base,a.events,torch.tensor([[0.,1.]]).expand(10,-1)),a.base+a.events[:,1])


def test_empty_truth_and_localization_geometry():
    r=location([(0,400)],[],[(0,200)])
    assert r['topk_hit'] is None and r['empty_truth_selected_count']==1
    assert location([(0,400)],[(100,300)],[])['coverage']==1
    assert location([(800,1200)],[(100,300)],[(900,1100)])['false_nuisance_fraction']==1


@pytest.mark.parametrize('role',['reserved','train','val','real','external'])
def test_role_guard_precedes_any_path_access(role):
    class Trap:
        def __fspath__(self):raise AssertionError('forbidden path touched')
    with pytest.raises(ValueError,match='before filesystem'):private_output(Trap(),role)


def test_failed_learnability_never_constructs_evidence_model(tmp_path,monkeypatch):
    import mist_cove.phase6.experiment as module
    monkeypatch.setattr(module,'make_model',lambda *a,**k:(_ for _ in ()).throw(AssertionError('not permitted')))
    assert head_stage({'status':'FAIL'},None,None,tmp_path,None,[])['optimizer_updates']==0


def test_prior_publication_hashes_unchanged():assert historical()['phase5_publication_sha256']

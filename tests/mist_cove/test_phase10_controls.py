import torch
from mist_cove.phase10.mock_controls import ControlAdapter


def test_local_global_neutral_and_wrong_prediction_controls():
    ids=torch.tensor([0,1<<2,1<<10,(1<<2)|(1<<10),32767])
    masks=((ids[:,None]>>torch.arange(15))&1).bool()
    for name,expected in [('localized',[1,0,1,0,0]),('redundant',[1,0,0,0,0]),
                          ('complementary',[1,1,1,0,0]),('whole_epoch',[1,1,1,1,0]),
                          ('neutral_bias',[0,0,0,0,0]),('eventless',[1,1,1,1,1])]:
        a=ControlAdapter(name);cache=a.cache(a.waveform())
        assert a.logits(cache,masks)[:,0].argmax(-1).tolist()==expected
    a=ControlAdapter('incorrect_explainable');cache=a.cache(a.waveform())
    assert int(a.logits(cache,masks[-1:])[0,0].argmax())!=a.truth_label
    assert a.oracle_atoms==[2]


def test_changed_mean_semantics_can_create_class_change():
    a=ControlAdapter('aggregation_sensitivity');cache=a.cache(a.waveform());mask=torch.zeros(1,15,dtype=torch.bool);mask[0,2]=True
    trace=a.features(cache,mask);f=trace.features.clone();f[...,0]*=59/int(trace.valid_window_counts[0,0])
    assert float(a.score(trace.features))<0<float(a.score(f))


def test_privileged_control_hidden_samples_cannot_affect_visible_features():
    a=ControlAdapter('mixed_global_local');x=a.waveform();y=x.clone();mask=torch.zeros(1,15,dtype=torch.bool);mask[0,2]=True
    y[:,:,600:]+=20000
    assert torch.equal(a.features(a.cache(x),mask).features,a.features(a.cache(y),mask).features)


def test_constant_head_has_zero_feature_attribution_without_autograd_graph():
    from mist_cove.phase10.mock_study import gradient_ranking
    a=ControlAdapter('neutral_bias');ranking,receipt=gradient_ranking(a,a.cache(a.waveform()),0)
    assert ranking==list(range(15))
    assert receipt['atom_attribution_absolute_sum']==0
    assert receipt['global_attribution_absolute_sum']==0

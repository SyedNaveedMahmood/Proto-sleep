import numpy as np
import pytest
import torch
from mist_cove.phase11b.calibration import fit_temperature
from mist_cove.phase11b.training import training_stream
from mist_cove.phase11b.source import select_epochs
from mist_cove.phase11b.evaluation import clustered_mean,probability_metrics

@pytest.mark.parametrize('roles',[['val'],['reserved'],['train_fit','internal_dev'],[]])
def test_calibration_rejects_wrong_roles(roles):
    with pytest.raises(ValueError):fit_temperature(np.zeros((len(roles),5)),np.zeros(len(roles),int),roles)

def test_matched_optimization_streams_and_no_seed_reuse():
    a=list(training_stream(123,100));b=list(training_stream(123,100));c=list(training_stream(456,100))
    assert len(a)==300 and sum(len(x[0]) for x in a)==4800
    assert all(torch.equal(x,y) for row1,row2 in zip(a,b) for x,y in zip(row1,row2))
    assert not torch.equal(a[0][0],c[0][0])

def test_sampling_preserves_missing_strata_and_night_positions(monkeypatch):
    from mist_transfer import preprocessing
    records=[{'subject_id':'generated','recording_id':'B','night':2,'labels':np.array([4,0,-1,1])},
             {'subject_id':'generated','recording_id':'A','night':1,'labels':np.array([-1,0,1,-1,0])}]
    monkeypatch.setattr(preprocessing,'epoch_targets',lambda r:(r['labels'],r['labels']>=0,None))
    selected,missing=select_epochs(records,2)
    wake=[v for v in selected if v['stage']==0]
    assert [(v['recording_id'],v['epoch']) for v in wake]==[('A',1),('A',4)]
    assert sum(v['missing'] for v in missing)==5
    assert all(v['epoch']>=0 for v in selected)

def test_clustered_uncertainty_does_not_treat_epochs_as_subjects():
    a=clustered_mean([0,0,1],['a','a','b']);b=clustered_mean([0,1],['a','b'])
    assert a==b and a['mean']==.5 and a['n_participants']==2
    m=probability_metrics(np.zeros((2,5)),np.array([0,1]));assert m['Brier']==pytest.approx(.8)

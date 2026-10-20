import pytest
import torch
from mist_cove.phase8.calibration import fit_temperature,rejection_threshold,audit
from mist_cove.phase8.evaluation import paired_accuracy_interval

@pytest.mark.parametrize('role',['LOCKED','VAL','RESERVED','TRAIN','test'])
def test_fit_boundary_before_tensor_access(role):
    with pytest.raises(ValueError):fit_temperature(None,None,role=role)
    with pytest.raises(ValueError):rejection_threshold(None,role=role)

def test_temperature_fitted_on_separate_role_and_metrics():
    torch.manual_seed(321);y=torch.arange(100)%5;logits=torch.randn(100,5)*5
    t=fit_temperature(logits,y,role='SYNTHETIC_CALIBRATION');assert .25<=t<=6.4
    threshold=rejection_threshold(logits/t,role='SYNTHETIC_CALIBRATION');result=audit(logits,y,t,threshold)
    assert result['temperature_scaled']['nll']<=result['raw']['nll']
    assert sum(x['count'] for x in result['raw']['reliability_bins'])==100
    assert len(result['raw']['per_stage_recall'])==5
    assert paired_accuracy_interval(logits,logits,y,123)['bootstrap_ci95']==[0.,0.]

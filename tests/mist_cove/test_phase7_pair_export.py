import importlib.util
from pathlib import Path
import math
import torch


def test_per_pair_export_retains_signed_margins_and_all_five_probabilities():
    script=Path(__file__).resolve().parents[2]/'analysis/mist_cove_phase7_v1/complete_pair_audits.py'
    spec=importlib.util.spec_from_file_location('phase7_pair_export',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.per_example(torch.tensor([[2.,0.,0.,0.,0.],[0.,2.,0.,0.,0.]]),torch.tensor([0,0]))
    assert result['logits'].shape==result['probabilities'].shape==(2,5)
    assert result['prediction'].tolist()==[0,1]
    assert result['margin'].tolist()==[2.,-2.]
    torch.testing.assert_close(result['probabilities'].sum(1),torch.ones(2))
    assert abs(float(result['confidence'][0])-math.exp(2)/(math.exp(2)+4))<1e-6
    assert result['loss'][0]<result['loss'][1] and torch.isfinite(result['entropy']).all()

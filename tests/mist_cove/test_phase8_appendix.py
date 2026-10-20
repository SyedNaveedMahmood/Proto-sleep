import importlib.util
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]

def test_descriptive_endpoints_reconcile_identity_and_single_group(monkeypatch):
    spec=importlib.util.spec_from_file_location('phase8_collect_fixture',ROOT/'analysis/mist_cove_phase8_v1/collect.py')
    collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect);monkeypatch.setitem(sys.modules,'collect',collect)
    spec=importlib.util.spec_from_file_location('phase8_appendix_fixture',ROOT/'analysis/mist_cove_phase8_v1/complete_diagnostics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from mist_cove.phase4.model import make_model
    from mist_cove.phase6.generator import generate
    from mist_cove.phase7.prospective import encode
    m=make_model('M1').eval();d=generate(10,92312);f,r=encode(m,d.x)
    with torch.no_grad():l=m.decision(f,r)[0]
    result=module.local_summary(m,d,f,l)
    assert result['prediction_change']==0 and result['pooled_drift']==0
    assert abs(result['diagnostic_feature_cosine']-1)<1e-6
    assert result['pooled_head_arithmetic_residual']==0
    group=module.group_summary(m,d)
    assert group['joint_minus_sum_individual_abs_mean']==0
    assert group['mean_abs_true_joint']==group['mean_largest_abs_individual']

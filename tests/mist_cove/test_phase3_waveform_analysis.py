"""Generated-input analysis checks; no real source loader is invoked."""
import importlib.util
from pathlib import Path
import pytest
import torch
from mist_cove.phase2.model import make_model
from mist_cove.phase2.report import local_evidence_report
from mist_cove.phase2.traces import EpochIdentity,source_ledger
MODULE=Path(__file__).resolve().parents[2]/'analysis/mist_cove_phase3_v1/waveform_analysis_v1.py'
spec=importlib.util.spec_from_file_location('phase3_waveform_analysis',MODULE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.fixture
def generated():
 torch.set_num_threads(2);torch.manual_seed(7)
 x=torch.randn(1,1,3000);model=make_model('q1').eval();return model,x

def test_disjoint_donor():
 torch.manual_seed(9);donor,info=m.choose_donor(torch.randn(3000)*.1,[(0,200),(400,800)])
 assert info['supported'] and donor is not None
 assert all(not(donor[0]<b and donor[1]>a) for a,b in [(0,200),(400,800)])

def test_fixed_compatibility_cap_not_relaxed():
 donor,info=m.choose_donor(torch.arange(3000).float(),[(0,200)])
 assert donor is None and not info['supported'] and info['minimum_rms_uV']>3

def test_groups_declared_from_original_witnesses(generated):
 model,x=generated;ev=model(x,return_evidence=True)
 groups,intervals,bands,_=m.declared_groups(model,x,ev)
 assert groups['all_selected']==intervals
 assert all(len(v)>=2 for k,v in groups.items() if k!='all_selected')
 assert all(0<=a<b<=3000 and b-a==200 for a,b in intervals)

def test_transforms_reproducible_and_physical_calibration(generated):
 model,x=generated;a=m.transformations(model,x,123);b=m.transformations(model,x,123)
 assert all(torch.equal(a[k],b[k]) for k in a)
 torch.testing.assert_close(model.physical_uV(a['physical_gain_1.01']),model.physical_uV(x)*1.01)

def test_all_stage_counterfactual_accounting(generated):
 model,x=generated;ev=model(x,return_evidence=True);groups,*_=m.declared_groups(model,x,ev)
 report=local_evidence_report(model,x,groups,donor_specs=[{'strategy':'bridge'}])
 for group in report['groups'].values():
  for donor in group['donors']:
   assert len(donor['A_additive_contribution'])==len(donor['C_group_counterfactual_effect'])==5
   assert max(abs(v) for v in donor['reconciliation_residual'])<1e-5

def test_source_ledger_preserves_original_epoch(generated):
 model,x=generated;ev=model(x,return_evidence=True)
 identity=EpochIdentity('generated_val','generated_record','1','val',17,'a'*64,ev.physical_input_sha256[0],True)
 ledger=source_ledger(ev,[identity])
 assert ledger and all(17*3000<=v['absolute_start_sample']<18*3000 for v in ledger)

def test_eventless_input_does_not_fabricate_groups(generated):
 model,x=generated;flat=torch.zeros_like(x);ev=model(flat,return_evidence=True)
 assert m.declared_groups(model,flat,ev)[0]=={} and torch.isfinite(ev.logits).all()

def test_empty_summary_is_json_compatible():
 assert m.safe_mean([None,None]) is None and m.safe_mean([1.,None,3.])==2.

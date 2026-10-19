"""Tiny generated plumbing and internal accounting, not scientific scores."""
import json
from pathlib import Path
import pytest
import torch
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate
from mist_cove.phase6.correctness import exact_resume
from mist_cove.phase6.evaluation import evaluate
from mist_cove.phase6.audit import audit,donor_diagnostics
from mist_cove.phase5.runtime import Budget


@pytest.mark.parametrize('method',['M1','M2','M3','M2_rms'])
def test_exact_generated_model_optimizer_rng_resume(tmp_path,method):
    assert exact_resume(tmp_path,method)['model_optimizer_python_numpy_cpu_cuda_rng_exact']


@pytest.mark.parametrize('method',['M1','M2','M3','M2_rms'])
def test_five_outputs_localization_and_signed_counterfactual(method):
    c=json.loads(Path('configs/mist_cove_phase6_synthetic_v1.json').read_text())
    torch.manual_seed(908);model=make_model(method).eval();data=generate(5,909,'multiple')
    result=evaluate(model,data,c,Budget(30),localize=True,seed=910)
    assert result['max_additive_residual']<1e-5
    assert 0<=result['localization']['model']['precision']<=1
    report=audit(model,data,c,909)
    assert report['max_reconciliation_residual']<1e-5
    assert all(r['internal_deletion_is_physiological'] is False for r in report['examples'])
    for r in report['examples']:
        assert len(r['individual_full_effects'])==2
        assert len(r['oracle_effect']['full'])==5
        if r['accepted_donor_count']<2:assert r['accepted_donor_spread'] is None


def test_rejected_donors_are_not_accepted_by_finite_values():
    guard={'relative_energy_max':.5,'relative_magnitude_l1_max':.5,'boundary_change_uV_max':5}
    x=torch.ones(1,1,3000);changed=x.clone();changed[:,:,100:300]=1000
    assert not donor_diagnostics(x,changed,[(100,300)],guard)['accepted']


def test_exploratory_end_to_end_runner_plumbing(tmp_path,capsys):
    from mist_cove.phase6.experiment import leakage_stage,conventional_stage,head_stage
    c=json.loads(Path('configs/mist_cove_phase6_synthetic_v1.json').read_text())
    c.update(tasks=['single'],seeds=[914],train_examples=10,eval_examples=10,stress_examples=10,pair_examples=10,updates=1,head_updates=1,bootstrap_replicates=2)
    c['conditions']=['clean','competitive','removed','eventless']
    rows=[];budget=Budget(60)
    leakage_stage(c,tmp_path,budget,rows)
    gate,common=conventional_stage(c,tmp_path,budget,rows)
    # Explicit mocked prerequisite exercises plumbing only, at disjoint debug
    # seeds and one update; it is never a scientific gate override in run().
    result=head_stage({'status':'PASS'},common,c,tmp_path,budget,rows)
    assert result['status'] in {'PASS','FAIL'}
    assert len([r for r in rows if r['stage']=='VI-4'])==4
    assert all(r['training']['optimizer_updates']==1 for r in rows)
    assert len({r['encoder_sha256'] for r in rows if r['stage']=='VI-4'})==1
    capsys.readouterr()

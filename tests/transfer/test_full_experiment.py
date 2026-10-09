"""P3B operational controls and deterministic CPU/CUDA boundary resume."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_source_protocol import array_data
from mist_evidence.runtime import (EpochPaused,TrainConfig,train,seed_all,load_trusted,tensor_digest)
from mist_transfer.baselines import make_baseline
from mist_transfer.experiment import evaluate_source
from mist_transfer.full_experiment import (validate_configuration,training_config,verify_grid,
    verify_boundary,EpochObserver,aggregate_results,VARIANTS,SEEDS,run_full,configure_precision)


@pytest.fixture(autouse=True)
def cpu_threads():torch.set_num_threads(2)


def configuration():return json.loads((Path(__file__).resolve().parents[2]/'configs/p3b_sc_matched_v1.json').read_text())


@pytest.mark.parametrize('change',[{'max_epochs':1},{'patience':2},{'smoke':True},{'amp':True},
    {'seeds':[123]},{'variants':['evidence_aug']},{'reserved_access':True},{'lr':.001}])
def test_frozen_config_rejects_recipe_changes(change):
    c=configuration();validate_configuration(c);c.update(change)
    with pytest.raises(ValueError,match='frozen full-data'):validate_configuration(c)


def test_proposed_hyperparameters_retained_and_pause_does_not_shorten_budget():
    cfg=configuration();prior=json.loads((Path(__file__).resolve().parents[2]/'configs/p3b_sc_matched_proposed.json').read_text())
    for k in ['variants','seeds','max_epochs','patience','lr','weight_decay','core_epochs','encode_batch','grad_clip','amp']:
        assert cfg[k]==prior[k]
    recipe=training_config(cfg,123);assert recipe.epochs==60 and recipe.patience==12 and recipe.deterministic
    with pytest.raises(ValueError,match='unapproved seed'):training_config(cfg,999)


def test_explicit_full_authorization_rejected_before_any_file_open():
    with patch.object(Path,'open',side_effect=AssertionError('access before authorization')):
        with pytest.raises(ValueError,match='authorize-full'):run_full('unused','unused')


def test_smoke_or_wrong_grid_cannot_enter_full_experiment():
    rows,_=array_data()
    with pytest.raises(ValueError,match='full-data'):verify_grid([rows[0]],[rows[1]])


def resume_fixture(tmp_path,variant,device):
    configure_precision()
    records,bank=array_data();cfg=TrainConfig(epochs=2,patience=12,core_epochs=4,encode_batch=4,seed=123)
    def fresh():seed_all(123,True);return make_baseline(variant,bank,{'mean_uV':0.,'std_uV':1.})
    args=([records[0]],[records[1]],cfg)
    train(fresh(),*args,tmp_path/'full',bank,{'test':'p3b-exact'},device,evaluate_fn=evaluate_source,monitor=True)
    with pytest.raises(EpochPaused):
        train(fresh(),*args,tmp_path/'resumed',bank,{'test':'p3b-exact'},device,evaluate_fn=evaluate_source,pause_after_epoch=1,monitor=True)
    assert not (tmp_path/'resumed/COMPLETE.json').exists()
    assert (tmp_path/'resumed/PAUSED.json').exists()
    assert load_trusted(tmp_path/'resumed/last.pt')['epoch']==1
    train(fresh(),*args,tmp_path/'resumed',bank,{'test':'p3b-exact'},device,resume=True,evaluate_fn=evaluate_source,monitor=True)
    assert not (tmp_path/'resumed/PAUSED.json').exists()
    assert (tmp_path/'resumed/COMPLETE.json').exists()
    a=load_trusted(tmp_path/'full/last.pt');b=load_trusted(tmp_path/'resumed/last.pt')
    assert a['epoch']==b['epoch']==2 and tensor_digest(a['state_dict'])==tensor_digest(b['state_dict'])
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    for x,y in zip(a['rng']['cuda'],b['rng']['cuda']):assert torch.equal(x,y)
    for k in a['optimizer']['state']:
        for name,value in a['optimizer']['state'][k].items():assert torch.equal(value,b['optimizer']['state'][k][name])
    assert [h['val_subject_macro_f1'] for h in a['history']]==[h['val_subject_macro_f1'] for h in b['history']]


def test_cpu_pause_resume(tmp_path):resume_fixture(tmp_path,'evidence',torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required; CPU fallback forbidden')
@pytest.mark.parametrize('variant',VARIANTS)
def test_cuda_exact_pause_resume_all_models(tmp_path,variant):resume_fixture(tmp_path,variant,torch.device('cuda'))


def test_paused_checkpoint_corruption_fails_before_resume(tmp_path):
    rows,bank=array_data();cfg=TrainConfig(epochs=2,core_epochs=4,encode_batch=4)
    def fresh():seed_all(123);return make_baseline('evidence',bank,{'mean_uV':0.,'std_uV':1.})
    with pytest.raises(EpochPaused):train(fresh(),[rows[0]],[rows[1]],cfg,tmp_path,bank,{},torch.device('cpu'),evaluate_fn=evaluate_source,pause_after_epoch=1)
    (tmp_path/'last.pt').write_bytes(b'corrupted; never deserialize')
    with pytest.raises(ValueError,match='paused checkpoint digest'):
        train(fresh(),[rows[0]],[rows[1]],cfg,tmp_path,bank,{},torch.device('cpu'),resume=True,evaluate_fn=evaluate_source)


def test_unsealed_boundary_and_incomplete_aggregation_rejected(tmp_path):
    (tmp_path/'last.pt').write_bytes(b'never deserialize')
    with pytest.raises(ValueError,match='sealed epoch boundary'):verify_boundary(tmp_path)
    with pytest.raises(ValueError,match='three seeds'):aggregate_results([])

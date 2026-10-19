import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mist_cove.phase4 import source
from mist_cove.phase4.correctness import exact_resume
from mist_cove.phase4.donors import fit_compatibility
from mist_cove.phase2.synthetic_runtime import mock_recordings,synthetic_bank

@pytest.fixture
def config():return json.loads(Path('configs/mist_cove_phase4_source_prepared_v1.json').read_text())

def test_unapproved_source_rejected_before_any_file_access(monkeypatch,config):
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('unauthorized file access'))
    with pytest.raises(ValueError,match='authorization'):source.launch_gate(config,'none','none')

@pytest.mark.parametrize('key,value',[('seeds',[123]),('variants',['M2']),('split_fingerprint','bad')])
def test_source_config_drift_rejected(config,key,value):
    config[key]=value
    with pytest.raises(ValueError):source.validate_config(config)

def test_masked_supervision_excludes_unknown_positions():
    model=source.Adapter('M2',{'mean_uV':0.,'std_uV':1.});logits=torch.randn(3,5,requires_grad=True);labels=torch.tensor([0,-1,2]);mask=labels>=0
    loss=model.loss(logits,labels,mask)
    torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(logits[mask],labels[mask]));loss.backward()
    assert not logits.grad[1].any()
    assert model.loss(logits,labels,torch.zeros(3,dtype=torch.bool))==0

@pytest.mark.parametrize('method',['M0','M1','M2','M3','M2_rms','M1_phase'])
def test_exact_resume_generated_source_interface(tmp_path,method):
    torch.set_num_threads(2);assert exact_resume(tmp_path/method,method)['model_optimizer_rng_history_exact']

def test_donor_compatibility_rejects_validation_before_array_access():
    class Trap:
        def __array__(self,*a,**k):pytest.fail('forbidden fitted data opened')
    with pytest.raises(ValueError):fit_compatibility(Trap(),Trap(),role='val',provenance={})
    with pytest.raises(ValueError):fit_compatibility(Trap(),Trap(),role='train',provenance={'scope':'REAL'})

def test_separate_train_compatibility_accepts_some_mock_donors():
    rng=np.random.default_rng(12);fit=rng.normal(0,10,(100,400));cal=rng.normal(0,10,(100,400))
    model=fit_compatibility(fit,cal,role='train',provenance={'scope':'SYNTHETIC_TRAIN_FIXTURE','fit_calibration_disjoint':True,'source_sha256':'a'*64})
    assert model.accepts(rng.normal(0,10,400)) and not model.accepts(rng.normal(0,1000,400))

def test_mock_gate_a_and_b_preserve_all_21_recipes_and_resume(tmp_path,monkeypatch,config):
    monkeypatch.chdir(tmp_path);proposal_path=tmp_path/'proposal.json';proposal_path.write_text('{}')
    train,val=mock_recordings();grid={'generated':True};proofs={'generated':True};proposal={'grid':grid,'source_proofs':proofs}
    monkeypatch.setattr(source,'launch_gate',lambda *a,**k:proposal)
    output=tmp_path/'mist_transfer_runs/mist_cove/phase4_mock';calls=[]
    def loader():return [train],[val],{'mean_uV':0.,'std_uV':1.},{s:synthetic_bank() for s in [123,456,789]},grid,proofs
    def trainer(model,tr,va,cfg,run,**kwargs):
        assert cfg.epochs==60 and cfg.patience==12 and cfg.core_epochs==cfg.encode_batch==16 and cfg.lr==.0003
        assert tr[0].scoring_mask.sum()==9 and len(va[0].indices)==10
        calls.append((cfg.seed,model.cfg,kwargs['resume'],kwargs['pause_after_epoch']))
        if kwargs['pause_after_epoch']:
            run.mkdir(parents=True);(run/'EPOCH_SEAL.json').write_text(json.dumps({'resources':{'seconds':1.},'checkpoint_write_seconds':.1}))
            raise source.EpochPaused()
        return 'MOCK_NOT_REAL'
    source.run_reviewed(config,proposal_path,None,output,authorized=True,gate='a',loader=loader,trainer=trainer)
    result=source.run_reviewed(config,proposal_path,None,output,authorized=True,gate='b',loader=loader,trainer=trainer)
    assert len(calls)==28 and len(result)==21
    assert all(v[2] for v in calls[7:14]) and all(not v[2] for v in calls[14:])

def test_incompatible_resume_rejected_before_checkpoint_load(tmp_path):
    from mist_cove.phase2.training import sealed_train
    from mist_evidence.runtime import TrainConfig
    tr,va=mock_recordings();model=source.Adapter('M2',{'mean_uV':0.,'std_uV':1.})
    cfg=TrainConfig(epochs=1,core_epochs=4,encode_batch=4);p=tmp_path/'run'
    sealed_train(model,[tr],[va],cfg,p,provenance={'scope':'MOCK'})
    with pytest.raises(ValueError,match='identity'):sealed_train(model,[tr],[va],TrainConfig(epochs=1,lr=.01),p,provenance={'scope':'MOCK'},resume=True)

"""Generated correctness fixtures, not confirmatory scientific comparisons."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from mist_cove.phase4.synthetic import generate
from mist_cove.phase4.model import make_model
from mist_cove.phase4.audit import decompose
from mist_cove.phase2.synthetic_runtime import assert_exact
from mist_cove.phase2.training import configure_precision
from mist_evidence.runtime import rng_state,seed_all
from mist_cove.phase5.signals import historical,harmonic,harmonic_component,diagnostic_swap,reverse_pair,relative_phase_oracle
from mist_cove.phase5.features import patches,zero_phase,restricted,transform_diagnostics
from mist_cove.phase5.models import WaveformControl,RestrictedProbe
from mist_cove.phase5.runtime import private_output,save_checkpoint,restore_checkpoint,wilson,historical_proofs,frozen_proofs
from mist_cove.phase5.experiment import evidence_stage


@pytest.mark.parametrize('task',['substitutable','complementary'])
@pytest.mark.parametrize('condition',['train','clean','competitive','overlap','erased','abrupt_artifact'])
def test_historical_replay_exact(task,condition):
    a=generate(10,901,task,condition);b=historical(10,901,task,condition)
    assert torch.equal(a.x,b.x) and torch.equal(a.y,b.y)
    assert (b.x-b.diagnostic-b.nuisance).abs().max()<2e-5
    for row in b.latent:
        assert all(0<=lo<hi<=3000 for lo,hi in row['intervals']+row['nuisance_intervals'])
        assert row['noise_stream_sample_offset']==row['generation_order']*3000


@pytest.mark.parametrize('family',['substitutable','complementary','harmonic'])
@pytest.mark.parametrize('condition',['train','clean','competitive','erased'])
def test_label_swap_preserves_all_nuisance(family,condition):
    data=harmonic(10,902,condition) if family=='harmonic' else historical(10,902,family,condition)
    other=diagnostic_swap(data,(data.y.numpy()+1)%5)
    assert torch.equal(data.nuisance,other.nuisance)
    for a,b in zip(data.latent,other.latent):
        excluded={'class_code','diagnostic_template_codes','diagnostic_phase'}
        assert {k:v for k,v in a.items() if k not in excluded}=={k:v for k,v in b.items() if k not in excluded}
    if condition=='erased':assert torch.equal(data.x,other.x)
    else:assert not torch.equal(data.diagnostic,other.diagnostic)


@pytest.mark.parametrize('condition',['train','clean','competitive','erased','artifact'])
def test_harmonic_determinism_and_geometry(condition):
    a=harmonic(10,903,condition);b=harmonic(10,903,condition)
    assert torch.equal(a.x,b.x) and a.latent==b.latent
    assert a.x.shape==(10,1,3000) and a.x.dtype==torch.float32 and torch.isfinite(a.x).all()
    assert torch.bincount(a.y).tolist()==[2]*5
    assert torch.equal(a.x,a.diagnostic+a.nuisance)
    assert all(row['intervals']==[[0,3000]] for row in a.latent)
    if condition!='erased': assert torch.equal(relative_phase_oracle(a.x),a.y)


@pytest.mark.parametrize('scope',['SYNTHETIC_ONLY','real','reserved','train','val'])
def test_scope_rejection_before_filesystem(tmp_path,monkeypatch,scope):
    import mist_cove.phase5.runtime as runtime
    if scope=='SYNTHETIC_ONLY':
        monkeypatch.setattr(runtime,'ROOT',tmp_path)
        out=tmp_path/'mist_transfer_runs/mist_cove/phase5_v1/test'
        assert private_output(out,scope)==out
        with pytest.raises(FileExistsError): private_output(out,scope)
        with pytest.raises(ValueError): private_output(tmp_path/'outside',scope)
    else:
        class Trap:
            def __fspath__(self):raise AssertionError('path accessed before role rejection')
        with pytest.raises(ValueError,match='before file access'):private_output(Trap(),scope)


@pytest.mark.parametrize('family',['substitutable','complementary','harmonic'])
@pytest.mark.parametrize('feature',['epoch_magnitude','patch_magnitude','amplitude','position','nuisance_magnitude','nuisance_latent','permuted_patch'])
def test_restricted_dimensions_fixed_across_conditions(family,feature):
    gen=harmonic if family=='harmonic' else lambda n,s,c:historical(n,s,family,c)
    a=restricted(gen(10,904,'train'),feature);b=restricted(gen(10,905,'competitive'),feature)
    assert a.ndim==2 and a.shape==b.shape and torch.isfinite(a).all()
    torch.manual_seed(41);model=RestrictedProbe(a)
    assert model(b).shape==(10,5)
    assert torch.equal(model.train_mean,a.mean(0))


def test_full_and_patch_noiseless_magnitudes_match():
    x=torch.stack([harmonic_component(code,.123,10.) for code in range(5)])[:,None]
    for signal in [x,patches(x)]:
        magnitude=torch.fft.rfft(signal.double()).abs()
        assert (magnitude-magnitude[:1]).abs().max()/magnitude.max()<1e-5
    assert torch.equal(relative_phase_oracle(x),torch.arange(5))


def test_overlap_inconsistency_and_magnitude_preservation():
    d=historical(10,906);r=transform_diagnostics(d.x)
    assert r['original_overlap_rms']==0 and r['coherent_zero_overlap_rms']==0
    assert r['patch_zero_overlap_rms']>1
    assert r['patch_magnitude_relative_max_error']<1e-5
    assert r['patch_zero_circular_even_rms']<1e-5
    assert torch.equal(patches(d.x)[:,0,200:],patches(d.x)[:,1,:200])


def test_zero_phase_conventions():
    t=torch.arange(3000);x=(torch.cos(2*torch.pi*2*t/100+.7)+3+(-1.)**t*.4)[None,None]
    z=zero_phase(x)
    assert torch.allclose(z.mean(-1),x.mean(-1),atol=1e-5)
    assert torch.allclose(torch.fft.rfft(z).abs(),torch.fft.rfft(x).abs(),atol=.001,rtol=1e-5)
    assert torch.allclose(z,torch.roll(z.flip(-1),1,-1),atol=1e-5)


@pytest.mark.parametrize('control',['original','patch_zero','coherent_zero'])
def test_shared_encoder_and_finite_backward(control):
    torch.manual_seed(42);m=WaveformControl(control)
    torch.manual_seed(42);ref=WaveformControl('original')
    assert_exact(m.state_dict(),ref.state_dict())
    x=harmonic(5,907).x
    value=m(x,return_evidence=True)
    assert value.logits.shape==(5,5) and torch.isfinite(value.logits.softmax(1)).all()
    assert torch.allclose(value.logits,value.background_logits+value.contributions.sum(1),atol=1e-5)
    value.logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.numel() for p in m.parameters())==3593


def test_historical_phase_transform_source_preserved():
    x=historical(5,908).x
    m=make_model('M1_phase');m.eval()
    with torch.no_grad():
        a=m.encode_candidates(x);b=m.encoder(zero_phase(patches(x)))
    assert_exact(a,b)


def test_reverse_pair_changes_no_diagnostic():
    a=harmonic(5,909);b=harmonic(5,910)
    x=reverse_pair(a,b)
    assert torch.equal(x,a.diagnostic+b.nuisance)
    assert not torch.equal(x,a.x)


@pytest.mark.parametrize('method',['M1','M2','M3'])
def test_counterfactual_accounting_and_intervals(method):
    torch.manual_seed(43);model=make_model(method).eval();d=harmonic(5,911)
    x=d.x[:1];changed=x.clone();changed[:,:,200:600]-=d.diagnostic[:1,:,200:600]
    report=decompose(model,x,changed,[(200,600)])
    assert max(abs(v) for v in report['residual'])<1e-5
    assert report['internal_paths_are_physiological'] is False
    e=model(x,return_evidence=True)
    for lo,hi in e.event_intervals[0,e.valid[0]]: assert lo%200==0 and hi-lo==400 and 0<=lo<hi<=3000


def test_v4_stops_without_constructing_a_model(monkeypatch,tmp_path):
    import mist_cove.phase5.experiment as module
    monkeypatch.setattr(module,'make_model',lambda *a,**k:(_ for _ in ()).throw(AssertionError('model created')))
    assert evidence_stage({'status':'FAIL'},None,None,tmp_path,None,[])=={'status':'NOT_TESTED_V3_FAILED','optimizer_updates':0}


def test_equivalence_not_null_hypothesis_nonrejection():
    lo,hi=wilson(400,2000)
    assert .15<lo<.2<hi<.25
    lo,hi=wilson(2,10)
    assert lo<.15 and hi>.25
    with pytest.raises(ValueError):wilson(12,10)


@pytest.mark.parametrize('control',['original','patch_zero','coherent_zero'])
def test_exact_checkpoint_optimizer_rng_recovery(tmp_path,control):
    configure_precision('cpu');seed_all(912,True)
    model=WaveformControl(control);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=harmonic(5,913).x;y=torch.arange(5)
    def step(m,o):
        # Consume all protected RNGs as well as checking deterministic updates.
        random.random();np.random.random();ix=torch.randperm(5)
        o.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x[ix]),y[ix]);loss.backward();o.step()
    step(model,optimizer)
    path=tmp_path/'synthetic.pt';identity={'scope':'SYNTHETIC_ONLY','transform':control}
    save_checkpoint(path,model,optimizer,identity)
    step(model,optimizer);expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),rng_state())
    other=WaveformControl(control);opt=torch.optim.AdamW(other.parameters(),lr=.001)
    restore_checkpoint(path,other,opt,identity);step(other,opt)
    assert_exact(expected,(other.state_dict(),opt.state_dict(),rng_state()))
    with pytest.raises(FileExistsError):save_checkpoint(path,other,opt,identity)
    with pytest.raises(ValueError):restore_checkpoint(path,other,opt,{'changed':True})
    with path.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='before deserialization'):restore_checkpoint(path,other,opt,identity)


def test_historical_and_preregistered_artifact_hashes():
    assert historical_proofs()['historical_implementation_sha256']=='e8f84002bd71a57259e596a547b101b5733e2d7a755b45bed706a830ff59142d'
    assert len(frozen_proofs()['files'])==5


@pytest.mark.parametrize('family',['substitutable','complementary','harmonic'])
def test_exploratory_runner_plumbing(tmp_path,family,capsys):
    # Deliberately different seeds and one update; not a scientific outcome.
    from mist_cove.phase5.experiment import suite
    from mist_cove.phase5.runtime import Budget
    c=json.loads(Path('configs/mist_cove_phase5_identifiability_v1.json').read_text())
    c.update(seeds=[914],train_examples=10,eval_examples=10,pair_examples=10,updates=1,bootstrap_replicates=2)
    c['historical_probes']=['isolated_magnitude','position','nuisance_latent','patch_magnitude']
    c['new_probes']=['position','nuisance_latent','patch_magnitude']
    rows=[];diagnostics=[];common={}
    suite(family,c,tmp_path,Budget(30),rows,diagnostics,common)
    assert len(rows)==3+len(c['new_probes'] if family=='harmonic' else c['historical_probes'])
    assert all(r['training']['optimizer_updates']==1 for r in rows)
    assert all(np.isfinite(r['conditions']['clean']['nll']) for r in rows)
    assert rows[0]['witness_audit']['max_reconciliation_residual']<1e-5
    assert rows[0]['pairing']['paired_nuisance_exact']
    capsys.readouterr()


@pytest.mark.parametrize('control',['original','patch_zero','coherent_zero'])
def test_generated_recovery_helper(tmp_path,control):
    from mist_cove.phase5.correctness import exact_resume
    assert exact_resume(tmp_path,control)['model_optimizer_python_numpy_cpu_cuda_rng_exact']

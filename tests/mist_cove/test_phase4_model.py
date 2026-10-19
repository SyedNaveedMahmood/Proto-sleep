import pytest
import torch
from mist_cove.phase4.model import make_model,MorphConfig
from mist_cove.phase2.model import WitnessSet
from mist_cove.phase2.traces import EpochIdentity,source_ledger
from mist_cove.phase4.audit import decompose,group_report
from mist_cove.phase4.synthetic import templates,generate,localization

@pytest.mark.parametrize('method',['M0','M1','M2','M3','M2_rms','M1_phase'])
def test_finite_five_logits_additive_and_backward(method):
    torch.set_num_threads(2);torch.manual_seed(11)
    model=make_model(method);x=torch.randn(2,1,3000)
    ev=model(x,return_evidence=True)
    assert ev.logits.shape==(2,5) and torch.isfinite(ev.logits).all()
    assert torch.equal(ev.logits,ev.background_logits+ev.contributions.sum(1))
    torch.testing.assert_close(ev.logits.softmax(-1).sum(-1),torch.ones(2))
    ev.logits.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_equal_capacity_and_identical_encoder_initialization():
    models=[]
    for name in ['M1','M2','M2_rms','M3','M1_phase']:
        torch.manual_seed(14);models.append(make_model(name))
    assert sum(p.numel() for p in models[0].parameters())==3593
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    assert sum(p.numel() for p in models[3].parameters() if p.requires_grad)==3503
    for m in models[1:]:
        assert all(torch.equal(v,m.encoder.state_dict()[k]) for k,v in models[0].encoder.state_dict().items())

def test_historical_q1_factory_exactly_unchanged():
    from mist_cove.phase2.model import make_model as old
    torch.manual_seed(18);a=make_model('M0')
    torch.manual_seed(18);b=old('q1')
    x=torch.randn(2,1,3000)
    assert torch.equal(a(x),b(x)) and sum(p.numel() for p in a.parameters())==75

def test_phase_control_retains_patch_mean_rms_and_full_spectrum():
    m=make_model('M1_phase');x=torch.randn(1,1,3000);seen=[]
    handle=m.encoder.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
    m(x);handle.remove();original=x[:,0].unfold(-1,400,200);changed=seen[0]
    torch.testing.assert_close(original.mean(-1),changed.mean(-1),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(torch.fft.rfft(original).abs(),torch.fft.rfft(changed).abs(),atol=2e-5,rtol=1e-4)

def test_contrast_ranking_can_ignore_stronger_irrelevant_amplitude():
    m=make_model('M2');r=make_model('M2_rms');r.load_state_dict(m.state_dict())
    with torch.no_grad():
        m.event_head.weight.zero_();m.event_head.weight[0,0]=1;r.event_head.weight.copy_(m.event_head.weight)
    features=torch.zeros(1,14,18);features[0,4,0]=2
    rms=torch.ones(1,14);rms[0,0]=1000
    assert int(m.decision(features,rms)[4][0,0])==4
    assert int(r.decision(features,rms)[4][0,0])==0

def test_calibrated_inverse_and_source_interval_ledger():
    m=make_model('M2',mean_uV=2.,std_uV=7.).eval();physical=torch.randn(1,1,3000)*10
    ev=m((physical-2)/7,return_evidence=True)
    torch.testing.assert_close(m.physical_uV((physical-2)/7),physical)
    assert ev.local_representations.shape==(1,4,18)
    assert ((ev.event_intervals[ev.valid][:,1]-ev.event_intervals[ev.valid][:,0])==400).all()
    identity=EpochIdentity('generated_person','generated_record','1','val',7,'a'*64,ev.physical_input_sha256[0],False)
    ledger=source_ledger(ev,[identity]);assert len(ledger)==4
    assert all(v['absolute_start_sample']==21000+v['start_sample'] and v['units']=='uV' and not v['scoring_mask'] for v in ledger)

def test_local_features_depend_only_on_declared_interval():
    torch.manual_seed(18);m=make_model('M2').eval();x=torch.randn(1,1,3000);changed=x.clone();changed[...,400:]+=100
    a=m.encode_candidates(x)[0];b=m.encode_candidates(changed)[0]
    assert torch.equal(a[:,0],b[:,0]) and not torch.equal(a[:,1:],b[:,1:])

@pytest.mark.parametrize('method',['M1','M2','M3','M2_rms','M1_phase'])
def test_flat_input_no_fabricated_events(method):
    ev=make_model(method)(torch.zeros(1,1,3000),return_evidence=True)
    assert not ev.valid.any() and not ev.contributions.any() and torch.isfinite(ev.logits).all()

def test_fixed_overlap_validity_and_exact_counterfactual_paths():
    torch.manual_seed(8);m=make_model('M2').eval();x=torch.randn(1,1,3000)
    ev=m(x,return_evidence=True);fixed=WitnessSet.from_evidence(ev)
    changed=x.clone();changed[...,200:800]=0
    a=decompose(m,x,changed,[(200,800)])
    assert max(abs(v) for v in a['residual'])<1e-5
    assert torch.equal(m(changed,return_evidence=True,witnesses=fixed).event_intervals,ev.event_intervals)
    fixed.intervals[0,1]=fixed.intervals[0,0]
    with pytest.raises(ValueError,match='duplicate'):m(x,witnesses=fixed)

@pytest.mark.parametrize('shape',[(1,2,3000),(1,1,2999),(1,3000)])
def test_wrong_input_geometry_rejected(shape):
    with pytest.raises(ValueError):make_model('M2')(torch.zeros(shape))

def test_generated_equal_spectrum_and_no_clinical_annotation():
    bank=templates();power=torch.fft.rfft(bank).abs()
    torch.testing.assert_close(power,power[0].expand_as(power),atol=2e-5,rtol=2e-5)
    a=generate(10,9);b=generate(10,9)
    assert torch.equal(a.x,b.x) and a.condition=='clean'
    assert not torch.equal(bank[0],bank[1])
    assert all(0<=p['interval'][0]<p['interval'][1]<=3000 for parts in a.components for p in parts)

def test_localization_ground_truth_and_overlap():
    v=localization([(100,500),(300,700)],[(200,400)])
    assert v['top1_hit']==1 and v['diagnostic_recall']==1 and v['selected_overlap_fraction']==1

def test_complete_oracle_erasure_and_donor_paths():
    data=generate(5,42);m=make_model('M2').eval();parts=data.components[0]
    intervals=[p['interval'] for p in parts if p['diagnostic']]
    report=group_report(m,data.x[:1],intervals,parts,seed=15)
    assert len(report['donors'])==3 and all(len(d['full'])==5 for d in report['donors'])
    assert all(max(abs(v) for v in d['residual'])<1e-5 for d in report['donors'])

def test_frozen_head_training_does_not_change_encoder():
    from mist_cove.phase4.experiment import train_model
    torch.manual_seed(13);m=make_model('M2');m.encoder.requires_grad_(False)
    before={k:v.clone() for k,v in m.encoder.state_dict().items()};batch=generate(10,12)
    with torch.no_grad():f=m.encode_candidates(batch.x)
    train_model(m,batch,{'learning_rate':.001,'weight_decay':.0001,'batch_size':5,'gradient_clip':5.,'updates':2},13,features=f)
    assert all(torch.equal(v,m.encoder.state_dict()[k]) for k,v in before.items())

@pytest.mark.parametrize('method',['M0','M1','M2','M3','M2_rms','M1_phase'])
def test_generated_evaluation_and_audit_plumbing(method):
    import json
    from mist_cove.phase4.experiment import evaluate,audit_examples
    torch.manual_seed(99);m=make_model(method);data=generate(5,99991,'complementary','overlap')
    value={'evaluation':evaluate(m,data),'audit':audit_examples(m,data,99)}
    assert value['audit']['n_examples']==5
    json.dumps(value,allow_nan=False)

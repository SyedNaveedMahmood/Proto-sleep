"""Generated-only Phase XV leakage, epoch geometry, and sampling fixtures."""
import numpy as np
import pytest

from mist_cove.phase15.forensics import _distances, _prob, _round_robin, _stratum, _event_present
from mist_cove.phase15.wave_diagnostics import _wave_features, _require_st_rate
from mist_cove.phase15.explanation import _align
from mist_cove.phase15.comparison import _ci_person, _eligible
from mist_cove.phase14.event_alignment import matched_candidates
from mist_cove.phase14.edf_reader import MultiReader
from mist_cove.phase9.operators import atom_to_candidate_mask
from mist_cove.phase9.clauses import verify_clause
from mist_cove.phase15.governance import canonical, sha
import torch


def test_physical_positions_are_not_compressed():
    y=np.array([-1,4,4,-1,0,0,4])
    d=_distances(y)
    assert len(d)==len(y)
    assert d[4]==2  # Unscored predecessor is not an invented transition.
    assert d[6]==0
    assert _distances(np.array([4,4,-1,0,4]))[0]==5


def test_panel_seed_does_not_use_model_response_or_labels():
    rows=[{'person':'p','night':1,'recording':'r','epoch':i,'truth':4,'pred':0} for i in range(30)]
    first=[r['epoch'] for r in _round_robin(rows,8,15011)]
    changed=[{**r,'truth':0,'pred':4} for r in rows]
    second=[r['epoch'] for r in _round_robin(changed,8,15011)]
    assert first==second
    assert first==[r['epoch'] for r in _round_robin(rows,8,15011)]


def test_rem_strata_and_softmax_are_exact():
    assert _stratum({'truth':4,'pred':0})=='REM_to_W'
    assert _stratum({'truth':4,'pred':4})=='correct_REM'
    assert _stratum({'truth':2,'pred':2})=='correct_N2'
    assert _stratum({'truth':0,'pred':1})=='non_REM_error'
    p=_prob(np.array([[1001.,1000.,-100.]]))
    assert np.isfinite(p).all()
    assert p.sum()==pytest.approx(1.)
    assert p[0,0]>p[0,1]


def test_generated_channel_spectral_measure_has_known_peak():
    t=np.arange(3000)/100
    x=np.sin(2*np.pi*13*t)[None,:]
    f=_wave_features(x)
    assert f['sigma_fraction'][0]>.99
    assert f['delta_fraction'][0]<1e-6
    assert f['RMS_uV'][0]==pytest.approx(2**-.5,rel=1e-4)


def test_proxy_comparators_have_equal_atom_budget_without_labels():
    t=np.arange(3000)/100
    wave=np.sin(2*np.pi*13*t)
    c=matched_candidates(wave,0b10101,0b111, list(range(15)),15011)
    assert all(c[k].bit_count()==3 for k in ('exact_positive','RMS_matched',
        'gradient_matched','random_matched','position_matched','sigma_matched','delta_matched'))
    out=_align({'EEG Pz-Oz|spindle':[]},0,c)
    assert out['EEG Pz-Oz|spindle']['event_count']==0


def test_participant_bootstrap_keeps_nested_cases_together():
    r=_ci_person([('a',1),('a',3),('b',-2)],15015)
    assert r['people']==2 and r['cases']==3
    assert r['person_mean_delta']==pytest.approx(0.)


def test_no_event_is_explicitly_ineligible():
    row={'summary':{'positive_minimum_mask':3},'proxy_alignment':
         {'EEG Pz-Oz|spindle':{'event_count':0}}}
    assert _eligible(row)==(False,'NO_PZ_SPINDLE_CANDIDATE')


def test_stage_blind_event_epoch_geometry():
    e={'Pz':[{'start':29.9,'end':30.1}]}
    assert _event_present(e,0,'Pz') and _event_present(e,1,'Pz')
    assert not _event_present(e,2,'Pz')


def test_unauthorized_role_rejected_before_source_io():
    reader=MultiReader.__new__(MultiReader)
    reader.manifest={'records':[]}
    with pytest.raises(ValueError):_require_st_rate(1)
    with pytest.raises(Exception):reader.records('SC_VAL')
    with pytest.raises(Exception):reader.records('SC_RESERVED')


def test_invalid_clause_mask_and_coordinate_reordering_fail():
    with pytest.raises(ValueError):atom_to_candidate_mask(torch.ones(1,14,dtype=torch.bool))
    truth=np.zeros(1<<15,dtype=bool);truth[-1]=True
    assert not verify_clause(truth,0,0)
    a={'coordinates':[['r',1],['r',2]],'panel_sha256':'x'}
    b={'coordinates':[['r',2],['r',1]],'panel_sha256':'x'}
    assert sha(canonical(a))!=sha(canonical(b))

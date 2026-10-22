"""Generated-only Phase XV leakage, epoch geometry, and sampling fixtures."""
import numpy as np
import pytest

from mist_cove.phase15.forensics import _distances, _prob, _round_robin, _stratum
from mist_cove.phase15.wave_diagnostics import _wave_features
from mist_cove.phase15.explanation import _align
from mist_cove.phase14.event_alignment import matched_candidates


def test_physical_positions_are_not_compressed():
    y=np.array([-1,4,4,-1,0,0,4])
    d=_distances(y)
    assert len(d)==len(y)
    assert d[4]==2  # Unscored predecessor is not an invented transition.
    assert d[6]==0


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

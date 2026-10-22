"""Generated-only Phase XV leakage, epoch geometry, and sampling fixtures."""
import numpy as np
import pytest

from mist_cove.phase15.forensics import _distances, _prob, _round_robin, _stratum


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

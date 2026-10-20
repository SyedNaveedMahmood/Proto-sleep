import numpy as np
import pytest
import torch
from mist_cove.phase10.mock_controls import ControlAdapter
from mist_cove.phase10.response_circuits import response_table,summarize_response
from mist_cove.phase9.circuits import compile_robdd,evaluate_robdd
from mist_cove.phase9.clauses import verify_clause


@pytest.mark.parametrize('name',['redundant','complementary','disruptor','whole_epoch','eventless'])
def test_all_32768_source_geometry_control_rows(name):
    a=ControlAdapter(name);table=response_table(a,a.cache(a.waveform()));summary=summarize_response(table)
    ids=np.arange(32768);aa=(ids&(1<<2))!=0;bb=(ids&(1<<10))!=0;cc=(ids&(1<<6))!=0
    expected={'redundant':aa|bb,'complementary':aa&bb,'disruptor':(aa&bb)|((aa|bb)&~cc),
              'whole_epoch':ids==32767,'eventless':np.ones(32768,dtype=bool)}[name]
    assert np.array_equal(table['truth'],expected) and summary['exact_all_rows']
    for cl in summary['signed']['clauses']:
        p,n=cl['positive_bits'],cl['negative_bits'];assert verify_clause(expected,p,n)
        for j in range(15):
            if p&(1<<j):assert not verify_clause(expected,p^(1<<j),n)
            if n&(1<<j):assert not verify_clause(expected,p,n^(1<<j))
    if name=='disruptor':
        assert summary['nonmonotone_edges']['response_lost_when_adding_atom']>0
        assert any(6 in c['negative'] for c in summary['signed']['clauses'])
    if name=='eventless':assert summary['signed']['prior_only']
    if name=='whole_epoch':assert summary['signed']['clauses'][0]['retained_seconds']==30


def test_xor_and_unsupported_exact_controls():
    ids=np.arange(32768);truth=((ids&1)>0)^((ids&2)>0)
    assert np.array_equal(evaluate_robdd(compile_robdd(truth)),truth)
    summary=summarize_response({'truth':np.zeros(32768,dtype=bool),'margins':np.full(32768,-1.),'target':0})
    assert summary['signed']['status']=='NO_SUPPORT' and not summary['full_response']

import numpy as np
import pytest
from mist_cove.phase9.circuits import compile_robdd,evaluate_robdd,nonmonotone_edges
from mist_cove.phase9.clauses import verify_clause,minimize_clause,extract_readable_clauses,greedy_sufficient_mask

@pytest.mark.parametrize('kind',['or','and','blocker','mixed','xor','true','false'])
def test_exact_all_32768_masks_and_verified_minimal_clauses(kind):
    ids=np.arange(32768);a=(ids&1)>0;b=(ids&2)>0;c=(ids&4)>0;d=(ids&8)>0;e=(ids&16)>0
    truth={'or':a|b,'and':a&b,'blocker':(a|b)&~c,'mixed':(a&b)|(d&~e),'xor':a^b,'true':np.ones(32768,dtype=bool),'false':np.zeros(32768,dtype=bool)}[kind]
    circuit=compile_robdd(truth);assert np.array_equal(evaluate_robdd(circuit),truth)
    summary=extract_readable_clauses(truth)
    for clause in summary['clauses']:
        p,q=clause['positive_bits'],clause['negative_bits'];assert verify_clause(truth,p,q)
        for j in range(15):
            if p&(1<<j):assert not verify_clause(truth,p^(1<<j),q)
            if q&(1<<j):assert not verify_clause(truth,p,q^(1<<j))
        assert clause['minimum_cardinality'] is False
    if kind=='true':assert summary['prior_only'] and not summary['clauses'][0]['waveform_support_claim']
    if kind=='false':assert summary['status']=='NO_SUPPORT'
    if kind in ('xor','blocker'):assert nonmonotone_edges(truth)['response_lost_when_adding_atom']>0


def test_signed_blocker_and_greedy_counterexample():
    ids=np.arange(8);truth=((ids&1)>0)&((ids&4)==0)
    assert truth[1] and not verify_clause(truth,1,0) and verify_clause(truth,1,4)
    summary=extract_readable_clauses(truth);clause=summary['clauses'][0]
    assert clause['counterfactual_state_only'] and not clause['active_at_original_full_mask']
    assert not extract_readable_clauses(truth,positive_only=True)['clauses']
    # Greedy canonical sufficiency need not imply universal retained-set validity.
    truth=np.array([False,True,True,False,True,False,True,True])
    result=greedy_sufficient_mask(truth)
    assert result['supported'] and not result['universal_positive_valid']


def test_partial_summary_finite_views_and_unsupported_inputs():
    ids=np.arange(32768);truth=(ids%2==1)
    views=np.stack((truth,~truth));assert not verify_clause(views,1,0)
    assert extract_readable_clauses(views)['status']=='NO_SUPPORT'
    with pytest.raises(ValueError):verify_clause(truth,1,1)
    with pytest.raises(ValueError):compile_robdd(np.ones(30,dtype=bool))
    with pytest.raises(ValueError):compile_robdd(truth,list(reversed(range(15))))
    with pytest.raises(ValueError):minimize_clause(truth,0,1)

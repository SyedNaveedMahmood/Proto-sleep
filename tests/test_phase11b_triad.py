import numpy as np
import pytest
from mist_cove.phase11b.triad import exact_support,waveform_probes,distortion,replacements,fixed_clause_stability,triangulation_graph

@pytest.mark.parametrize('kind',['or','and','xor','interference','prior','none'])
def test_exact_support_nonmonotone(kind):
    ids=np.arange(32768);a=(ids&1)>0;b=(ids&2)>0
    truth={'or':a|b,'and':a&b,'xor':a^b,'interference':a&~b,'prior':np.ones(32768,bool),'none':np.zeros(32768,bool)}[kind]
    result=exact_support(truth)
    expected={'or':(1,1),'and':(3,3),'xor':(1,None),'interference':(1,None),'prior':(0,0),'none':(None,None)}[kind]
    assert (result['canonical_minimum_mask'],result['positive_minimum_mask'])==expected

def test_probe_identity_phase_and_local_replacement():
    x=np.sin(np.arange(3000)*.3).astype(np.float32);p=waveform_probes(x,123)
    assert np.array_equal(p['identity'],x)
    np.testing.assert_allclose(abs(np.fft.rfft(p['phase'])),abs(np.fft.rfft(x)),atol=1e-4)
    values,rows=replacements(x)
    assert len(rows)==30 and all(r['nontarget_exact'] for r in rows)
    assert distortion(x,x)['relative_energy_distortion']==0
    assert not distortion(x,x*10)['historical_compatibility']

def test_counterfactual_clause_cannot_substitute_at_full():
    truth=(np.arange(32768)&2)==0
    summary={'signed':{'clauses':[{'positive_bits':1,'negative_bits':0},{'positive_bits':0,'negative_bits':2}]}}
    row=fixed_clause_stability(summary,truth,1,0)
    assert row['any_original_clause_valid'] and not row['active_substitution']
    assert not row['any_original_active_clause_valid'] and not row['prediction_stable']

def test_graph_excludes_rejected_interventions():
    truth=np.ones(32768,bool)
    rows=[{'status':'ATTEMPTED','historical_compatibility':False,'atom':0,'kind':'bridge','target_retained':False}]
    graph=triangulation_graph({'signed':{'clauses':[]}},truth,rows,{},truth,.9)
    assert graph['compatible_comparisons']==0 and graph['operator_disagreements']==0

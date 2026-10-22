import copy
import hashlib
import json
import numpy as np
import pytest
from mist_cove.phase13.statistics import clustered_interval
from mist_cove.phase13.annotation_interface import ReviewStore
from mist_cove.phase13.sampling import primary_panel


def test_cluster_bootstrap_keeps_optimization_seeds_inside_people():
    a=np.array([[.1,.2,.3,.4,.5],[.11,.21,.31,.41,.51],[.12,.22,.32,.42,.52]])
    first=clustered_interval(a);again=clustered_interval(a)
    assert first==again and first['people']==5 and first['seeds']==3
    assert first['mean']==pytest.approx(.31)
    assert first['participant_cluster_bootstrap_95_percent'][0]<.31<first['participant_cluster_bootstrap_95_percent'][1]
    with pytest.raises(ValueError):clustered_interval([1,2,3])


def test_primary_next_two_ranks_and_missing_strata(monkeypatch):
    from mist_cove.phase13 import sampling
    coords=[{'subject':p,'stage':s,'epoch':j} for p in ('generated_A','generated_B') for s in range(5) for j in range(12 if s<4 else 10)]
    monkeypatch.setattr(sampling,'select_epochs',lambda records,n:(coords,{}))
    panel,missing=primary_panel([])
    assert len(panel)==16 and missing==134
    assert all(c['epoch'] in (10,11) for c in panel)
    assert not any(c['stage']==4 for c in panel)


def test_annotation_replay_rejects_packet_protocol_substitution(tmp_path):
    reviewers={'r':{'role':'reviewer','token_sha256':hashlib.sha256(b'token').hexdigest()}}
    cases={'a':{'master':{'case_id':'a'},'wave_uV':np.zeros(3000).tolist()}}
    s=ReviewStore(cases,reviewers,tmp_path,protocol_identity={'codebook':'one'})
    s.append('r','lock',{})
    with pytest.raises(ValueError):ReviewStore(cases,reviewers,tmp_path,protocol_identity={'codebook':'changed'})
    changed=copy.deepcopy(cases);changed['a']['wave_uV'][0]=1
    with pytest.raises(ValueError):ReviewStore(changed,reviewers,tmp_path,protocol_identity={'codebook':'one'})
    restored=ReviewStore(cases,reviewers,tmp_path,protocol_identity={'codebook':'one'})
    assert 'r' in restored.locked


def test_original_gradient_comparator_explicitly_enables_gradients():
    import torch
    from mist_cove.phase10.mock_models import make_mock,mock_input
    from mist_cove.phase10.source_adapter import SourceAdapter
    from mist_cove.phase13.mechanistic import analyze_case,Budget
    m=make_mock().network;op=SourceAdapter(m);x=mock_input(n=1)
    analyze_case.normalization={'mean_uV':0.,'std_uV':1.}
    row,tables=analyze_case(op,x,x.numpy()[0,0],2,{},Budget(),physical=False)
    assert row['summary']['exact_all_rows'] and tables.shape==(1,32768)


def test_locked_alignment_contract_on_generated_fixture():
    from mist_cove.phase13.evidence_alignment import align_locked_panel,candidate_regions
    events={'case':{'assessments':[{'category':'spindle_like','state':'present','intervals':[[0,200]],'confidence':2}]}}
    # Synthetic dictionaries exercise the contract, not real human evidence.
    exports=[{'reviewer':str(i),'locked':True,'qualified_approved':True,'domain':'REAL_TRAIN','packet_sha256':'generated_packet','assessments':events} for i in (0,1)]
    model={'alignment_candidates':candidate_regions(np.ones(3000),1,1,list(range(15))),
           'correct':False,'summary':{'signed':{'clauses':[]}},'branch_predictions':{'local_only':1,'global_only':2}}
    result=align_locked_panel(exports,{'case':model},{'case':'generated_person'},['case'])
    assert len(result['reviewers'])==2 and not result['automatic_adjudication']
    assert result['paired_event_intersections'][0]['intersection_seconds']==2
    assert not result['reviewers'][0]['rows'][0]['correct']


def test_safe_aggregate_rejects_private_waveforms_fitted_values_and_people():
    from mist_cove.phase13.publication import safe_aggregate
    assert safe_aggregate({'participants':15,'human_annotation':'NOT_TESTED','macro_F1':[.4,.5]})
    for bad in ({'coordinate':{'epoch':1}},{'wave_uV':[0.]},{'normalization':{'std_uV':2.}}, {'x':'SC9999E0'}):
        with pytest.raises(ValueError):safe_aggregate(bad)

import hashlib
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pytest
from mist_cove.phase13.annotation_schema import seconds_to_samples,validate_assessment,blind_case
from mist_cove.phase13.annotation_metrics import overlap_metrics,event_agreement,categorical_agreement,controls,review_quality
from mist_cove.phase13.evidence_alignment import evaluate_locked,candidate_regions
from mist_cove.phase13.annotation_interface import ReviewStore,handler_for


def assessment(state='present',intervals=None):
    return {'category':'spindle_like','state':state,'intervals':[[100,200]] if intervals is None else intervals,'confidence':2}

def test_coordinate_conversion_and_assessability():
    assert seconds_to_samples(.125,1.25)==[12,125]
    for bounds in [(-.1,1),(1,31),(float('nan'),1),(1,1)]:
        with pytest.raises(ValueError):seconds_to_samples(*bounds)
    assert validate_assessment(assessment('not_assessable',[]))['state']=='not_assessable'
    with pytest.raises(ValueError):validate_assessment(assessment('absent'))

def test_blinding_drops_all_coordinator_outcomes():
    value=blind_case({'case_id':'opaque','stage':1,'prediction':2,'participant':'generated_secret','duplicate_of':'other'},np.zeros(3000))
    assert not set(value)&{'stage','prediction','participant','duplicate_of'}
    assert len(value['wave_uV'])==3000 and value['grid_uV']==50

def test_overlap_tolerances_and_missing_denominators():
    result=overlap_metrics([[100,200]],[[150,250]])
    assert result['precision']==.5 and result['event_coverage']==.5 and result['overlap_seconds']==.5
    assert overlap_metrics([],[])['precision'] is None
    assert overlap_metrics([[100,200]],[[210,300]],.1)['overlap_seconds']==0
    assert overlap_metrics([[100,200]],[[210,300]],.25)['overlap_seconds']>.0

def test_event_matching_kappa_and_repeatability():
    assert event_agreement([[0,100]],[[10,110]])['F1']==1
    assert event_agreement([],[])['F1'] is None
    assert categorical_agreement(['p','p','a','a'],['p','a','p','a'])['kappa']==pytest.approx(0)
    export={'assessments':{'a':{'assessments':[assessment()]},'b':{'assessments':[assessment()]}}}
    qa=review_quality(export,export,['a','b','missing'],[('a','b')])
    assert qa['first_completion']==pytest.approx(2/3) and all(v['identical_assessments'] for v in qa['duplicates'])

def test_matched_controls_and_absent_expert_gate():
    source=(1<<0)|(1<<2)|(1<<8)
    c=controls(source);assert c['length_matched_mask'].bit_count()==3 and c['position_matched_mask'].bit_count()==3
    result=candidate_regions(np.sin(np.arange(3000)),source,3,list(range(15)))
    assert result['RMS_matched_budget'].bit_count()==3 and not result['sigma_is_clinical_event_label']
    with pytest.raises(ValueError):evaluate_locked([],source,[[0,100]])

def store(tmp_path,domain='GENERATED',qualified=False):
    token='generated-test-token';reviewers={'r_demo':{'role':'reviewer','token_sha256':hashlib.sha256(token.encode()).hexdigest(),'qualified_approved':qualified}}
    cases={'opaque':{'master':{'case_id':'opaque','stage':3,'prediction':0},'wave_uV':np.sin(np.arange(3000)).tolist()}}
    return ReviewStore(cases,reviewers,tmp_path,domain),token

def test_real_review_cannot_use_generated_or_unapproved_reviewer(tmp_path):
    with pytest.raises(PermissionError):store(tmp_path,'REAL_TRAIN')
    with pytest.raises(PermissionError):ReviewStore({}, {},tmp_path,'REAL_TRAIN')

def test_locked_export_and_chain_replay(tmp_path):
    s,token=store(tmp_path);alias=s.authenticate(token)
    s.append(alias,'assessment',{'case_id':'opaque','assessments':[assessment()],'notes':'generated'})
    s.append(alias,'lock',{})
    export=json.loads((tmp_path/'r_demo_locked.json').read_text());assert export['locked'] and export['domain']=='GENERATED'
    with pytest.raises(PermissionError):s.append(alias,'assessment',{})
    restored,_=store(tmp_path)
    assert alias in restored.locked
    with pytest.raises(ValueError):evaluate_locked([export,export],1,[[0,100]])

def test_http_local_credentials_blinding_and_no_master_routes(tmp_path):
    s,token=store(tmp_path);server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(s));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(HTTPError):urlopen(base+'/api/cases')
        request=Request(base+'/api/case/opaque',headers={'Authorization':'Bearer '+token})
        with urlopen(request) as response:
            value=json.loads(response.read());assert 'stage' not in value and 'prediction' not in value
            assert "connect-src 'self'" in response.headers['Content-Security-Policy']
        for path in ['/api/master','/../../private']:
            with pytest.raises(HTTPError):urlopen(Request(base+path,headers={'Authorization':'Bearer '+token}))
        with pytest.raises(HTTPError):urlopen(Request(base+'/api/cases',headers={'Authorization':'Bearer '+token,'Origin':'https://unapproved.example'}))
    finally:server.shutdown();server.server_close();thread.join()

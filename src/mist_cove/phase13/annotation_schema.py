"""Observable morphology candidates, never automatic clinical ground truth."""
import math
import re

CATEGORIES = ('spindle_like','k_complex_like','slow_wave_like','alpha_like','theta_like',
              'sharp_transient','broad_spectral_pattern','possible_artifact','mixed_pattern')
STATES = ('present','absent','uncertain','not_assessable','possible_artifact','insufficient_signal_quality')


def seconds_to_samples(start, stop):
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (start,stop)):
        raise ValueError('finite interval seconds required')
    a,b = round(start*100),round(stop*100)
    if not 0 <= a < b <= 3000:raise ValueError('half-open original epoch interval required')
    return [a,b]


def validate_assessment(value):
    allowed={'category','state','intervals','confidence','artifact_flag','quality_flag'}
    if set(value)-allowed:raise ValueError('unknown annotation field')
    if value.get('category') not in CATEGORIES or value.get('state') not in STATES:
        raise ValueError('codebook category/state required')
    if type(value.get('confidence')) is not int or not 0<=value['confidence']<=3:
        raise ValueError('confidence ordinal0..3 required')
    intervals=value.get('intervals',[])
    for pair in intervals:
        if len(pair)!=2 or any(type(i) is not int for i in pair) or not 0<=pair[0]<pair[1]<=3000:
            raise ValueError('bounded physical sample coordinates required')
    if value['state']=='present' and not intervals:raise ValueError('present observations need localized intervals')
    if value['state'] in ('absent','not_assessable','insufficient_signal_quality') and intervals:
        raise ValueError('non-observation status cannot carry positive intervals')
    if len(intervals)>100:raise ValueError('bounded annotation size')
    return {**value,'intervals':intervals}


def validate_submission(value, known_cases):
    if set(value)-{'case_id','assessments','notes','display'}:raise ValueError('unknown submission fields')
    if value.get('case_id') not in known_cases:raise ValueError('unknown blinded case')
    if not isinstance(value.get('notes',''),str) or len(value.get('notes',''))>1000:raise ValueError('bounded notes')
    assessments=value.get('assessments')
    if not isinstance(assessments,list) or not 1<=len(assessments)<=100:raise ValueError('explicit assessments required')
    if len({a.get('category') for a in assessments})!=len(assessments):raise ValueError('one category assessment with multiple intervals required')
    return {**value,'assessments':[validate_assessment(v) for v in assessments]}


def blind_case(master, wave, context=None):
    # An explicit allowlist: never forward arbitrary coordinator metadata.
    return {'case_id':master['case_id'],'wave_uV':list(map(float,wave)),
            'fs':100,'samples':3000,'duration_seconds':30,'categories':list(CATEGORIES),
            'states':list(STATES),'grid_uV':50,'context_uV':context}

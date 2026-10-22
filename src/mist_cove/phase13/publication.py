"""Conservative aggregate release checks; no private artifacts are admitted."""
import re

FORBIDDEN_KEYS={'subject_metrics','subject_macro_f1','recording_id','subject_id','physical_epochs','coordinate',
    'coordinates','wave_uV','context_uV','mean_uV','std_uV','assessments','notes','token_sha256','reviewer_identity'}


def safe_aggregate(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN_KEYS:raise ValueError('private data-bearing publication field')
        for key,item in value.items():safe_aggregate(item)
    elif isinstance(value,list):
        for item in value:safe_aggregate(item)
    elif isinstance(value,str):
        if '/home/' in value or re.search(r'\bSC\d{3,}',value):raise ValueError('private source identifier/path')
    return True

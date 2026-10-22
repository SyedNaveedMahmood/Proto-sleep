"""Private randomization key, optional local service, and truthful readiness."""
import json
import secrets
import numpy as np
from mist_cove.phase11b.security import sha256
from .provenance import PUBLIC,PRIVATE,canonical,write_new,authenticated_json_identity
from .sampling import frozen_selection
from .data import load_cache
from .resources import measured


def prepare_packets():
    with measured('annotation_packets','A'):
        selection=frozen_selection();primary=selection['primary'];secondary=selection['secondary']
        cases=[]
        for index,_ in enumerate(primary+secondary):
            cases.append({'case_id':'case_'+secrets.token_hex(8),'source_index':index,
                          'group':'primary' if index<len(primary) else 'secondary','duplicate_of':None})
        duplicates=0
        for ordinal,subject in enumerate(sorted({c['subject'] for c in primary})):
            candidates=[i for i,c in enumerate(primary) if c['subject']==subject and c['stage']==ordinal%5]
            if candidates:
                i=candidates[0];cases.append({'case_id':'case_'+secrets.token_hex(8),'source_index':i,
                                            'group':'duplicate','duplicate_of':cases[i]['case_id']});duplicates+=1
        order=np.random.default_rng(13007).permutation(len(cases));cases=[cases[int(i)] for i in order]
        master={'domain':'REAL_TRAIN','cases':cases,'sampling_sha256':json.loads((PUBLIC/'sample_selection.json').read_text())['canonical_sha256'],
                'display':{'grid_uV':50,'Hz':100,'initial_seconds':[0,30],'filter':'none','stage_overlay':False,'model_overlay':False},
                'codebook_version':'phase13-annotation-v1'}
        write_new(PRIVATE/'annotation_master.json',master)
        write_new(PUBLIC/'annotation_readiness.json',{'engineering_status':'READY_FOR_APPROVED_LOCAL_REVIEW','human_annotation':'NOT_TESTED',
            'physiological_validation':'NOT_TESTED','approved_reviewers':0,'reason':'User confirmed no approved qualified reviewers are available',
            'primary':len(primary),'secondary':len(secondary),'duplicate_presentations':duplicates,'total_presentations':len(cases),
            'private_master_canonical_sha256':sha256(canonical(master)),'blind_initial_pass':True,'external_transfer':False,
            'human_annotation_blocker':'Requires explicitly approved local qualified reviewers and locked independent exports; no clinical labels fabricated'})


def real_store(registry_path):
    # Check human access BEFORE loading any packet waveforms.
    from .annotation_interface import ReviewStore
    registry=json.loads(registry_path.read_text())
    reviewers=registry.get('reviewers',{})
    if not reviewers or any(not v.get('qualified_approved') or v.get('scope')!='PHASE13_TRAIN_LOCAL' for v in reviewers.values()):
        raise PermissionError('approved qualified local reviewers required before waveform packets')
    receipt=json.loads((PUBLIC/'annotation_readiness.json').read_text())
    master=authenticated_json_identity(PRIVATE/'annotation_master.json',receipt['private_master_canonical_sha256'])
    context=load_cache('annotation_context.npz');cases={}
    for row in master['cases']:
        index=row['source_index']
        cases[row['case_id']]={'master':row,'wave_uV':context['wave_uV'][index,1].tolist(),
              'context_uV':[context['wave_uV'][index,j].tolist() if context['valid'][index,j] else None for j in range(3)]}
    return ReviewStore(cases,reviewers,PRIVATE/'human_reviews',domain='REAL_TRAIN')

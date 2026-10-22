"""Committed pre-ST seal; no access to ST values before fixed identities exist."""
import json
import subprocess
import sys
from pathlib import Path
from .common import ROOT,PRIVATE,PUBLIC,read_json,write_new,sha256,canonical,IntegrityError,ledger


def freeze_st():
    config=read_json(ROOT/'configs/mist_cove_phase14_v1.json');context=read_json(PUBLIC/'context_results.json')
    if context['fits']!=6 or context['updates']!=18000 or read_json(PUBLIC/'context_operator_results.json')['status']!='PASS':raise IntegrityError('contextstage must close beforeST')
    paths=sorted((ROOT/'src/mist_cove/phase14').glob('*.py'))+[ROOT/'configs/mist_cove_phase14_v1.json',ROOT/'docs/mist_cove/PHASE_XIV_ST_FROZEN_EVALUATION_PROTOCOL_V1.md']
    source={str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in paths}
    python=PRIVATE/'detector_environment/bin/python'
    packages=json.loads(subprocess.check_output([str(python),'-m','pip','list','--format=json'],text=True))
    write_new(PUBLIC/'detector_environment.json',{'python':'3.12.9','packages':packages,'historical_patched_environment_modified':False,'pretrained_weights_downloaded':False})
    geometry=read_json(PRIVATE/'geometry.json');manifest=read_json(PRIVATE/'source_manifest.json');scope=read_json(PUBLIC/'data_scope_receipt.json')
    receipt={'version':'phase14-ST-final-freeze-v1','config':config,'config_sha256':sha256(canonical(config)),
             'source_manifest_sha256':sha256(canonical(manifest)),'geometry_sha256':sha256(canonical(geometry)),
             'source_sha256':source,'detector_environment_sha256':sha256(canonical(read_json(PUBLIC/'detector_environment.json'))),
             'original_checkpoint_byte_sha256':scope['historical_model']['checkpoint_byte_sha256'],
             'original_state_digest':scope['historical_model']['named_state_digest'],
             'context_states':[{key:r[key] for key in ('arm','seed','checkpoint_byte_sha256','named_state_digest')} for r in context['rows']],
             'context_SC_gate':context['primary_gate'],'ST_candidate_selection':'all6fixedfinalstates; no STpromotion',
             'ST_waveform_values_previously_opened':False,'ST_classifier_scores_previously_opened':False,
             'ST_witness_sampling':'first1eligibleepochperstagepersonnight,maximum220; pre-STrefinementforpairedconditions; missingnight/stageskept',
             'initial_sampling_proposal_retained':'6074419config proposedfirst2poolingnights; finalrefinementdoesnotchangefullgridinference/models/thresholds',
             'condition_metadata_sha256':read_json(PRIVATE/'conditions.json')['source_sha256'],
             'head_query_allocation_max':{'SC_balanced':24576000,'SC_prevalence':4915200,'ST_balanced':36044800,'context_coarse':7864320,'auxiliary_and_controls':2000000},
             'human_validation':'NOT_TESTED','ST_training_updates':0,'SC_VAL_reads':0,'SC_RESERVED_reads':0}
    write_new(PUBLIC/'frozen_st_protocol.json',receipt)
    ledger([32],'COMPLETE','Complete STpackage, all6states, strict detector environment, official conditions, paired-night audit and unchanged numeric gates sealed before anySTvalues',['reports/mist_cove/phase14_v1/frozen_st_protocol.json'])
    print('STpackage prepared; commit required before any STwaveform access')


def verify_st_seal():
    path=PUBLIC/'frozen_st_protocol.json'
    if not path.exists():raise IntegrityError('committed final STprotocol required before waveform values')
    frozen=read_json(path)
    relative=str(path.relative_to(ROOT));commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=ROOT,text=True).strip()
    if not commit:raise IntegrityError('STprotocol notcommitted')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True,capture_output=True)
    published=subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT)
    if sha256(published)!=sha256(path.read_bytes()):raise IntegrityError('STseal differs fromcommitted bytes')
    source=dict(frozen['source_sha256'])
    amendment=PUBLIC/'ST_engineering_amendment_v1.json'
    if amendment.exists():
        relative_amendment=str(amendment.relative_to(ROOT));amend_commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative_amendment],cwd=ROOT,text=True).strip()
        if not amend_commit:raise IntegrityError('engineering amendment must be committed')
        subprocess.run(['git','merge-base','--is-ancestor',amend_commit,'HEAD'],cwd=ROOT,check=True,capture_output=True)
        if subprocess.check_output(['git','show',amend_commit+':'+relative_amendment],cwd=ROOT)!=amendment.read_bytes():raise IntegrityError('engineering amendment uncommitted change')
        approved=read_json(amendment)
        if approved['original_seal_byte_sha256']!=sha256(path.read_bytes()):raise IntegrityError('amendment points to different scientific seal')
        for name,changes in approved['transport_only_source_changes'].items():
            if source.get(name)!=changes['original_sha256']:raise IntegrityError('amendment original source differs')
            source[name]=changes['corrected_sha256']
    for name,digest in source.items():
        if sha256((ROOT/name).read_bytes())!=digest:raise IntegrityError('pre-STfrozen implementation changed')
    if sha256(canonical(read_json(ROOT/'configs/mist_cove_phase14_v1.json')))!=frozen['config_sha256']:raise IntegrityError('STanalysisconfig changed')
    if sha256(canonical(read_json(PRIVATE/'source_manifest.json')))!=frozen['source_manifest_sha256']:raise IntegrityError('STmanifest changed')
    if sha256(canonical(read_json(PUBLIC/'detector_environment.json')))!=frozen['detector_environment_sha256']:raise IntegrityError('detector environment receipt changed')
    return frozen

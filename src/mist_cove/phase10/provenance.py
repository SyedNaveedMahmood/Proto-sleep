"""Only committed safe aggregates. Never reads private source artifacts."""
import json
import subprocess
from .dependency_graph import ROOT,BASE

REQUIRED=('variant','seed','split_fingerprint','implementation_commit','training_config_identity',
          'environment_identity','normalization_identity','bank_identity','creation_record_identity',
          'original_checkpoint_byte_digest','model_state_identity','independent_historical_reference')
SAFE_RECORDS=('reports/p3b/results.json','reports/mist_cove/phase3_source_v1/execution_receipt.json',
              'reports/mist_cove/phase3_source_v1/source_results.json')


def committed_aggregate(name):
    if name not in SAFE_RECORDS:raise PermissionError('metadata allowlist excludes this source')
    # Git object read does not resolve any actual EEG/bank/checkpoint path.
    return json.loads(subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT))


def assess_metadata(evidence):
    missing=[k for k in REQUIRED if evidence.get(k) in (None,'',False)]
    if evidence.get('variant') not in (None,'raw_context'):missing.append('reviewed_raw_context_variant')
    if evidence.get('seed') not in (None,123,456,789):missing.append('reviewed_seed')
    return {'metadata_status':'INCOMPLETE' if missing else 'READY_FOR_SEPARATE_REVIEW',
            'authentication_status':'BLOCKED','missing_independent_evidence':missing,
            'reason':'Phase X does not open or independently authenticate actual source artifacts; separate artifact-access review required'}


def provenance_review():
    p3b=committed_aggregate(SAFE_RECORDS[0]);phase3=committed_aggregate(SAFE_RECORDS[1]);results=committed_aggregate(SAFE_RECORDS[2])
    selected=next(r for r in p3b['per_seed'] if r['variant']=='raw_context' and r['seed']==123)
    evidence={'variant':'raw_context','seed':123,'split_fingerprint':p3b['split_fingerprint'],
              'implementation_commit':p3b['implementation_commit'],
              'training_config_identity':'configs/p3b_sc_matched_v1.json at pinned base; original run signature must still be checked',
              'environment_identity':None,'normalization_identity':None,'bank_identity':None,
              'creation_record_identity':None,'original_checkpoint_byte_digest':None,
              'model_state_identity':selected['best_state_digest'],
              'independent_historical_reference':'committed original P3B aggregate state digest; not a checkpoint byte digest'}
    alternatives=[{'variant':r['variant'],'seed':r['seed'],'historical_complete_seal_sha256':r['complete_seal_sha256'],
                   'identity':'separate Phase III raw_context fit; not assumed identical to original P3B weights'}
                  for r in results['rows'] if r['variant']=='raw_context']
    return {'version':'phase10-source-provenance-v1','status':'REVIEW_COMPLETE_AUTHENTICATION_BLOCKED',
            'selected_future_candidate':evidence,'assessment':assess_metadata(evidence),
            'original_p3b_run_fingerprint':p3b['run_fingerprint'],
            'separately_identified_phase3_seal_references':alternatives,
            'published_phase3_environment':{k:phase3['environment'][k] for k in ('python','torch','numpy','cuda_build','deterministic','tf32_matmul','tf32_cudnn')},
            'published_split_file_sha256':phase3['split_file_sha256'],
            'artifact_paths_resolved':0,'new_source_artifact_checksums_calculated':0,'source_artifacts_deserialized':0,
            'real_artifacts_opened':0,'safe_records':list(SAFE_RECORDS)}

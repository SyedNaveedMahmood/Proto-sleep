"""Phase-specific historical gate. Fixture or metadata success never unlocks IO."""
from .provenance import verify_inventory,InvalidEvidence
from .evidence_graph import validate_graph

STATUSES={'PASS','FAIL','INCONCLUSIVE','BLOCKED','NOT TESTED'}


def historical_gate(inventory,graph):
    # No artifact-proof/authorization/force-PASS parameter exists in XI-A.
    if any(inventory.get(k)!=v for k,v in {'historical_private_artifacts_opened':0,'real_checkpoint_bytes_hashed':0,
            'real_checkpoints_deserialized':0,'real_EEG_accessed':False}.items()):
        raise InvalidEvidence('out-of-scope actual-artifact claims in metadata inventory')
    proof=verify_inventory(inventory);shape=validate_graph(graph)
    rows=[]
    def add(i,title,status,scope,basis,missing):
        if status not in STATUSES:raise InvalidEvidence('unknown gate status')
        rows.append({'id':f'XI-A{i}','title':title,'status':status,'scope':scope,'basis':basis,'missing_evidence':missing})
    add(0,'Repository provenance','PASS','PINNED_GIT_METADATA','Original source/config/publication objects and ancestry verified;42+9 source hashes reproduce published manifests',[])
    add(1,'Run identity','PASS','PUBLISHED_REFERENCE_ONLY','Original P3B/raw_context/123 row and run fingerprint recovered from pinned publication',['Actual candidate payload identity not tested'])
    add(2,'Split identity','PASS','PUBLISHED_REFERENCE_ONLY','Published split fingerprint and split-file reference agree; fingerprint not relabeled normalization',['Actual split-file byte comparison outside scope'])
    add(3,'Training lineage','INCONCLUSIVE','CODE_AND_PUBLISHED_METADATA','Original protocol/code recovered; detailed creation/environment/TRAIN normalization/bank links uninspected',['Authenticated original root/fit/provenance/creation references; no fitted values access'])
    add(4,'Historical byte reference','INCONCLUSIVE','HISTORICAL_ROOT_ANCHOR','Original root marker905ba78c reference found; code defines transitive checkpoint output-map chain',['Authorized root and selected child JSON read; direct selected checkpoint-byte value not materialized'])
    add(5,'Artifact-byte match','BLOCKED','NO_REAL_ARTIFACT_ACCESS','Phase XI-A intentionally has no actual artifact-byte reader',['Independent selected byte reference and separately authorized reviewed file tool'])
    add(6,'Restricted deserialization','NOT TESTED','FUTURE_REVIEWED_RELEASE','No actual trained artifact loaded; safe generated JSON loader is a separate domain',['Actual artifact authentication and explicit restricted-loader release'])
    add(7,'Authenticated-model adapter identity','NOT TESTED','FUTURE_XI_B','Phase X mock correctness retained; trained-weight behavior untested',['Authenticated original state plus separately authorized generated-input tests'])
    return {'version':'phase11a-historical-gates-v1','domain':'HISTORICAL_P3B_RAW_CONTEXT_123','overall_status':'BLOCKED',
            'subgates':rows,'metadata_proof':proof,'graph_validation':shape,'actual_artifact_authenticated':False,
            'generated_fixture_PASS_can_promote_historical_gate':False,'ready_for_XI_B_launch':False,
            'next_minimal_permission':'Read only pinned original root completion JSON and exact selected child completion JSON; no checkpoint/bank/scaler/EEG access',
            'real_EEG_accessed':False,'real_checkpoint_bytes_hashed':0,'real_checkpoints_deserialized':0,'GPU_queries':0}

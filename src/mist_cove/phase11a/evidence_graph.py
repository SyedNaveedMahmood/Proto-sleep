"""Deterministic graph validation without implicit trust promotion."""
from .provenance import InvalidEvidence,canonical,sha256

EDGE_FIELDS={'id','from','to','required_evidence','available_evidence','evidence_source',
             'independent_of_current_candidate','link_identifiers','linkage','cryptographically_verified_link',
             'original_record_authentication','missing_dependency','falsifier','evidence_level'}


def validate_graph(graph):
    if graph.get('domain')!='HISTORICAL_METADATA_ONLY' or graph.get('current_artifact_authenticated') is not False or graph.get('current_level3_edges')!=0:
        raise InvalidEvidence('Phase XI-A graph cannot claim actual artifact authentication')
    ids=[v['id'] for v in graph['nodes']]
    if len(set(ids))!=len(ids):raise InvalidEvidence('duplicate graph node')
    edge_ids=set();adj={v:[] for v in ids}
    for edge in graph['edges']:
        if set(edge)!=EDGE_FIELDS or edge['id'] in edge_ids or edge['from'] not in adj or edge['to'] not in adj:raise InvalidEvidence('duplicate/unknown/malformed graph edge')
        edge_ids.add(edge['id']);adj[edge['from']].append(edge['to'])
        if edge['evidence_level'] not in (None,1,2):raise InvalidEvidence('no Level3 historical proof in XI-A')
        if edge['evidence_level']==2 and (not edge['independent_of_current_candidate'] or not edge['cryptographically_verified_link']):raise InvalidEvidence('unsupported historical anchor level')
        if not edge['required_evidence'] or not edge['falsifier']:raise InvalidEvidence('missing edge criterion')
    visiting=set();done=set()
    def walk(node):
        if node in visiting:raise InvalidEvidence('cyclic evidence graph')
        if node in done:return
        visiting.add(node)
        for target in adj[node]:walk(target)
        visiting.remove(node);done.add(node)
    for node in adj:walk(node)
    return {'status':'PASS','nodes':len(ids),'edges':len(edge_ids),'graph_sha256':sha256(canonical(graph)),'historical_level3_edges':0}

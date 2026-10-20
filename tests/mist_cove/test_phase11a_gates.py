import copy
import importlib.util
import json
from pathlib import Path
import pytest
from mist_cove.phase11a import gates
from mist_cove.phase11a.evidence_graph import validate_graph
from mist_cove.phase11a.provenance import InvalidEvidence


def historical_metadata():
    return {'historical_private_artifacts_opened':0,'real_checkpoint_bytes_hashed':0,'real_checkpoints_deserialized':0,'real_EEG_accessed':False}


def test_metadata_and_synthetic_pass_cannot_force_real_authentication(monkeypatch):
    monkeypatch.setattr(gates,'verify_inventory',lambda value:{'status':'PASS'})
    monkeypatch.setattr(gates,'validate_graph',lambda value:{'status':'PASS'})
    result=gates.historical_gate(historical_metadata(),{})
    assert result['overall_status']=='BLOCKED' and not result['actual_artifact_authenticated']
    by_id={r['id']:r['status'] for r in result['subgates']}
    assert by_id=={'XI-A0':'PASS','XI-A1':'PASS','XI-A2':'PASS','XI-A3':'INCONCLUSIVE','XI-A4':'INCONCLUSIVE','XI-A5':'BLOCKED','XI-A6':'NOT TESTED','XI-A7':'NOT TESTED'}
    with pytest.raises(TypeError):gates.historical_gate(historical_metadata(),{},force_pass=True)
    wrong=historical_metadata();wrong['real_checkpoint_bytes_hashed']=1
    with pytest.raises(InvalidEvidence):gates.historical_gate(wrong,{})


def graph():
    # A complete minimal synthetic metadata graph; does not refer to actual files.
    return {'domain':'HISTORICAL_METADATA_ONLY','current_artifact_authenticated':False,'current_level3_edges':0,
            'nodes':[{'id':'a'},{'id':'b'}], 'edges':[{'id':'E','from':'a','to':'b','required_evidence':'reference','available_evidence':['published'],
            'evidence_source':'generated','independent_of_current_candidate':True,'link_identifiers':{},'linkage':'direct',
            'cryptographically_verified_link':True,'original_record_authentication':'synthetic metadata','missing_dependency':[],
            'falsifier':'hash mismatch','evidence_level':2}]}


@pytest.mark.parametrize('attack',['duplicate_node','duplicate_edge','cycle','unknown_node','false_level3','unanchored_level2'])
def test_graph_rejects_contradiction_cycles_and_false_trust(attack):
    g=graph();assert validate_graph(g)['status']=='PASS'
    if attack=='duplicate_node':g['nodes'].append({'id':'a'})
    elif attack=='duplicate_edge':g['edges'].append(copy.deepcopy(g['edges'][0]))
    elif attack=='cycle':g['edges'].append({**g['edges'][0],'id':'F','from':'b','to':'a'})
    elif attack=='unknown_node':g['edges'][0]['to']='missing'
    elif attack=='false_level3':g['edges'][0]['evidence_level']=3
    else:g['edges'][0]['independent_of_current_candidate']=False
    with pytest.raises(InvalidEvidence):validate_graph(g)


def test_cli_has_no_artifact_or_force_pass_switch():
    root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('xia_cli',root/'scripts/check_mist_cove_phase11a.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for args in [['--checkpoint','fake.pt'],['--authorize-real-artifact'],['--force-pass'],['--root','/fake'],['load']]:
        with pytest.raises(SystemExit) as error:module.main(args)
        assert error.value.code==2


def test_aggregate_atomicity_privacy_symlink_and_tamper(tmp_path,monkeypatch):
    from mist_cove.phase11a import publication as p
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'PUBLIC',tmp_path/'reports/mist_cove/phase11a_v1')
    p.write_report('gate_status.json',{'status':'BLOCKED'});path=p.PUBLIC/'gate_status.json';before=path.read_bytes()
    with pytest.raises(FileExistsError):p.write_report('gate_status.json',{'status':'PASS'})
    assert path.read_bytes()==before
    for value in [{'private_path':'fake'},{'where':'/unopened/private'},{'checkpoint_bytes':'forbidden'}]:
        with pytest.raises(InvalidEvidence):p.write_report('correctness_receipt.json',value)
    manifest={'files':{'reports/mist_cove/phase11a_v1/gate_status.json':p.sha256(before)}}
    p.write_report('publication_manifest.json',manifest);assert p.verify_publication()['status']=='PASS'
    path.write_text('{}')
    with pytest.raises(InvalidEvidence):p.verify_publication()
    path.unlink();path.symlink_to(tmp_path/'outside_generated.json')
    with pytest.raises(InvalidEvidence):p.read_report('gate_status.json')

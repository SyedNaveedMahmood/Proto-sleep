"""Tests for the actual ST midnight header, partial index and retention problems."""
from pathlib import Path
from types import SimpleNamespace
import datetime as dt
import hashlib
import json
import zipfile
from unittest.mock import patch
import numpy as np
import pytest
from mist_transfer.annotation_compat import (parse_tals, read_annotation_profile, paired_annotations)
from mist_transfer.p1_5 import (selection_analysis, recovered_indices, manifest_analysis, scope_gates, verify_reuse)
from mist_transfer.integrity import index_runs


def pad(value,n):
    return str(value).encode('ascii').ljust(n,b' ')


def fixture(tmp_path, previous_day=False, study='ST', payload=None):
    start=dt.datetime(1994,9,20,0,0,30)
    recording_day=start.date()-dt.timedelta(days=int(previous_day))
    record='Startdate '+recording_day.strftime('%d-%b-%Y').upper()+' X X X'
    def header(records,duration):
        return b'0       '+pad('X X X X',80)+pad(record,80)+start.strftime('%d.%m.%y%H.%M.%S').encode()+pad(512,8)+pad('EDF+C',44)+pad(records,8)+pad(duration,8)+pad(1,4)
    data=payload if payload is not None else b'+0\x14\x14\x00+30\x1530\x14Sleep stage W\x14\x00+60\x1530\x14Sleep stage R\x14\x00'
    if len(data)%2:data+=b'\x00'
    signal=pad('EDF Annotations',16)+b' '*80+b' '*8+pad(0,8)+pad(1,8)+pad(-32768,8)+pad(32767,8)+b' '*80+pad(len(data)//2,8)+b' '*32
    stem='ST7011J' if study=='ST' else 'SC4001E'
    hyp=tmp_path/(stem+'P-Hypnogram.edf');psg=tmp_path/(stem+'0-PSG.edf')
    hyp.write_bytes(header(1,0)+signal+data)
    psg.write_bytes(header(4,30)+b' '*256)
    return hyp,psg


def test_midnight_compatibility_is_read_only_and_preserves_leading_gap(tmp_path):
    hyp,psg=fixture(tmp_path,previous_day=True)
    before=hyp.read_bytes()
    labels,meta=paired_annotations(hyp,psg,120)
    assert labels.tolist()==[-1,0,4,-1]
    assert meta['recording_date_delta_days']==-1
    assert meta['pyedflib_comparison']['status']=='COMPATIBILITY_REQUIRED'
    assert meta['mne_comparison']['status']=='VERIFIED'
    assert meta['fixed_start_datetime']=='1994-09-20T00:00:30'
    assert meta['excluded_epoch_ranges']['unannotated'][0]['original_start']==0
    assert hyp.read_bytes()==before and not meta['original_bytes_modified']


def test_same_day_control_agrees_with_both_independent_readers(tmp_path):
    hyp,psg=fixture(tmp_path)
    labels,meta=paired_annotations(hyp,psg,120)
    assert meta['pyedflib_comparison']['status']=='VERIFIED'
    assert labels.tolist()==[-1,0,4,-1]


@pytest.mark.parametrize('payload',[
    b'+0\x14\x14\x00garbage+30\x1530\x14Sleep stage W\x14\x00',
    b'+0\x14\x14\x00+30\x1530\x14Sleep stage W\x14',
    b'+0\x14\x14\x00+30\x15-30\x14Sleep stage W\x14\x00',
    b'+0\x14\x14\x00+30\x1530\x14Sleep stage X\x14\x00',
    b'+0\x14\x14\x00+30\x1530\x14Sleep stage W\x14\x00\x00HIDDEN',
    b'+30\x14\x14\x00+30\x1530\x14Sleep stage W\x14\x00',
])
def test_malformed_tal_is_not_regex_skipped(payload):
    with pytest.raises(ValueError):parse_tals(payload)


def test_header_size_and_format_and_pairing_are_not_relaxed(tmp_path):
    hyp,psg=fixture(tmp_path,previous_day=True)
    source=hyp.read_bytes()
    hyp.write_bytes(source+b'\x00\x00')
    with pytest.raises(ValueError,match='payload size'):read_annotation_profile(hyp)
    hyp.write_bytes(source)
    p=psg.read_bytes();psg.write_bytes(p[:176]+b'00.01.00'+p[184:])
    with pytest.raises(ValueError,match='paired fixed clock'):paired_annotations(hyp,psg,120)


def test_only_the_documented_st_previous_day_exception_is_permitted(tmp_path):
    hyp,psg=fixture(tmp_path,previous_day=True,study='SC')
    with pytest.raises(ValueError,match='unresolved Recordingfield'):paired_annotations(hyp,psg,120)
    for p in (hyp,psg):
        b=p.read_bytes();p.write_bytes(b[:98]+b'18-SEP-1994'+b[109:])
    with pytest.raises(ValueError,match='unresolved Recordingfield'):paired_annotations(hyp,psg,120)


def test_reader_disagreement_quarantines_instead_of_selecting_favorable_decoder(tmp_path):
    hyp,psg=fixture(tmp_path)
    ann=SimpleNamespace(onset=np.array([30.,60.]),duration=np.array([30.,60.]),
                        description=np.array(['Sleep stage W','Sleep stage R']),orig_time=None)
    with patch('mne.read_annotations',return_value=ann):
        with pytest.raises(ValueError,match='MNE annotation disagreement'):paired_annotations(hyp,psg,120)


@pytest.mark.parametrize('onset,duration',[(31,30),(30,31),(90,60)])
def test_off_grid_and_beyond_signal_stages_fail(tmp_path,onset,duration):
    data=f'+0\x14\x14\x00+{onset}\x15{duration}\x14Sleep stage W\x14\x00'.encode()
    hyp,psg=fixture(tmp_path,payload=data)
    with pytest.raises(ValueError):paired_annotations(hyp,psg,120)


def test_selection_analysis_retains_gaps_and_does_not_claim_exact_lineage():
    labels=np.zeros(250,dtype=int);labels[100:151]=2;labels[110]=-1
    scored=np.flatnonzero(labels>=0);nw=np.flatnonzero(labels[scored]>0)
    indices=scored[max(0,nw[0]-60):nw[-1]+61]
    r=selection_analysis(labels,indices)
    assert r['candidate_rule_matches']['remove_unscored_then_60_scored_epochs_each_edge']
    assert len(r['retained_original_epoch_runs'])==2
    assert r['counts_by_stage']['N2']['excluded']==0
    assert r['counts_by_stage']['Wake']['excluded']>0
    assert r['exact_upstream_recipe_status']=='UNKNOWN'
    assert r['zero_shot_evaluation_use'].startswith('BLOCKED')
    with pytest.raises(ValueError,match='unknown/unannotated'):selection_analysis(labels,[110])


def test_retained_cache_topology_and_digest_are_checked():
    idx=np.array([1,2,5],dtype=np.int64)
    row={'original_epoch_runs':index_runs(idx),'n_epochs':3,
         'original_epoch_index_sha256':hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()}
    assert recovered_indices(row).tolist()==idx.tolist()
    row['original_epoch_index_sha256']='wrong'
    with pytest.raises(ValueError,match='digest mismatch'):recovered_indices(row)


def test_partial_bundled_index_is_not_corruption_or_an_invented_full_manifest(tmp_path):
    root=tmp_path/'edf78';root.mkdir();a=root/'SC4001E0.npz';b=root/'SC4011E0.npz'
    a.write_bytes(b'one');b.write_bytes(b'two')
    text='SC4001E0.npz (application/octet-stream) 3 bytes.\n'
    (root/'MANIFEST.TXT').write_text(text)
    archive=tmp_path/'bundle.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for p in root.iterdir():z.write(p,root.name+'/'+p.name)
    r=manifest_analysis({'sleep_edf_78_npz':str(root)},archive)
    assert r['listed_files']==1 and r['actual_files']==2
    assert r['archive_evidence']['manifest_member_identical']
    assert r['unlisted_files']==['SC4011E0.npz'] and not r['data_corruption_evidence']
    assert r['scope_intent_status'].startswith('UNKNOWN')
    assert (root/'MANIFEST.TXT').read_text()==text
    a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='sizes disagree'):manifest_analysis({'sleep_edf_78_npz':str(root)},archive)


def test_cached_waveform_proof_reuse_refuses_changed_inputs(tmp_path):
    root=tmp_path/'npz';root.mkdir();p=root/'SC4001E0.npz';p.write_bytes(b'synthetic bytes')
    config={k:str(tmp_path/k) for k in ['sleep_edf_20_npz','sleep_edf_78_npz','sleep_edfx_original_root',
                  'sleep_edfx_sc_raw','sleep_edfx_st_raw','shhs_nsrr_root_if_approved']}
    config['sleep_edf_20_npz']=str(root)
    prior={'signature':'prior','provenance':{'config':config,'source_recording_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}}
    proof=tmp_path/'proof.json'
    assert verify_reuse(prior,proof)['files_rehashed']==1
    with patch('mist_transfer.p1_5.sha256_file',side_effect=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()) as h:
        assert verify_reuse(prior,proof)['files_rehashed']==0
        assert all(call.args[0]!=p for call in h.call_args_list)
    p.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='source hash changed'):verify_reuse(prior,proof)


def test_sc_success_never_automatically_approves_st_or_shhs():
    prior={'inventory':{'raw_sc':{'recordings':1},'raw_st':{'recordings':1}},
           'subject_disjoint_partitions':{'edf78_sc_extension_subjects':['SC:20']}}
    r=scope_gates(prior,[{'study':'SC','status':'VERIFIED','recording_id':'SC4001E0'},
                         {'study':'ST','status':'BLOCKED','recording_id':'ST7011J0'}])
    assert r['sc_within_cohort']['p2_eligibility']=='VERIFIED'
    assert r['sc_to_st_proxy']['p2_eligibility']=='BLOCKED'
    assert r['sc_to_st_proxy']['quarantined_recordings']==['ST7011J0']
    assert r['shhs_external']['p2_eligibility']=='BLOCKED'
    assert not r['p2_executed']


def test_single_incomplete_signal_tail_is_diagnostic_not_a_fabricated_epoch(tmp_path):
    payload=b'+0\x14\x14\x00+0\x15150\x14Sleep stage W\x14\x00'
    hyp,psg=fixture(tmp_path,payload=payload)
    b=psg.read_bytes();psg.write_bytes(b[:236]+pad(5,8)+pad(25,8)+b[252:])
    labels,meta=paired_annotations(hyp,psg,125)
    assert len(labels)==4
    assert meta['partial_signal_tail_seconds']==5
    assert meta['scored_tail_status'].startswith('ONE_INCOMPLETE')
    assert meta['annotation_signal_end_overhang_seconds']==25
    assert meta['annotation_tails'][0]['stop_seconds']==150


def test_anonymous_sex_conflict_is_quarantined_and_all_nights_stay_grouped(tmp_path):
    hyp,psg=fixture(tmp_path,study='SC')
    b=hyp.read_bytes();hyp.write_bytes(b[:8]+pad('X F X X',80)+b[88:])
    with pytest.raises(ValueError,match='patient/recording fields'):paired_annotations(hyp,psg,120)
    rows=[{'study':'SC','subject_id':'SC:00','recording_id':'SC4001E0','status':'BLOCKED'},
          {'study':'SC','subject_id':'SC:00','recording_id':'SC4002E0','status':'VERIFIED'},
          {'study':'SC','subject_id':'SC:01','recording_id':'SC4011E0','status':'VERIFIED'},
          {'study':'ST','subject_id':'ST:01','recording_id':'ST7011J0','status':'VERIFIED'}]
    prior={'inventory':{'raw_sc':{'recordings':3},'raw_st':{'recordings':1}},
           'subject_disjoint_partitions':{'edf78_sc_extension_subjects':['SC:00','SC:01']}}
    gates=scope_gates(prior,rows)
    assert gates['sc_within_cohort']['eligible_recordings']==['SC4011E0']
    assert gates['sc_within_cohort']['quarantined_subjects']==['SC:00']
    assert rows[1]['proposed_scope_eligibility']=='BLOCKED'
    assert gates['sc_to_st_proxy']['p2_eligibility']=='CONDITIONAL'


def test_derived_equipment_field_difference_is_recorded_without_changing_identity(tmp_path):
    hyp,psg=fixture(tmp_path,study='SC')
    b=psg.read_bytes();record=b[88:168].decode().strip();record=record[:-1]+'Polyman'
    psg.write_bytes(b[:88]+pad(record,80)+b[168:])
    _,meta=paired_annotations(hyp,psg,120)
    assert meta['recording_equipment_field_difference']
    assert meta['pyedflib_comparison']['status']=='VERIFIED'

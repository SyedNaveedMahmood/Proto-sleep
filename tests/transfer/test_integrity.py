from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pytest
import pyedflib
from mist_transfer.integrity import (identity, validate_splits, open_development_npz, validate_indices,
                                    index_runs, annotation_grid, calibration, affine_match, align_epochs)
from mist_transfer.audit import read_npz, npz_headers, pair_raw, overlap, load_config, audit


def test_study_subject_night_identity():
    a=identity('SC4002E0.npz'); b=identity('ST7012J0-PSG.edf','psg')
    assert (a['subject_id'],a['night'])==('SC:00',2)
    assert (b['subject_id'],b['night'])==('ST:01',2)
    assert identity('SC4002EC-Hypnogram.edf','hyp')['recording_id']==a['recording_id']
    assert identity('SC4261F0.npz')['subject_id']=='SC:26'
    assert identity('SC4701G0.npz')['subject_id']=='SC:70'
    for name in ['SC400E0.npz','SC4003E0.npz','ST7011E0.npz','renamed.npz','SC4002EA.npz']:
        with pytest.raises(ValueError): identity(name)


@pytest.mark.parametrize('key,value',[('subject_id','SC:01'),('recording_id','SC4011E0'),
                                      ('source_sha256','bytes'),('signal_sha256','signal'),('shape_sha256','shape')])
def test_split_identity_and_hash_leakage_before_open(key,value):
    rows=[{'subject_id':'SC:00','recording_id':'SC4001E0','split':'train'},
          {'subject_id':'SC:01','recording_id':'SC4011E0','split':'test'}]
    rows[0][key]=value; rows[1][key]=value
    with patch('numpy.load',side_effect=AssertionError('must not open data')):
        with pytest.raises(ValueError,match='leakage'): validate_splits(rows)


def test_nights_grouped_and_test_opening_prohibited():
    rows=[{'subject_id':'SC:01','recording_id':'SC4011E0','split':'train'},
          {'subject_id':'SC:01','recording_id':'SC4012E0','split':'val'}]
    with pytest.raises(ValueError,match='subject_id'):validate_splits(rows)
    with patch('numpy.load',side_effect=AssertionError('test opened')):
        with pytest.raises(ValueError,match='test opening'):open_development_npz({'split':'test','path':'missing.npz'})


def test_annotation_mapping_gaps_exclusions_and_original_positions():
    y,meta=annotation_grid([0,30,90,120,150,180],[30]*6,
        ['Sleep stage W','Sleep stage 3','Sleep stage 4','Movement time','Sleep stage ?','Sleep stage R'],210)
    assert y.tolist()==[0,3,-1,3,-1,-1,4]
    assert meta['excluded_epoch_ranges']['unannotated'][0]['original_start']==2
    assert meta['excluded_epoch_ranges']['movement'][0]['original_start']==4
    assert meta['excluded_epoch_ranges']['unknown'][0]['original_start']==5
    assert index_runs([0,1,3,6])==[
        {'array_start':0,'array_stop':2,'original_start':0,'original_stop':2},
        {'array_start':2,'array_stop':3,'original_start':3,'original_stop':4},
        {'array_start':3,'array_stop':4,'original_start':6,'original_stop':7}]


@pytest.mark.parametrize('on,dur,text,reason',[
    ([1],[30],['Sleep stage W'],'30-s'),([0],[31],['Sleep stage W'],'30-s'),
    ([30,0],[30,30],['Sleep stage W']*2,'non-monotone'),
    ([0,0],[30,30],['Sleep stage W']*2,'overlapping'),
    ([0],[30],['Sleep stage 5'],'unknown stage'),([0],[-30],['Sleep stage W'],'invalid')])
def test_annotation_failures(on,dur,text,reason):
    with pytest.raises(ValueError,match=reason):annotation_grid(on,dur,text,120)


def test_annotation_outside_signal_and_partial_epoch_are_explicit():
    y,meta=annotation_grid([0,30],[30,90],['Sleep stage W','Sleep stage ?'],65)
    assert y.tolist()==[0,-1]
    assert meta['partial_signal_tail_seconds']==5
    assert meta['annotation_tails'][0]['start_seconds']==60


@pytest.mark.parametrize('idx',[[0,0],[2,1],[-1,0],[0,1.5],[0,np.nan]])
def test_nonmonotone_invalid_epoch_indices(idx):
    with pytest.raises(ValueError):validate_indices(idx,2)


def test_calibration_missing_units_and_degenerate_ranges():
    h={'dimension':'uV','physical_min':-100,'physical_max':100,'digital_min':-32768,'digital_max':32767,'sample_frequency':100}
    calibration(h)
    for change in [{'dimension':''},{'dimension':'normalized'},{'physical_max':-100},{'sample_frequency':0}]:
        with pytest.raises(ValueError):calibration({**h,**change})


def test_full_waveform_affine_alignment_does_not_use_labels():
    raw=np.random.default_rng(123).normal(size=(8,3000))
    x=raw[[1,3,7]]*2.5+8
    idx,fit=align_epochs(x,raw)
    assert idx.tolist()==[1,3,7]
    assert fit['max_relative_residual']<1e-12
    assert fit['gain_min']==pytest.approx(2.5)
    assert affine_match(-raw[0],raw[0]) is None
    assert affine_match(np.zeros(3000),raw[0]) is None
    bad=raw.copy();bad[5]=raw[1]*2+3
    with pytest.raises(ValueError,match='ambiguous'):align_epochs(x,bad)
    with pytest.raises(ValueError,match='unverified'):align_epochs(x+np.random.default_rng(1).normal(size=x.shape),raw)
    with pytest.raises(ValueError,match='non-monotone'):align_epochs(raw[[2,1]],raw)


def test_npz_objects_never_unpickled_and_metadata_failures(tmp_path):
    p=tmp_path/'SC4001E0.npz';x=np.ones((2,3000,1));y=np.array([0,1])
    np.savez(p,x=x,y=y,fs=100,ch_label='EEG Fpz-Cz',header_raw=np.array({'private':'do not load'},dtype=object))
    assert npz_headers(p)['header_raw']['object_metadata_not_opened']
    assert read_npz(p)[3]['declared_units'] is None
    for kwargs,reason in [({'fs':99},'sampling-rate'),({'y':np.array([0,7])},'coding'),
                          ({'y':np.array([0])},'length'),({'epoch_indices':[1,0]},'non-monotone')]:
        np.savez(p,**{'x':x,'y':y,'fs':100,'ch_label':'EEG Fpz-Cz',**kwargs})
        with pytest.raises(ValueError,match=reason):read_npz(p)
    np.savez(p,x=x,y=y,ch_label='EEG Fpz-Cz')
    with pytest.raises(ValueError,match='missing'):read_npz(p)


def test_raw_pair_ambiguous_or_missing_is_rejected(tmp_path):
    config={'sleep_edfx_sc_raw':str(tmp_path),'sleep_edfx_st_raw':str(tmp_path/'ST')}
    (tmp_path/'SC4001E0-PSG.edf').touch()
    with pytest.raises(ValueError,match='unpaired'):pair_raw(config)
    (tmp_path/'SC4001EC-Hypnogram.edf').touch()
    assert len(pair_raw(config))==1
    (tmp_path/'SC4001EA-Hypnogram.edf').touch()
    with pytest.raises(ValueError,match='ambiguous'):pair_raw(config)


def test_same_cohort_overlap_partitions_remove_entire_subject():
    rows=[{'dataset':ds,'study':'SC','subject_id':sid,'recording_id':rid} for ds,sid,rid in
          [('edf20_npz','SC:01','SC4011E0'),('edf78_npz','SC:01','SC4012E0'),('edf78_npz','SC:02','SC4021E0')]]
    matrix,parts=overlap(rows)
    assert parts['edf78_sc_extension_subjects']==['SC:02']
    assert parts['edf78_sc_extension_recordings']==['SC4021E0']
    entry=next(r for r in matrix if r['dataset_a']!=r['dataset_b'])
    assert entry['overlap_subjects']==1 and entry['overlap_recordings']==0
    assert not entry['independent_cohort_comparison']


def tiny_fixture(tmp_path):
    root=tmp_path/'original';sc=root/'sleep-cassette';st=root/'sleep-telemetry';sc.mkdir(parents=True);st.mkdir()
    a=tmp_path/'edf20';b=tmp_path/'edf78';a.mkdir();b.mkdir()
    # Real generated EDF+ signals/annotations; nothing from a participant file.
    wave=(np.random.default_rng(123).normal(size=9000)*10)
    header={'label':'EEG Fpz-Cz','dimension':'uV','sample_frequency':100,
            'physical_min':-100,'physical_max':100,'digital_min':-32768,'digital_max':32767,
            'transducer':'synthetic','prefilter':''}
    import datetime
    start=datetime.datetime(2000,1,1,20)
    p=sc/'SC4001E0-PSG.edf';h=sc/'SC4001EC-Hypnogram.edf'
    with pyedflib.EdfWriter(str(p),1,file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.setStartdatetime(start);w.setSignalHeader(0,header);w.writeSamples([wave])
    with pyedflib.EdfWriter(str(h),0,file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.setStartdatetime(start)
        for onset,text in [(0,'Sleep stage W'),(30,'Sleep stage ?'),(60,'Sleep stage R')]:w.writeAnnotation(onset,30,text)
    with pyedflib.EdfReader(str(p)) as r:raw=r.readSignal(0).reshape(3,3000)
    np.savez(a/'SC4001E0.npz',x=raw[[0,2]].astype(np.float32),y=[0,4],fs=100,ch_label='EEG Fpz-Cz')
    return {'sleep_edf_20_npz':str(a),'sleep_edf_78_npz':str(b),'sleep_edfx_original_root':str(root),
            'sleep_edfx_sc_raw':str(sc),'sleep_edfx_st_raw':str(st),'shhs_nsrr_root_if_approved':str(tmp_path/'restricted'),
            'shhs_access_authorized':False}


def test_generated_edf_fixture_end_to_end_alignment_and_blockers(tmp_path):
    config=tiny_fixture(tmp_path)
    out=tmp_path/'audit'
    # Missing spreadsheets deliberately blocks acceptance but emits all inventory.
    assert audit(config,out)==2
    result=json.loads((out/'audit.json').read_text())
    assert not result['errors']
    assert not result['p2_can_begin']
    row=next(r for r in result['records'] if r['dataset']=='edf20_npz')
    assert row['indices_status']=='all_epochs_waveform_verified'
    assert [r['original_start'] for r in row['original_epoch_runs']]==[0,2]
    assert row['raw_units']=='uV'
    assert not result['leakage_flags']['shhs_opened']
    assert (out/'COMPLETE.json').exists()


def test_end_to_end_label_conflict_stops_science(tmp_path):
    config=tiny_fixture(tmp_path);p=Path(config['sleep_edf_20_npz'])/'SC4001E0.npz'
    with np.load(p) as z:x=z['x']
    np.savez(p,x=x,y=[0,3],fs=100,ch_label='EEG Fpz-Cz')
    out=tmp_path/'audit';assert audit(config,out)==2
    result=json.loads((out/'audit.json').read_text())
    assert any('stage mapping disagreement' in r['error'] for r in result['errors'])
    assert not result['p2_can_begin']


def test_env_and_explicit_path_overrides(tmp_path,monkeypatch):
    p=tmp_path/'config.yaml'
    p.write_text('\n'.join(f'{k}: /absent' for k in ['sleep_edf_20_npz','sleep_edf_78_npz','sleep_edfx_original_root',
                        'sleep_edfx_sc_raw','sleep_edfx_st_raw','shhs_nsrr_root_if_approved']))
    monkeypatch.setenv('MIST_SLEEP_EDF_20_NPZ','/env')
    assert load_config(p)['sleep_edf_20_npz']=='/env'
    assert load_config(p,['sleep_edf_20_npz=/cli'])['sleep_edf_20_npz']=='/cli'
    with pytest.raises(ValueError):load_config(p,['unknown=/bad'])


def test_conflicting_recording_variants_for_one_night(tmp_path):
    config={'sleep_edfx_sc_raw':str(tmp_path),'sleep_edfx_st_raw':str(tmp_path/'ST')}
    for name in ['SC4261E0-PSG.edf','SC4261EC-Hypnogram.edf','SC4261F0-PSG.edf','SC4261FC-Hypnogram.edf']:
        (tmp_path/name).touch()
    with pytest.raises(ValueError,match='conflicting raw recording variants'):pair_raw(config)


def test_incomplete_optional_manifest_is_explicit_blocker_not_invented_provenance(tmp_path):
    config=tiny_fixture(tmp_path)
    # An empty optional distribution manifest does not certify this recording.
    (Path(config['sleep_edf_20_npz'])/'MANIFEST.TXT').write_text('')
    out=tmp_path/'out';assert audit(config,out)==2
    r=json.loads((out/'audit.json').read_text())
    assert not r['errors']
    assert any('incomplete optional' in b for b in r['blockers'])
    assert r['public_provenance_checks']['edf20_npz_manifest']['unlisted_files']==['SC4001E0.npz']
    assert not r['p2_can_begin']


def test_manifest_changed_listed_file_size_stops_before_waveform_opening(tmp_path):
    config=tiny_fixture(tmp_path)
    (Path(config['sleep_edf_20_npz'])/'MANIFEST.TXT').write_text('SC4001E0.npz (application/octet-stream) 1 bytes.\n')
    out=tmp_path/'out'
    with patch('numpy.load',side_effect=AssertionError('must stop before signal interpretation')):
        assert audit(config,out)==2
    r=json.loads((out/'audit.json').read_text())
    assert r['errors'][0]['size_conflicts'][0]['manifest_bytes']==1


@pytest.mark.parametrize('metadata,reason',[
    ({'original_epoch_index':[0,1]},'declared original indices disagree'),
    ({'units':'V'},'physical units conflict'),({'ch_label':'EEG nonexistent'},'montage unavailable')])
def test_raw_differential_metadata_failures(tmp_path,metadata,reason):
    config=tiny_fixture(tmp_path);p=Path(config['sleep_edf_20_npz'])/'SC4001E0.npz'
    with np.load(p) as z:x=z['x'];y=z['y']
    np.savez(p,**{'x':x,'y':y,'fs':100,'ch_label':'EEG Fpz-Cz',**metadata})
    out=tmp_path/'out';assert audit(config,out)==2
    r=json.loads((out/'audit.json').read_text())
    assert any(reason in e['error'] for e in r['errors'])
    assert not r['p2_can_begin']


def test_explicit_test_manifest_stops_before_test_annotation_opening(tmp_path):
    config=tiny_fixture(tmp_path)
    manifest=tmp_path/'splits.json'
    manifest.write_text(json.dumps({'records':[{'subject_id':'SC:00','recording_id':'SC4001E0','split':'test'}]}))
    out=tmp_path/'out'
    with patch('numpy.load',side_effect=AssertionError('reserved test opened')):
        assert audit(config,out,split_manifest=manifest)==2
    r=json.loads((out/'audit.json').read_text())
    assert 'reserved test opening' in r['errors'][0]['error']


def test_expected_edf20_edf78_same_recording_overlap_is_reported_without_false_conflict(tmp_path):
    import shutil
    config=tiny_fixture(tmp_path)
    p=Path(config['sleep_edf_20_npz'])/'SC4001E0.npz'
    # >=3 unique epochs exercises the transformed-overlap threshold too.
    with pyedflib.EdfReader(str(Path(config['sleep_edfx_sc_raw'])/'SC4001E0-PSG.edf')) as reader:
        x=reader.readSignal(0).reshape(3,3000)
    # Three scored synthetic epochs exercise the overlap sketch threshold.
    with pyedflib.EdfWriter(str(Path(config['sleep_edfx_sc_raw'])/'SC4001EC-Hypnogram.edf'),0,
                           file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        import datetime
        w.setStartdatetime(datetime.datetime(2000,1,1,20))
        for onset,text in [(0,'Sleep stage W'),(30,'Sleep stage 2'),(60,'Sleep stage R')]:
            w.writeAnnotation(onset,30,text)
    np.savez(p,x=x.astype(np.float32),y=[0,2,4],fs=100,ch_label='EEG Fpz-Cz')
    shutil.copyfile(p,Path(config['sleep_edf_78_npz'])/p.name)
    out=tmp_path/'out';assert audit(config,out)==2  # missing public metadata blocks, as intended
    r=json.loads((out/'audit.json').read_text())
    assert not r['errors']
    assert r['inventory']['edf78_npz']['fully_aligned_recordings']==1
    entry=next(v for v in r['overlap_matrix'] if v['dataset_a']=='edf20_npz' and v['dataset_b']=='edf78_npz')
    assert entry['equal_full_signal_hashes']==1 and entry['raw_waveform_verified_common_recordings']==1


def test_failed_hypnogram_retains_known_psg_inventory_without_inventing_labels(tmp_path):
    config=tiny_fixture(tmp_path)
    (Path(config['sleep_edfx_sc_raw'])/'SC4001EC-Hypnogram.edf').write_bytes(b'invalid EDF+ header')
    out=tmp_path/'out';assert audit(config,out)==2
    r=json.loads((out/'audit.json').read_text())
    raw=next(v for v in r['records'] if v['dataset']=='raw_sc')
    assert raw['n_epochs']==3 and raw['channels'][0]['dimension']=='uV'
    assert raw['indices_status']=='invalid'
    assert 'mapped_stage_counts' not in raw
    assert r['inventory']['raw_sc']['annotation_verified_recordings']==0

"""SC P2 integrity fixtures; no participant signals or downstream model tests."""
import copy
import datetime as dt
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyedflib
import pytest
from mist_transfer.integrity import annotation_grid, STAGE_MAP
from mist_transfer.preprocessing import (CHANNEL, EXCLUDED, SourceProofs, physical_grid,
    load_sc_scope, validate_scope, signal_contract, build_record, epoch_targets,
    iter_recording, assign_subject_roles, iter_development_split)
from mist_transfer.manifest import (DEFAULT_GATE, DEFAULT_PRIOR, DEFAULT_RESOLUTION,
                                   pilot_selection, validate_real_record, build_manifest, load_complete)
from mist_transfer.provenance import sha256_file


def pad(value,width):return str(value).encode('ascii').ljust(width,b' ')


def synthetic(tmp_path,seconds=270,intervals=None):
    psg=tmp_path/'SC4001E0-PSG.edf';hyp=tmp_path/'SC4001EC-Hypnogram.edf'
    header={'label':CHANNEL,'dimension':'uV','sample_frequency':100,
            'physical_min':-100.,'physical_max':100.,'digital_min':-32768,'digital_max':32767,
            'transducer':'synthetic','prefilter':''}
    digital=np.arange(seconds*100,dtype=np.int32)%40001-20000
    start=dt.datetime(2000,1,1,20)
    with pyedflib.EdfWriter(str(psg),1,file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.setStartdatetime(start);w.setSignalHeader(0,header);w.writeSamples([digital],digital=True)
    intervals=intervals if intervals is not None else [(i*30,30,text) for i,text in enumerate([
        'Sleep stage W','Sleep stage 1','Sleep stage 2','Sleep stage 3','Sleep stage 4',
        'Sleep stage R','Movement time','Sleep stage ?'])]
    payload=b'+0\x14\x14\x00'+b''.join(f'+{on}\x15{dur}\x14{text}\x14\x00'.encode() for on,dur,text in intervals)
    if len(payload)%2:payload+=b'\x00'
    p=psg.read_bytes()[:256]
    h=p[:184]+pad(512,8)+pad('EDF+C',44)+pad(1,8)+pad(0,8)+pad(1,4)
    sh=pad('EDF Annotations',16)+b' '*80+b' '*8+pad(0,8)+pad(1,8)+pad(-32768,8)+pad(32767,8)+b' '*80+pad(len(payload)//2,8)+b' '*32
    hyp.write_bytes(h+sh+payload)
    with pyedflib.EdfReader(str(psg)) as r:
        channels=r.getSignalHeaders();duration=float(r.file_duration)
    try:labels,_=annotation_grid(*zip(*intervals),duration)
    except ValueError:labels=np.full(int(duration//30),-1)  # intentionally invalid synthetic annotations
    row={'study':'SC','subject_id':'SC:00','night':1,'recording_id':'SC4001E0','recording_variant':'E',
        'path':str(psg),'annotation_source':str(hyp),'source_sha256':sha256_file(psg),'annotation_sha256':sha256_file(hyp),
        'signal_sha256':'synthetic-signal','shape_sha256':'synthetic-shape',
        'channels':channels,'duration_seconds':duration,'start_datetime':start.isoformat(),
        'n_epochs':int(duration//30),'mapped_stage_counts':{name:int(np.sum(labels==i)) for i,name in enumerate(['Wake','N1','N2','N3','REM'])}}
    return row,digital


def test_geometry_precedes_annotation_access_and_preserves_complete_timeline(tmp_path):
    row,_=synthetic(tmp_path)
    with patch('mist_transfer.preprocessing.paired_annotations',side_effect=AssertionError('labels opened before geometry')):
        grid,contract=signal_contract(row)
    assert grid['n_epochs']==9 and grid['original_epoch_index']=={'start':0,'stop':9,'step':1}
    assert contract['raw_units']=='uV' and contract['normalization'].startswith('none')
    record=build_record(row,SourceProofs());y,mask,codes,desc,reasons=epoch_targets(record)
    assert y.tolist()==[0,1,2,3,3,4,-1,-1,-1]
    assert mask.tolist()==[True]*6+[False]*3
    assert codes.tolist()==['W','1','2','3','4','R','M','?',None]
    assert reasons.tolist()==['']*6+['movement','unknown','unannotated']
    assert desc[-1] is None and record['scoring_mask']['unscored_epochs']==3


def test_lazy_waveforms_exact_units_times_indices_chunks_and_masks(tmp_path):
    row,digital=synthetic(tmp_path);record=build_record(row,SourceProofs())
    batches=list(iter_recording(record,chunk_epochs=4,include_labels=True))
    assert [len(b['x']) for b in batches]==[4,4,1]
    x=np.concatenate([b['x'] for b in batches]);indices=np.concatenate([b['original_epoch_index'] for b in batches])
    assert x.shape==(9,1,3000) and x.dtype==np.float32 and indices.tolist()==list(range(9))
    h=record['signal_contract']['calibration']
    oracle=((digital.astype(np.float64)-h['digital_min'])*(h['physical_max']-h['physical_min'])/(h['digital_max']-h['digital_min'])+h['physical_min']).astype(np.float32)
    np.testing.assert_array_equal(x[:,0].ravel(),oracle)
    assert batches[-1]['physical_start_time']==['2000-01-01T20:04:00']
    assert batches[-1]['start_seconds'].tolist()==[240] and batches[-1]['start_sample'].tolist()==[24000]
    assert not batches[-1]['scoring_mask'][0] and batches[-1]['original_annotation_code'][0] is None
    assert batches[0]['recording']['source_sha256']==sha256_file(Path(row['path']))
    with patch('mist_transfer.preprocessing.epoch_targets',side_effect=AssertionError('inference consulted labels')):
        assert sum(len(b['x']) for b in iter_recording(record))==9


def test_stage_changes_do_not_change_inference_selection_or_signal(tmp_path):
    row,_=synthetic(tmp_path);r=build_record(row,SourceProofs());changed=copy.deepcopy(r)
    changed['annotation_segments']=[{'onset_seconds':0.,'duration_seconds':270.,'original_description':'Sleep stage R','original_code':'R'}]
    # Inference ignores even deliberately inconsistent scoring metadata.
    first=list(iter_recording(r));second=list(iter_recording(changed))
    assert first[0]['original_epoch_index'].tolist()==second[0]['original_epoch_index'].tolist()
    np.testing.assert_array_equal(first[0]['x'],second[0]['x'])
    with pytest.raises(ValueError,match='scoring metadata'):epoch_targets(changed)


def test_incomplete_epoch_geometry_is_not_fabricated(tmp_path):
    row,_=synthetic(tmp_path,seconds=65,intervals=[(0,60,'Sleep stage W')])
    r=build_record(row,SourceProofs())
    assert r['epoch_grid']['n_epochs']==2 and r['epoch_grid']['partial_tail_samples']==500
    assert r['epoch_grid']['partial_tail_seconds']==5
    assert sum(len(b['x']) for b in iter_recording(r))==2


@pytest.mark.parametrize('cut',[1,100,1000])
def test_truncated_signals_fail_before_annotations_and_reader(tmp_path,cut):
    row,_=synthetic(tmp_path);p=Path(row['path']);p.write_bytes(p.read_bytes()[:-cut])
    with patch('mist_transfer.preprocessing.pyedflib.EdfReader',side_effect=AssertionError('bad extent reached reader')):
        with pytest.raises(ValueError,match='truncated or extra EDF signal bytes'):signal_contract(row)


@pytest.mark.parametrize('change',[
    {'sample_frequency':99},{'dimension':'V'},{'physical_max':101},{'label':'EEG Pz-Oz'}])
def test_corrupted_channel_metadata_is_not_repaired(tmp_path,change):
    row,_=synthetic(tmp_path);row['channels'][0].update(change)
    with pytest.raises(ValueError,match='calibration metadata'):signal_contract(row)


def test_overlapping_annotations_and_corrupted_raw_codes_fail(tmp_path):
    row,_=synthetic(tmp_path,intervals=[(0,60,'Sleep stage W'),(30,30,'Sleep stage R')])
    with pytest.raises(ValueError,match='overlapping'):build_record(row,SourceProofs())
    # Rewrite only our generated fixture, never participant data.
    row,_=synthetic(tmp_path);r=build_record(row,SourceProofs())
    r['annotation_segments'][0]['original_code']='R'
    with pytest.raises(ValueError,match='original annotation code'):epoch_targets(r)


def test_source_hash_changes_are_hard_failures(tmp_path):
    row,_=synthetic(tmp_path);proof=SourceProofs();proof.verify(row['path'],row['source_sha256'])
    p=Path(row['path']);p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='SHA256 changed'):proof.verify(row['path'],row['source_sha256'])


def test_changed_annotation_source_blocks_supervision_without_affecting_label_free_inference(tmp_path):
    row,_=synthetic(tmp_path);record=build_record(row,SourceProofs())
    p=Path(row['annotation_source']);p.write_bytes(p.read_bytes()+b'bad')
    assert sum(len(b['x']) for b in iter_recording(record))==9
    with pytest.raises(ValueError,match='SHA256 changed'):list(iter_recording(record,include_labels=True))


def test_header_discontinuity_requires_a_separate_adapter(tmp_path):
    row,_=synthetic(tmp_path);p=Path(row['path']);b=p.read_bytes();p.write_bytes(b[:192]+pad('EDF+D',44)+b[236:])
    with pytest.raises(ValueError,match='discontinuous'):signal_contract(row)


def test_real_differential_validation_works_on_generated_data(tmp_path):
    row,_=synthetic(tmp_path);record=build_record(row,SourceProofs());v=validate_real_record(record)
    assert v['status']=='PASS' and v['streamed_epochs']==9
    assert v['unscored_epochs']==3 and v['variant']=='E'
    assert v['peak_float32_batch_bytes']==9*3000*4


def test_exact_frozen_scope_and_pilot_cases_without_opening_recordings():
    with patch('pyedflib.EdfReader',side_effect=AssertionError('scope check opened EDF')):
        rows,scope=load_sc_scope(DEFAULT_GATE,DEFAULT_PRIOR,DEFAULT_RESOLUTION)
    assert len(rows)==146 and len({r['subject_id'] for r in rows})==74
    assert not ({r['subject_id'] for r in rows}&EXCLUDED)
    assert sum(r['n_epochs'] for r in rows)==397832
    pilot=pilot_selection(rows,123)
    assert pilot==pilot_selection(rows,123) and len({r['subject_id'] for r in pilot})==10
    assert {r['recording_variant'] for r in pilot}=={'E','F','G'}
    assert {'SC4092E0','SC4571F0'}<={r['recording_id'] for r in pilot}
    assert scope['gate_sha256']


def test_frozen_artifact_changes_are_not_accepted(tmp_path):
    p=tmp_path/'gate.json';p.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):load_sc_scope(p,DEFAULT_PRIOR,DEFAULT_RESOLUTION)


@pytest.mark.parametrize('sid',['SC:06','SC:23','SC:36','SC:74'])
def test_excluded_people_cannot_enter_manifests_or_lazy_access(sid):
    stem='SC4'+sid[-2:]+'1E'
    row={'recording_id':stem+'0','subject_id':sid,'study':'SC','night':1,
        'path':stem+'0-PSG.edf','annotation_source':stem+'C-Hypnogram.edf','source_sha256':'fake'}
    with pytest.raises(ValueError,match='excluded person'):validate_scope([row],{row['recording_id']},{sid})


def test_duplicate_recordings_subject_nights_and_split_leakage_before_io(tmp_path):
    row,_=synthetic(tmp_path);record=build_record(row,SourceProofs())
    with pytest.raises(ValueError,match='duplicate'):assign_subject_roles([record,record],{'SC:00':'train'})
    with patch('pyedflib.EdfReader',side_effect=AssertionError('test opened')):
        with pytest.raises(ValueError,match='reserved test'):list(iter_development_split([record],{'SC:00':'test'},'test'))
    night2=copy.deepcopy(record);night2.update(recording_id='SC4002E0',night=2,path=str(tmp_path/'SC4002E0-PSG.edf'),
        annotation_source=str(tmp_path/'SC4002EC-Hypnogram.edf'),source_sha256='night2-hash')
    assigned=assign_subject_roles([record,night2],{'SC:00':'val'})
    assert [r['split'] for r in assigned]==['val','val']
    with pytest.raises(ValueError,match='cannot be reassigned'):assign_subject_roles(assigned,{'SC:00':'train'})
    other=copy.deepcopy(record);other.update(recording_id='SC4011E0',subject_id='SC:01',path=str(tmp_path/'SC4011E0-PSG.edf'),
        annotation_source=str(tmp_path/'SC4011EC-Hypnogram.edf'),source_sha256='other-hash')
    # Identical inherited signal/shape hashes also block cross-role leakage.
    with pytest.raises(ValueError,match='signal_sha256'):assign_subject_roles([record,other],{'SC:00':'train','SC:01':'test'})


def test_complete_build_requires_pilot_without_data_open_or_output(tmp_path):
    with patch('pyedflib.EdfReader',side_effect=AssertionError('complete build before pilot')):
        with pytest.raises(ValueError,match='passing --pilot-dir'):build_manifest(tmp_path/'full',mode='complete')
    assert not (tmp_path/'full').exists()


def test_atomic_interruption_resume_and_completed_no_reload(tmp_path):
    row,_=synthetic(tmp_path);out=tmp_path/'pilot'
    scope={'historical_signature':'synthetic-fixture'}
    with patch('mist_transfer.manifest.load_sc_scope',return_value=([row],scope)), patch('mist_transfer.manifest.pilot_selection',return_value=[row]):
        with patch('mist_transfer.manifest.validate_real_record',side_effect=RuntimeError('synthetic interruption')):
            with pytest.raises(RuntimeError,match='interruption'):build_manifest(out)
        assert (out/'records/SC4001E0.json').exists() and not (out/'COMPLETE.json').exists()
        with patch('mist_transfer.manifest.build_record',side_effect=AssertionError('completed descriptor rebuilt')):
            assert build_manifest(out,resume=True)==0
        complete=load_complete(out)
        assert complete['validation']['record_descriptors_resumed']==1
        with patch('pyedflib.EdfReader',side_effect=AssertionError('completed run reopened EDF')):
            assert build_manifest(out,resume=True)==0
        import json
        descriptor=out/'records/SC4001E0.json';saved=json.loads(descriptor.read_text());saved['record']['night']=2
        descriptor.write_text(json.dumps(saved))
        with pytest.raises(ValueError,match='output digest mismatch'):build_manifest(out,resume=True)

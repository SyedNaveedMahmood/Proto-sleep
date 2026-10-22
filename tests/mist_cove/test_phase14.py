import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from mist_cove.phase14.access_control import RootAccess
from mist_cove.phase14.common import IntegrityError
from mist_cove.phase14.event_schema import validate_arrays,event_intervals,validate_record_events,shifted_intervals
from mist_cove.phase14.signal_preprocessing import frequency_checks,segment_bounds
from mist_cove.phase14.context_model import ContextLedger,draw_update
from mist_cove.phase14.context_operator import ContextOperator,summarize
from mist_cove.phase13.provenance import isolated_rng


def arrays():
 return {'EEG Fpz-Cz':np.ones(3000),'EEG Pz-Oz':np.ones(3000),'EOG horizontal':np.ones(3000),'EMG submental':np.ones(30),'EMG submental__fs':np.array(1.),'EEG Fpz-Cz__bounds':np.array([-1000,1000]),'EEG Pz-Oz__bounds':np.array([-1000,1000])}


@pytest.mark.parametrize('path',['sleep-cassette/SC9999E0-PSG.edf','sleep-cassette/generated_val.edf','sleep-telemetry/unissued.edf','../escape.edf','/absolute.edf'])
def test_role_denied_before_any_file_stat_open(tmp_path,monkeypatch,path):
 a=RootAccess(str(tmp_path));monkeypatch.setattr(os,'open',lambda *args,**kwargs:pytest.fail('unauthorized open'))
 with pytest.raises(IntegrityError):
  with a.file(path,'source'):pass
 a.close()


def test_root_and_source_symlink_hardlink_and_type_rejected(tmp_path):
 actual=tmp_path/'actual';actual.mkdir();link=tmp_path/'link';link.symlink_to(actual,target_is_directory=True)
 with pytest.raises(OSError):RootAccess(str(link))
 root=RootAccess(str(actual));target=actual/'RECORDS';target.write_text('generated');target.unlink();target.symlink_to(tmp_path/'outside')
 with pytest.raises(OSError):root.bytes('RECORDS')
 target.unlink();target.write_text('generated');os.link(target,actual/'duplicate')
 with pytest.raises(IntegrityError):root.bytes('RECORDS')
 root.close()


def test_bounded_metadata_hash_and_descriptor_swap(tmp_path):
 p=tmp_path/'RECORDS';p.write_bytes(b'generated');a=RootAccess(str(tmp_path))
 assert a.bytes('RECORDS',expected=hashlib.sha256(b'generated').hexdigest())==b'generated'
 with pytest.raises(IntegrityError):a.bytes('RECORDS',limit=2)
 with pytest.raises(IntegrityError):a.bytes('RECORDS',expected='0'*64)
 with pytest.raises(IntegrityError):
  with a.file('RECORDS') as(fd,s):p.write_bytes(b'changed size')
 a.close()


@pytest.mark.parametrize('bad',['truth','predicted_stage','hypnogram'])
def test_detector_never_accepts_stages_or_model_predictions(bad):
 a=arrays();a[bad]=np.zeros(30)
 with pytest.raises(ValueError):validate_arrays(a)


def test_native_EMG_rates_and_units_are_distinct():
 a=arrays();validate_arrays(a);a['EMG submental__fs']=np.array(100.)
 with pytest.raises(ValueError):validate_arrays(a)
 a['EMG submental']=np.ones(3000);validate_arrays(a)
 a['EEG Fpz-Cz'][0]=np.nan
 with pytest.raises(ValueError):validate_arrays(a)


def test_record_boundaries_intervals_and_timestamp_falsification():
 assert event_intervals([{'start':29.,'end':31.}],0)==[[2900,3000]]
 assert event_intervals([{'start':29.,'end':31.}],1)==[[0,100]]
 assert sum(b-a for a,b in shifted_intervals([[2500,2900]],10))==400
 with pytest.raises(ValueError):validate_record_events([{'start':-1,'end':1}],30)
 with pytest.raises(ValueError):validate_record_events([{'start':29,'end':31}],30)
 assert list(segment_bounds(3600))==[(0,1830,0,1800),(1770,3600,1800,3600)]
 assert all(v['midband_gain']>.9 for v in frequency_checks().values())


@pytest.mark.parametrize('arm',['C0','C1'])
def test_context_accounting_missing_neighbors_and_physical_support(arm):
 m=ContextLedger(arm).eval();x=torch.randn(1,3,1,3000);v=torch.tensor([[True,True,True]]);out=m(x,v)
 torch.testing.assert_close(out['logits'],m.base.bias+out['global'].sum(1)+out['contributions'].sum((1,2)),atol=0,rtol=0)
 op=ContextOperator(m);cache=op.cache(x,v);full=torch.ones(1,17,dtype=torch.bool)
 torch.testing.assert_close(op.logits(cache,full)[0],out['logits'][0],atol=1e-6,rtol=1e-6)
 mask=full.clone();mask[:,0]=False;mask[:,4]=False
 changed=x.clone();changed[:,0]+=50;changed[:,1,:,600:800]+=100
 other=op.cache(changed,v)
 for neutral in (0.,.01):torch.testing.assert_close(op.logits(cache,mask,neutral),op.logits(other,mask,neutral),atol=1e-6,rtol=1e-6)
 invalid=torch.tensor([[False,True,False]]);a=m(x,invalid)['logits'];changed=x.clone();changed[:,0]+=50;changed[:,2]+=100
 torch.testing.assert_close(a,m(changed,invalid)['logits'],atol=0,rtol=0)
 if arm=='C0':torch.testing.assert_close(m(x,v)['logits'],m.base(x[:,1])['logits'],atol=1e-6,rtol=1e-6)


def test_context_exposure_rng_and_equal_parameter_count():
 x=torch.randn(20,3,1,3000);y=torch.arange(20)%5;valid=torch.ones(20,3,dtype=torch.bool);streams=[];counts=[]
 for arm in ('C0','C1'):
  with isolated_rng(123):
   m=ContextLedger(arm);o=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(10001)
   streams.append([draw_update(m,o,x,y,valid,g)['draw_sha256'] for _ in range(2)]);counts.append(sum(p.numel() for p in m.parameters()))
 assert streams[0]==streams[1] and counts[0]==counts[1]==11974


def test_context_complete17variable_boolean_circuit_generated_only():
 with isolated_rng(14031):
  m=ContextLedger('C1').eval();x=torch.randn(1,3,1,3000);v=torch.ones(1,3,dtype=torch.bool);op=ContextOperator(m);cache=op.cache(x,v)
  table=op.table(cache,int(cache[3].argmax()));s=summarize(table)
  assert s['exact_all_rows'] and s['masks']==131072
  with torch.no_grad():m.context_weights[0]=.3
  with pytest.raises(ValueError):op.logits(cache,torch.ones(1,17,dtype=torch.bool))


@pytest.mark.parametrize('arm',['C0','C1'])
def test_context_sealed_exact_optimizer_and_generator_resume(tmp_path,arm):
 if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))<(2,10):pytest.skip('positive restricted resume requires patched runtime')
 from mist_cove.phase13.optimization import training_payload,restore_payload
 from mist_cove.phase13.seals import save_sealed,load_sealed
 with isolated_rng(14029):
  x=torch.randn(20,3,1,3000);v=torch.ones(20,3,dtype=torch.bool);y=torch.arange(20)%5
  m=ContextLedger(arm);o=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(99);identity={'generated':True,'arm':arm}
  history=[draw_update(m,o,x,y,v,g) for _ in range(2)];p=tmp_path/'generated.pt';save_sealed(p,training_payload(m,o,g,2,history,identity),identity)
  expected=draw_update(m,o,x,y,v,g);values={k:t.clone() for k,t in m.state_dict().items()}
  payload,_=load_sealed(p,identity);b=ContextLedger(arm);bo=torch.optim.Adam(b.parameters(),lr=.001);bg=torch.Generator()
  restore_payload(payload,b,bo,bg,identity);actual=draw_update(b,bo,x,y,v,bg)
  assert expected==actual and all(torch.equal(t,b.state_dict()[k]) for k,t in values.items())
  for a,c in zip(o.state.values(),bo.state.values()):
   assert all(torch.equal(a[k],c[k]) for k in a)


def test_metadata_reader_cannot_expose_signal_access():
 from mist_cove.phase14.edf_reader import ReaderView
 class Fake:
  file_duration=30
  def readSignal(self,*args,**kwargs):pytest.fail('forbidden waveform read')
 view=ReaderView(Fake(),False);assert view.file_duration==30
 with pytest.raises(IntegrityError):view.readSignal(0)
 with pytest.raises(IntegrityError):view._physical_read(0,0,10)


def test_subject_grouped_interval_matching_and_budget_fairness():
 from mist_cove.phase14.event_alignment import matched_candidates,align,interval_concordance
 wave=np.sin(2*np.pi*13*np.arange(3000)/100);mask=(1<<1)|(1<<8)
 c=matched_candidates(wave,mask,mask,list(range(15)),14024)
 assert all(v.bit_count()==2 for k,v in c.items() if k!='greedy_own_budget')
 events={'EEG Pz-Oz|spindle':[{'start':2.,'end':3.}], 'EEG Fpz-Cz|spindle':[{'start':4.,'end':5.}]}
 result=align(events,0,c,{'EEG Pz-Oz|spindle':[[100,200]]})
 assert 'cross_channel_permuted_same_recording' not in result['proxies']['EEG Fpz-Cz|spindle']['methods']
 a=[{'start':0.,'end':1.},{'start':2.,'end':3.}];b=[{'start':.1,'end':1.1},{'start':2.2,'end':3.2}]
 assert interval_concordance(a,b)['F1']==1


def test_ST_missing_seal_denies_before_data_access(tmp_path,monkeypatch):
 import mist_cove.phase14.protocol as p
 monkeypatch.setattr(p,'PUBLIC',tmp_path)
 with pytest.raises(IntegrityError):p.verify_st_seal()


def test_isolated_detector_command_avoids_stdlib_shadowing(tmp_path,monkeypatch):
 import subprocess,sys
 import mist_cove.phase14.proxies as p
 captured=[]
 class Dummy:pass
 with monkeypatch.context() as m:
  m.setattr(p,'PRIVATE',tmp_path);m.setattr(p.subprocess,'Popen',lambda cmd,**kwargs:(captured.append(cmd) or Dummy()))
  d=p.Detector();d.stderr.close()
 command=captured[0];assert '-I' in command
 (tmp_path/'statistics.py').write_text("raise AssertionError('stdlibshadowed')")
 worker=tmp_path/'worker.py';worker.write_text('import statistics; assert statistics.NormalDist().mean==0')
 command=[sys.executable,*command[1:-1],str(worker)]
 result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
 assert result.returncode==0,result.stderr


def test_root_authorization_exact_digest_required(tmp_path,monkeypatch):
 import mist_cove.phase14.common as c
 monkeypatch.setattr(c,'PRIVATE',tmp_path)
 with pytest.raises(FileNotFoundError):c.authorized_root()
 (tmp_path/'local_authorization.json').write_text(json.dumps({'root':'/generated/unapproved'}))
 with pytest.raises(IntegrityError):c.authorized_root()


@pytest.mark.parametrize('role',['SC_VAL','SC_RESERVED','SHHS','ST_TRAIN'])
def test_no_fitting_or_excluded_role_record_issuance(role):
 from mist_cove.phase14.edf_reader import MultiReader
 m=MultiReader.__new__(MultiReader)
 with pytest.raises(IntegrityError):m.records(role)

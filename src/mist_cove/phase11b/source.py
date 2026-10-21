"""Role-first authenticated descriptor transport and sealed in-memory EDF reads."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path, PurePath
import time
from types import FunctionType
import numpy as np
from .security import authenticated_snapshot,IntegrityError,sha256


def development_records(frozen,role):
    if role not in ('train','val'):raise IntegrityError('unauthorized role before any descriptor access')
    from mist_transfer import protocol
    protocol.validate_role_records(frozen['records'])
    selected=[r for r in frozen['records'] if r['split']==role]
    if not selected or any(frozen['subject_roles'][r['subject_id']]!=role for r in selected):raise IntegrityError('participant role substitution')
    verified={}
    for ref in selected:
        path=Path(ref['descriptor'])
        with authenticated_snapshot(path,ref['descriptor_sha256'],allowlist=[path],limit=2*1024*1024) as snap:
            verified[str(path)]=snap.data
    class SnapshotPath:
        def __init__(self,value):self.key=str(value)
        def __str__(self):return self.key
        @property
        def name(self):return PurePath(self.key).name
        def read_text(self):return verified[self.key].decode('utf-8')
    # Execute the unchanged historical load_development code against authenticated
    # immutable bytes. Only its Path/read/hash transport is replaced, locally.
    env={**protocol.load_development.__globals__,'Path':SnapshotPath,
         'sha256_file':lambda p:sha256(verified[str(p)])}
    reader=FunctionType(protocol.load_development.__code__,env,'load_development')
    return reader(frozen,role)


def validate_geometry(record):
    if record.get('split') not in ('train','val') or record.get('study')!='SC':raise IntegrityError('unauthorized source role')
    contract=record['signal_contract'];grid=record['epoch_grid']
    if contract['montage']!='Fpz-Cz' or contract['fs_native']!=100 or contract['fs_output']!=100 or contract['raw_units']!='uV' or grid['samples_per_epoch']!=3000:
        raise IntegrityError('original physical EEG geometry differs')
    if grid['original_epoch_index']!={'start':0,'stop':grid['n_epochs'],'step':1}:raise IntegrityError('compressed physical grid')


@contextmanager
def sealed_memfile(data):
    if hasattr(os,'memfd_create'):
        fd=os.memfd_create('mist-approved-source',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    else:
        # This Linux Python build omits os.memfd_create; use the identical libc
        # entrypoint, not an unsealed temporary-file fallback.
        import ctypes
        libc=ctypes.CDLL(None,use_errno=True)
        create=libc.memfd_create;create.argtypes=[ctypes.c_char_p,ctypes.c_uint];create.restype=ctypes.c_int
        fd=create(b'mist-approved-source',3)
        if fd<0:raise OSError(ctypes.get_errno(),'sealed source memory unavailable')
    try:
        view=memoryview(data);offset=0
        while offset<len(view):offset+=os.write(fd,view[offset:offset+1048576])
        fcntl.fcntl(fd,getattr(fcntl,'F_ADD_SEALS',1033),15)
        yield '/proc/self/fd/'+str(fd)
    finally:os.close(fd)


class SourceAccess:
    def __init__(self,frozen,normalization,adapter_receipt):
        if adapter_receipt.get('status')!='PASS' or not adapter_receipt.get('state_unchanged'):
            raise IntegrityError('trained adapter gate required before data access')
        self.frozen=frozen;self.normalization=normalization;self.by_role={};self.counts={'descriptor_files':0,'waveform_file_reads':0,'annotation_hash_reads':0,'waveform_bytes':0,'physical_epochs_decoded':0,'source_io_seconds':0.0}

    def records(self,role):
        if role not in ('train','val'):raise IntegrityError('unauthorized source role')
        if role not in self.by_role:
            self.by_role[role]=development_records(self.frozen,role);self.counts['descriptor_files']+=len(self.by_role[role])
        return self.by_role[role]

    def require_record(self,record):
        role=record.get('split')
        if role not in ('train','val'):raise IntegrityError('unauthorized record before source access')
        if not any(record is r for r in self.records(role)):raise IntegrityError('unissued or substituted record')
        validate_geometry(record)

    @contextmanager
    def reader(self,record):
        self.require_record(record)
        import pyedflib
        start=time.perf_counter()
        annotation=Path(record['annotation_source'])
        with authenticated_snapshot(annotation,record['annotation_sha256'],allowlist=[annotation],limit=2*1024*1024):pass
        self.counts['annotation_hash_reads']+=1
        path=Path(record['path'])
        with authenticated_snapshot(path,record['source_sha256'],allowlist=[path],limit=128*1024*1024) as snap:
            self.counts['waveform_file_reads']+=1;self.counts['waveform_bytes']+=len(snap.data)
            with sealed_memfile(snap.data) as immutable_path:
                with pyedflib.EdfReader(immutable_path) as edf:
                    c=record['signal_contract'];channel=c['channel_index']
                    if edf.getSignalHeader(channel)!=c['calibration'] or int(edf.getNSamples()[channel])!=record['epoch_grid']['n_samples_native'] or edf.file_duration!=c['duration_seconds'] or edf.getStartdatetime().isoformat()!=c['start_datetime']:
                        raise IntegrityError('authenticated source decoder/header geometry differs')
                    self.counts['source_io_seconds']+=time.perf_counter()-start
                    yield edf

    def read(self,record,reader,indices):
        self.require_record(record);indices=np.asarray(indices,dtype=np.int64)
        if indices.ndim!=1 or len(indices)>256 or np.any(indices<0) or np.any(indices>=record['epoch_grid']['n_epochs']):raise IntegrityError('bounded original physical indices required')
        started=time.perf_counter();wave=np.empty((len(indices),1,3000),np.float32)
        cuts=np.r_[0,np.flatnonzero(np.diff(indices)!=1)+1,len(indices)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi==lo:continue
            values=reader.readSignal(record['signal_contract']['channel_index'],start=int(indices[lo])*3000,n=int(hi-lo)*3000)
            if len(values)!=(hi-lo)*3000 or not np.isfinite(values).all():raise IntegrityError('source sample count/finiteness')
            wave[lo:hi]=values.reshape(-1,1,3000)
        normalized=(wave-self.normalization['mean_uV'])/self.normalization['std_uV']
        self.counts['physical_epochs_decoded']+=len(indices);self.counts['source_io_seconds']+=time.perf_counter()-started
        return normalized,wave


def select_epochs(records,per_stage):
    from mist_transfer.preprocessing import epoch_targets
    result=[];missing=[]
    for subject in sorted({r['subject_id'] for r in records}):
        ordered=sorted((r for r in records if r['subject_id']==subject),key=lambda r:(r['recording_id'],r['night']))
        for stage in range(5):
            pool=[]
            for r in ordered:
                y,mask,*_=epoch_targets(r)
                pool.extend({'recording_id':r['recording_id'],'subject':subject,'stage':stage,'epoch':int(i)} for i in np.flatnonzero(mask&(y==stage)))
            result.extend(pool[:per_stage]);missing.append({'subject':subject,'stage':stage,'selected':min(per_stage,len(pool)),'missing':max(0,per_stage-len(pool))})
    return result,missing

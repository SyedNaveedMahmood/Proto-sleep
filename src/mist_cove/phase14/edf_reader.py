"""Decode verified immutable source bytes; waveform access requires freeze."""
from contextlib import contextmanager
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pyedflib
from mist_cove.phase11b.source import sealed_memfile
from mist_transfer.annotation_compat import paired_annotations
from .common import PRIVATE,PUBLIC,read_json,IntegrityError,sha256,canonical,write_new,measured,ledger
from .access_control import RootAccess


def manifest():
    value=read_json(PRIVATE/'source_manifest.json');receipt=read_json(PUBLIC/'data_scope_receipt.json')
    if sha256(canonical(value))!=receipt['manifest_canonical_sha256']:raise IntegrityError('source manifest identity changed')
    return value


class ReaderView:
    """Metadata views cannot accidentally expose waveform-reader methods."""
    def __init__(self,reader,values):self._reader=reader;self._values=values
    def __getattr__(self,name):
        if name not in {'getSignalHeaders','getSignalHeader','getSignalLabels','getNSamples','getSampleFrequency','getStartdatetime','file_duration'}:
            raise IntegrityError('waveform methods unavailable through metadata view')
        return getattr(self._reader,name)
    def _physical_read(self,index,start,n):
        if not self._values:raise IntegrityError('waveform capability not issued')
        return self._reader.readSignal(index,start=start,n=n)


class MultiReader:
    def __init__(self):
        self.manifest=manifest();self.access=RootAccess(allowed=self.manifest['allowed']);self.issued={id(r):r for r in self.manifest['records']}
    def records(self,role):
        if role not in ('SC_TRAIN','ST_EVAL'):raise IntegrityError('role denied before source IO')
        return [r for r in self.manifest['records'] if r['role']==role]
    def require(self,record):
        if self.issued.get(id(record)) is not record or record['role'] not in ('SC_TRAIN','ST_EVAL'):raise IntegrityError('unissued record')
    @contextmanager
    def open(self,record,values=False):
        self.require(record)
        if values and record['role']=='ST_EVAL':
            from .protocol import verify_st_seal
            verify_st_seal()
        rel=record['psg'];expected=self.manifest['allowed'][rel]['sha256']
        data=self.access.bytes(rel,'source',128*1024*1024,expected)
        with sealed_memfile(data) as path:
            with pyedflib.EdfReader(path) as reader:
                self.access.counts['waveform_reads' if values else 'header_reads']+=1
                yield ReaderView(reader,values),data[:256]
    def read(self,record,reader,channel,start,stop):
        self.require(record)
        if record['role']=='ST_EVAL':
            from .protocol import verify_st_seal
            verify_st_seal()
        allowed={'EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental','Event marker','Marker'}
        if channel not in allowed:raise IntegrityError('channel outside frozen contract')
        labels=reader.getSignalLabels()
        if labels.count(channel)!=1:raise IntegrityError('missing/ambiguous exact channel')
        j=labels.index(channel);fs=reader.getSampleFrequency(j)
        if not 0<=start<stop<=reader.file_duration or abs(start*fs-round(start*fs))>1e-6 or abs(stop*fs-round(stop*fs))>1e-6:raise IntegrityError('bounded native sample grid required')
        h=reader.getSignalHeader(j);factor={'uV':1.,'µV':1.,'mV':1000.,'V':1e6}.get(h['dimension'])
        # Event marker and SC envelope have nonvoltage native units; preserve.
        if factor is None and channel not in ('EMG submental','Event marker','Marker'):raise IntegrityError('unapproved physical units')
        x=reader._physical_read(j,start=int(round(start*fs)),n=int(round((stop-start)*fs)))
        if len(x)!=int(round((stop-start)*fs)) or not np.isfinite(x).all():raise IntegrityError('nonfinite/truncated calibrated samples')
        return x*(factor if factor is not None else 1.),{'channel':channel,'native_fs':fs,'units':'uV' if factor is not None else h['dimension'],
                'source_sha256':self.manifest['allowed'][record['psg']]['sha256'],'start_seconds':start,'stop_seconds':stop}
    def hypnogram(self,record,psg_header,duration):
        self.require(record);rel=record['hypnogram'];data=self.access.bytes(rel,'source',2*1024*1024,self.manifest['allowed'][rel]['sha256'])
        # Reuse the unchanged paired strict TAL/MNE/EDF compatibility proof on
        # private immutable snapshots with their exact original basenames.
        with tempfile.TemporaryDirectory(dir=PRIVATE) as directory:
            hp=Path(directory)/Path(rel).name;pp=Path(directory)/Path(record['psg']).name
            for path,content in ((hp,data),(pp,psg_header)):
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'wb') as stream:stream.write(content)
            y,detail=paired_annotations(hp,pp,duration)
        self.access.counts['hypnogram_reads']+=1
        return y,detail


def inspect():
    with measured('file_integrity_headers_and_hypnograms') as counts:
        source=MultiReader();rows=[];failures=[]
        for r in source.manifest['records']:
            try:
                with source.open(r) as (reader,header):
                    headers=reader.getSignalHeaders();samples=reader.getNSamples().tolist();duration=float(reader.file_duration)
                    if header[192:236].strip() not in (b'',b'EDF+C'):raise IntegrityError('discontinuous EDF not authorized')
                    if len({h['label'] for h in headers})!=len(headers):raise IntegrityError('duplicate channel labels')
                    for h,n in zip(headers,samples):
                        if not np.isfinite([h['sample_frequency'],h['physical_min'],h['physical_max'],h['digital_min'],h['digital_max']]).all() or h['physical_max']<=h['physical_min'] or h['digital_max']<=h['digital_min'] or abs(n/h['sample_frequency']-duration)>1e-6:raise IntegrityError('invalid channel calibration or duration')
                    names=[h['label'] for h in headers]
                    for name in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal'):
                        if name not in names or headers[names.index(name)]['sample_frequency']!=100:raise IntegrityError('required EEG/EOG100Hz channel missing')
                    y,detail=source.hypnogram(r,header,duration)
                    if r['role']=='SC_TRAIN':
                        from mist_transfer.preprocessing import epoch_targets
                        if not np.array_equal(y,epoch_targets(r['historical'])[0]):raise IntegrityError('historical SC physical labels differ')
                    rows.append({'record':{k:r[k] for k in ('psg','hypnogram','role','subject','recording','night')},'headers':headers,'samples':samples,
                        'duration':duration,'start_datetime':reader.getStartdatetime().isoformat(),'labels':y.tolist(),'hypnogram':detail})
            except (ValueError,OSError,IntegrityError) as exc:failures.append({'recording':r['recording'],'role':r['role'],'reason':str(exc)})
        write_new(PRIVATE/'geometry.json',{'rows':rows,'failures':failures})
        channels={}
        for role in ('SC_TRAIN','ST_EVAL'):
            rr=[v for v in rows if v['record']['role']==role];groups={}
            for row in rr:
                for h in row['headers']:
                    key=h['label'];g=groups.setdefault(key,{'recordings':0,'sample_rates_Hz':set(),'units':set()});g['recordings']+=1;g['sample_rates_Hz'].add(h['sample_frequency']);g['units'].add(h['dimension'])
            channels[role]={key:{k:sorted(v) if isinstance(v,set) else v for k,v in value.items()} for key,value in groups.items()}
        write_new(PUBLIC/'file_integrity_receipt.json',{'verified_PSG':len(rows),'verified_hypnogram':len(rows),'failed_records':len(failures),'independent_official_hashes':True,
            'SC_verified':sum(r['record']['role']=='SC_TRAIN' for r in rows),'ST_verified':sum(r['record']['role']=='ST_EVAL' for r in rows),
            'geometry_canonical_sha256':sha256(canonical({'rows':rows,'failures':failures})),'ST_waveform_values_opened':False,'source_access':source.access.counts,
            'failures_by_role':{role:sum(f['role']==role for f in failures) for role in ('SC_TRAIN','ST_EVAL')}})
        write_new(PUBLIC/'channel_contract.json',{'audited':channels,'decoded_channels':['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental','Event marker'],
            'slow_auxiliary_not_decoded':['Resp oro-nasal','Temp rectal'],'SC_EMG':'native processed1Hz envelope, no rawEMG equivalence','ST_EMG':'native100Hz raw signal',
            'original_model_channel':'EEG Fpz-Cz only','human_event_ground_truth':False})
        counts.update(source.access.counts);source.access.close()
        ledger([5,6,7],'COMPLETE','Every issued source pair independently hashed and strict geometry audited; any failures quarantined; no ST waveform values', ['reports/mist_cove/phase14_v1/file_integrity_receipt.json','reports/mist_cove/phase14_v1/channel_contract.json'])
        print(json.dumps({'verified':len(rows),'failed':len(failures),'channels':channels}))

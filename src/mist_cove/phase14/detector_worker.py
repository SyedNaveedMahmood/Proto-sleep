"""Standalone isolated YASA process: calibrated arrays only, no model/stage input."""
import hashlib
import json
import os
import stat
import sys
import time
import io
import numpy as np
from scipy import signal
import yasa
from event_schema import validate_arrays


def snapshot(path,expected):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=256*1024*1024:raise ValueError('bounded private derivative required')
        data=b'';offset=0
        while offset<before.st_size:
            b=os.pread(fd,min(1048576,before.st_size-offset),offset)
            if not b:raise ValueError('truncated derivative')
            data+=b;offset+=len(b)
        after=os.fstat(fd);current=os.stat(path,follow_symlinks=False)
        stamp=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if stamp(before)!=stamp(after) or stamp(before)!=stamp(current) or hashlib.sha256(data).hexdigest()!=expected:raise ValueError('derivative identity changed')
        with np.load(io.BytesIO(data),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    finally:os.close(fd)


def intervals(mask,fs,offset=0.,minimum=0.,maximum=float('inf')):
    a=np.asarray(mask,dtype=bool);changes=np.diff(np.r_[False,a,False].astype(int));starts=np.flatnonzero(changes==1);stops=np.flatnonzero(changes==-1)
    return [{'start':float(x/fs+offset),'end':float(y/fs+offset),'duration':float((y-x)/fs)} for x,y in zip(starts,stops) if minimum<=(y-x)/fs<=maximum]


def finite_row(row,offset):
    result={str(k):float(v) if isinstance(v,(int,float,np.number)) and np.isfinite(v) else str(v) if isinstance(v,str) else None for k,v in row.items()}
    result['start']=float(row['Start']+offset);result['end']=float(row['End']+offset);result['duration']=result['end']-result['start']
    return result


def detect_block(arrays,config,start,core_start,core_stop,duration):
    validate_arrays(arrays)
    c=config['detector'];events={};failures=[];statuses={};runtime={}
    lo=max(10,core_start);hi=min(duration-10,core_stop)
    for channel in ('EEG Fpz-Cz','EEG Pz-Oz'):
        x=np.asarray(arrays[channel],dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError('nonfinite detector input')
        for kind in ('spindle','spindle_strict','slow_wave','slow_wave_relaxed'):
            t=time.perf_counter();key=channel+'|'+kind
            try:
                if kind.startswith('spindle'):
                    params={**c['spindle'],'thresh':dict(c['spindle']['thresh'])}
                    if kind=='spindle_strict':params['thresh']['rms']=c['spindle_sensitivity_rms']
                    output=yasa.spindles_detect(x,sf=100,ch_names=[channel],hypno=None,verbose='error',**params)
                else:
                    params=dict(c['slow_wave'])
                    if kind=='slow_wave_relaxed':params.update(c['slow_wave_relaxed'])
                    output=yasa.sw_detect(x,sf=100,ch_names=[channel],hypno=None,verbose='error',**params)
                found=[] if output is None else [finite_row(row,start) for row in output.summary().to_dict('records')]
                events[key]=[r for r in found if lo<=(r['start']+r['end'])/2<hi and r['start']>=10 and r['end']<=duration-10]
                statuses[key]='UNASSESSABLE_FLAT' if np.std(x)<1e-8 else 'NO_EVENTS' if output is None else 'SUCCESS'
            except Exception as exc:
                events[key]=[];statuses[key]='DETECTOR_FAILED';failures.append({'method':key,'type':type(exc).__name__,'reason':str(exc)[:300]})
            runtime[key]=time.perf_counter()-t
        n=len(x)//200;a=x[:n*200].reshape(n,200);power=abs(np.fft.rfft(a-a.mean(1,keepdims=True),axis=1))**2;freq=np.fft.rfftfreq(200,.01);total=power[:,(freq>=.5)&(freq<=45)].sum(1).clip(1e-12)
        for name,(low,high,threshold) in c['spectral_controls'].items():
            value=power[:,(freq>=low)&(freq<=high)].sum(1)/total
            found=intervals(value>=threshold,.5,start)
            events[channel+'|'+name]=[r for r in found if lo<=(r['start']+r['end'])/2<hi]
        for name,mask,minimum in [('flat',np.r_[False,np.abs(np.diff(x))<1e-6],1.),('step',np.r_[False,np.abs(np.diff(x))>100],.01),('extreme',np.abs(x)>500,.01)]:
            found=intervals(mask,100,start,minimum)
            events[channel+'|'+name]=[r for r in found if lo<=(r['start']+r['end'])/2<hi]
        bounds=np.asarray(arrays[channel+'__bounds']);clipped=(x<=bounds[0])|(x>=bounds[1])
        events[channel+'|clip']=[r for r in intervals(clipped,100,start,.01) if lo<=(r['start']+r['end'])/2<hi]
    eog=np.asarray(arrays['EOG horizontal'],float);sos=signal.butter(4,c['EOG']['band_Hz'],btype='bandpass',fs=100,output='sos');filtered=signal.sosfiltfilt(sos,eog)
    events['EOG horizontal|excursion']=[r for r in intervals(abs(filtered)>=50,100,start,.1,3) if lo<=(r['start']+r['end'])/2<hi]
    fs=float(arrays['EMG submental__fs']);emg=np.asarray(arrays['EMG submental'],float)
    if fs==100:
        sos=signal.butter(4,[20,45],btype='bandpass',fs=100,output='sos');v=signal.sosfiltfilt(sos,emg);a=v[:len(v)//100*100].reshape(-1,100);envelope=np.sqrt((a*a).mean(1));definition='20-45Hz1secondRMS'
    elif fs==1:envelope=emg;definition='native processed envelope, acquisition transfer unknown'
    else:raise ValueError('unsupported native EMG rate')
    for name,mask in [('low',envelope<3),('high',envelope>20)]:
        events['EMG submental|'+name]=[r for r in intervals(mask,1,start,1.) if lo<=(r['start']+r['end'])/2<hi]
    # Cross-channel coherence is independently computed, descriptive, not an event label.
    f,coh=signal.coherence(arrays['EEG Fpz-Cz'],arrays['EEG Pz-Oz'],fs=100,nperseg=400)
    return {'events':events,'statuses':statuses,'failures':failures,'detector_seconds':runtime,'eligible_seconds':max(0,hi-lo),
            'EMG_definition':definition,'coherence_delta':float(np.mean(coh[(f>=.5)&(f<=4)])) if np.isfinite(coh[(f>=.5)&(f<=4)]).all() else None,
            'coherence_sigma':float(np.mean(coh[(f>=12)&(f<=15)])) if np.isfinite(coh[(f>=12)&(f<=15)]).all() else None,'YASA_version':yasa.__version__,'hypnogram_used':False,'pretrained_weights':False}


def main():
    for line in sys.stdin:
        request=json.loads(line)
        try:
            arrays=snapshot(request['input'],request['input_sha256'])
            result=detect_block(arrays,request['config'],request['start'],request['core_start'],request['core_stop'],request['duration'])
            fd=os.open(request['output'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as stream:json.dump(result,stream,allow_nan=False)
            print(json.dumps({'status':'SUCCESS','output_sha256':hashlib.sha256(open(request['output'],'rb').read()).hexdigest()}),flush=True)
        except Exception as exc:print(json.dumps({'status':'FAILED','reason':type(exc).__name__+': '+str(exc)}),flush=True)

if __name__=='__main__':main()

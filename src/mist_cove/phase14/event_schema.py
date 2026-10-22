"""Explicit proxy timing and input schema; never expert-confirmed event truth."""
import numpy as np


def validate_arrays(arrays):
    required={'EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental','EMG submental__fs','EEG Fpz-Cz__bounds','EEG Pz-Oz__bounds'}
    if set(arrays)!=required:raise ValueError('only approved signal arrays; no stage/prediction inputs')
    n=len(arrays['EEG Fpz-Cz'])
    if n<3000 or any(np.asarray(v).dtype.hasobject or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('finite physical signal arrays required')
    if any(len(arrays[k])!=n for k in ('EEG Pz-Oz','EOG horizontal')):raise ValueError('simultaneous100Hz channels required')
    fs=float(arrays['EMG submental__fs'])
    if fs not in (1,100) or len(arrays['EMG submental'])!=n/100*fs:raise ValueError('nativeEMG sampling mismatch')
    for key in ('EEG Fpz-Cz__bounds','EEG Pz-Oz__bounds'):
        if np.shape(arrays[key])!=(2,) or arrays[key][1]<=arrays[key][0]:raise ValueError('calibration bounds required')


def event_intervals(events,epoch,duration=30):
    result=[];start=epoch*duration;stop=start+duration
    for e in events:
        a,b=e['start'],e['end']
        if not np.isfinite([a,b]).all() or not 0<=a<b:raise ValueError('malformed global event timestamps')
        if a<stop and b>start:result.append([max(0,int(round((a-start)*100))),min(3000,int(round((b-start)*100)))])
    return [[a,b] for a,b in result if a<b]


def validate_record_events(events,record_duration):
    for row in events:
        if not 0<=row['start']<row['end']<=record_duration:raise ValueError('event outside its recording')
    return events


def shifted_intervals(intervals,seconds=10):
    output=[]
    for a,b in intervals:
        duration=b-a;a=(a+round(seconds*100))%3000;b=a+duration
        if b<=3000:output.append([a,b])
        else:output.extend([[a,3000],[0,b-3000]])
    return output

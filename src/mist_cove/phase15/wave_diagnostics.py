"""Bounded approved ST read pass for REM waveform diagnostics and panel crops."""
import json
import time
from collections import defaultdict

import numpy as np

from mist_cove.phase14.edf_reader import MultiReader
from .forensics import _stratum, _summary, collect
from .governance import PRIVATE, PUBLIC, OLD_PRIVATE, LABEL, read, new, sha, canonical, verify_plan_commit


CHANNELS=('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental')


def _require_st_rate(rate):
    if rate!=100:raise ValueError('ST native rate changed')


def _wave_features(x, fs=100):
    x=np.asarray(x,dtype=np.float64)
    centered=x-x.mean(1,keepdims=True)
    power=np.abs(np.fft.rfft(centered,axis=1))**2
    freq=np.fft.rfftfreq(x.shape[1],1/fs)
    valid=(freq>=.5)&(freq<=45)
    p=power[:,valid];denom=p.sum(1).clip(1e-12);prob=p/denom[:,None]
    entropy=-(prob*np.log(prob.clip(1e-30))).sum(1)/np.log(p.shape[1])
    short=np.std(centered.reshape(len(x),6,-1),axis=2)
    robust=np.median(abs(np.diff(x,axis=1)),axis=1).clip(1e-6)
    transients=(abs(np.diff(x,axis=1))>5*robust[:,None]).sum(1)
    def band(a,b):return power[:,(freq>=a)&(freq<=b)].sum(1)/denom
    return {'spectral_entropy':entropy,'short_RMS_CV':short.std(1)/short.mean(1).clip(1e-12),
            'broadband_diff5_median_count':transients,'RMS_uV':centered.std(1),
            'delta_fraction':band(.5,4),'theta_fraction':band(4,7),'alpha_fraction':band(8,13),
            'sigma_fraction':band(12,15),'HF_fraction':band(30,45)}


def run():
    verify_plan_commit()
    started=time.perf_counter()
    panel=read(PRIVATE/'frozen_panels.json')
    if sha(canonical(panel))!=read(PUBLIC/'panel_seal.json')['private_panel_canonical_sha256']:
        raise ValueError('frozen private panel identity changed')
    coords={(c['recording'],c['epoch']) for name in ('E1','E3') for c in panel[name]}
    if (PRIVATE/'ST_panel_waves.npz').exists():raise ValueError('ST panel waveform cache already exists')
    rows,_,_=collect();selected=defaultdict(list)
    for r in rows:
        if (r['recording'],r['epoch']) in coords:selected[r['recording']].append(r['epoch'])
    source=MultiReader();records=source.records('ST_EVAL')
    if len(records)!=44:raise ValueError('ST source role count changed')
    record_rows=defaultdict(list)
    for r in rows:
        if r['truth']==4:record_rows[r['recording']].append(r)
    private_features=[];crops={k:[] for k in CHANNELS};crop_coords=[]
    for j,record in enumerate(records):
        rec=record['recording'];rr=record_rows[rec]
        if not rr:raise ValueError('missing REM record')
        event=read(OLD_PRIVATE/'ST_EVAL'/(rec+'_events.json'))
        n=event['n_epochs'];duration=event['duration'];waves={}
        with source.open(record,values=True) as (reader,_):
            for channel in CHANNELS:
                values,identity=source.read(record,reader,channel,0,duration)
                _require_st_rate(identity['native_fs'])
                if len(values)<n*3000:raise ValueError('ST channel physical grid truncated')
                waves[channel]=np.asarray(values[:n*3000],dtype=np.float32).reshape(n,3000)
        indices=np.array([r['epoch'] for r in rr],dtype=int)
        feature_by_channel={ch:_wave_features(waves[ch][indices]) for ch in CHANNELS[:3]}
        eog=event['events'].get('EOG horizontal|excursion',[])
        emg_rms=np.sqrt(np.mean(waves['EMG submental']**2,axis=1))
        rank=np.argsort(np.argsort(emg_rms,kind='stable'),kind='stable')/(len(emg_rms)-1)
        for z,r in enumerate(rr):
            i=r['epoch'];lo=i*30;hi=(i+1)*30
            excursions=[a for a in eog if a['start']<hi and a['end']>lo]
            private_features.append({'recording':rec,'epoch':i,'person':r['person'],'night':r['night'],
               'stratum':_stratum(r),'features':{
                 ch:{k:float(v[z]) for k,v in feat.items()} for ch,feat in feature_by_channel.items()},
               'EMG_within_night_RMS_rank':float(rank[i]),
               'EOG_excursion_count':len(excursions),
               'EOG_excursion_total_seconds':float(sum(max(0,min(hi,a['end'])-max(lo,a['start'])) for a in excursions))})
        for epoch in selected[rec]:
            crop_coords.append((rec,epoch))
            for ch in CHANNELS:crops[ch].append(waves[ch][epoch].copy())
        print(json.dumps({'ST_records_read':j+1,'REM_epochs_processed':len(private_features),
                          'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
    if len(crop_coords)!=len(coords):raise ValueError('selected crop count differs')
    np.savez_compressed(PRIVATE/'ST_panel_waves.npz',**{ch:np.stack(values) for ch,values in crops.items()})
    new(PRIVATE/'ST_panel_wave_coordinates.json',{'coordinates':crop_coords,'panel_sha256':sha(canonical(panel))})
    new(PRIVATE/'ST_REM_wave_features.json',{'rows':private_features})
    groups=defaultdict(list)
    for x in private_features:groups[x['stratum']].append(x)
    summary={}
    for s,rr in groups.items():
        keys=list(rr[0]['features']['EEG Fpz-Cz'])
        summary[s]={'n':len(rr),'EEG_Fpz_Cz':{k:_summary([r['features']['EEG Fpz-Cz'][k] for r in rr]) for k in keys},
                    'EEG_Pz_Oz':{k:_summary([r['features']['EEG Pz-Oz'][k] for r in rr]) for k in keys},
                    'EOG':{'excursion_count':_summary([r['EOG_excursion_count'] for r in rr]),
                           'excursion_seconds':_summary([r['EOG_excursion_total_seconds'] for r in rr]),
                           'low_frequency_fraction':_summary([r['features']['EOG horizontal']['delta_fraction'] for r in rr]),
                           'short_variability':_summary([r['features']['EOG horizontal']['short_RMS_CV'] for r in rr])},
                    'ST_EMG':{'within_night_rank':_summary([r['EMG_within_night_RMS_rank'] for r in rr])}}
    elapsed=time.perf_counter()-started
    new(PUBLIC/'rem_wave_diagnostics.json',{'status':LABEL,'ST_records_opened':44,
        'REM_epochs_processed':len(private_features),'frozen_panel_crops':len(crop_coords),
        'channels':list(CHANNELS),'native_Hz':100,'summary':summary,
        'source_access_counts':source.access.counts,'scientific_CPU_wall_seconds':elapsed,
        'private_wave_crop_sha256':sha((PRIVATE/'ST_panel_waves.npz').read_bytes()),
        'private_wave_coordinate_sha256':sha(canonical({'coordinates':crop_coords,'panel_sha256':sha(canonical(panel))})),
        'human_event_validation':'NOT_TESTED','interpretation':'exploratory associations, no label adjudication'})
    source.access.close()
    print(json.dumps({'ST_records':44,'REM_epochs':len(private_features),'panel_crops':len(crop_coords),
                      'elapsed_seconds':elapsed}))


if __name__=='__main__':run()

"""Physical-unit epoch statistics and continuous-segment timing controls."""
import numpy as np
from scipy import signal
FEATURES=('absolute_peak_uV','centered_RMS_uV','DC_uV','delta_fraction','theta_fraction','alpha_fraction','sigma_fraction','HF_fraction','EOG_RMS_uV','EMG_native_RMS_uV','EMG_derived_RMS_uV','flat_fraction','step_fraction','EEG_correlation','marker_change_fraction')


def epoch_features(channels,emg_fs,marker_fs,n):
    x=np.asarray(channels['EEG Fpz-Cz'][:n*3000]).reshape(n,3000);z=np.asarray(channels['EEG Pz-Oz'][:n*3000]).reshape(n,3000);e=np.asarray(channels['EOG horizontal'][:n*3000]).reshape(n,3000)
    power=abs(np.fft.rfft(x-x.mean(1,keepdims=True),axis=1))**2;f=np.fft.rfftfreq(3000,.01);total=power[:,(f>=.5)&(f<=45)].sum(1).clip(1e-12)
    bands=[power[:,(f>=a)&(f<=b)].sum(1)/total for a,b in [(.5,4),(4,7),(8,13),(12,15),(30,45)]]
    emg=channels['EMG submental'];a=emg[:int(n*30*emg_fs)].reshape(n,int(30*emg_fs));native=np.sqrt((a*a).mean(1))
    if emg_fs==100:
        filtered=signal.sosfiltfilt(signal.butter(4,[20,45],btype='bandpass',fs=100,output='sos'),emg)
        derived=np.sqrt((filtered[:n*3000].reshape(n,3000)**2).mean(1))
    else:derived=native.copy()
    x0=x-x.mean(1,keepdims=True);z0=z-z.mean(1,keepdims=True);corr=(x0*z0).sum(1)/np.sqrt((x0*x0).sum(1)*(z0*z0).sum(1)).clip(1e-12)
    marker=channels['marker'][:int(n*30*marker_fs)].reshape(n,int(30*marker_fs))
    return np.stack([abs(x).max(1),x.std(1),x.mean(1),*bands,np.sqrt((e*e).mean(1)),native,derived,(abs(np.diff(x,axis=1))<1e-6).mean(1),(abs(np.diff(x,axis=1))>100).mean(1),corr,(np.diff(marker,axis=1)!=0).mean(1)],axis=1)


def segment_bounds(duration,core=1800,padding=30):
    for center in range(0,int(np.ceil(duration)),core):yield max(0,center-padding),min(duration,center+core+padding),center,min(duration,center+core)


def frequency_checks():
    result={}
    for label,band in [('EOG',[.3,5]),('ST_EMG',[20,45])]:
        sos=signal.butter(4,band,btype='bandpass',fs=100,output='sos');f,h=signal.sosfreqz(sos,worN=10000,fs=100)
        middle=(band[0]+band[1])/2;value=float(abs(h[np.argmin(abs(f-middle))]));
        if value<.9:raise ValueError('passband response failed')
        result[label]={'midband_gain':value,'native_Hz':100,'zero_phase':True,'boundary_padding_seconds':30}
    return result

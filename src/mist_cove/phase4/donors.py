"""Separate TRAIN-derived compatibility proposal; pure arrays, no file access.

Passing this numerical envelope does not certify physiological counterfactuals.
Real fitting requires a separately authorized protocol; this phase tests mocks.
"""
from dataclasses import dataclass
import numpy as np


def features(waves):
    v=np.asarray(waves,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]<40 or not np.isfinite(v).all():raise ValueError('finite calibrated patch matrix required')
    centered=v-v.mean(1,keepdims=True)
    rms=np.sqrt(np.mean(centered**2,1))
    slopes=np.sqrt(np.mean(np.diff(v,axis=1)**2,1))
    power=np.abs(np.fft.rfft(centered,axis=1))**2
    high=power[:,power.shape[1]//2:].sum(1)/np.maximum(power.sum(1),1e-12)
    return np.stack((np.log1p(rms),np.log1p(slopes),high),1)


@dataclass
class Compatibility:
    center: np.ndarray
    scale: np.ndarray
    cutoff: float
    provenance: dict

    def accepts(self,wave):
        value=features(np.asarray(wave)[None])[0]
        return bool(np.max(np.abs((value-self.center)/self.scale))<=self.cutoff)


def fit_compatibility(fit_waves,calibration_waves,*,role,provenance):
    # Role and proof rejection happen before either array is inspected.
    if role!='train' or provenance.get('scope')!='SYNTHETIC_TRAIN_FIXTURE':
        raise ValueError('Phase IV permits only synthetic TRAIN compatibility fitting')
    if not provenance.get('fit_calibration_disjoint') or not provenance.get('source_sha256'):
        raise ValueError('explicit disjoint TRAIN fit/calibration identity required')
    fit=features(fit_waves);cal=features(calibration_waves)
    if len(fit)<20 or len(cal)<20:raise ValueError('insufficient compatibility calibration')
    center=np.median(fit,axis=0);scale=np.maximum(np.median(np.abs(fit-center),axis=0),1e-6)
    scores=np.max(np.abs((cal-center)/scale),axis=1)
    cutoff=float(np.quantile(scores,.95,method='higher'))
    return Compatibility(center,scale,cutoff,dict(provenance))

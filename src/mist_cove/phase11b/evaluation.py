"""Fixed-five descriptive metrics and participant-clustered uncertainty."""
import numpy as np
from scipy.special import softmax,logsumexp
from mist_transfer.metrics import classification_metrics

def distribution(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),
            'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75))} if len(x) else {'n':0}

def clustered_mean(values,subjects):
    values=np.asarray(values,dtype=float);subjects=np.asarray(subjects)
    units=np.array([values[subjects==s].mean() for s in sorted(set(subjects))])
    if not len(units):return {'n_participants':0,'mean':None,'bootstrap95':None}
    rng=np.random.default_rng(123);draws=units[rng.integers(0,len(units),(1000,len(units)))].mean(1)
    return {'n_participants':len(units),'mean':float(units.mean()),'bootstrap95':np.quantile(draws,[.025,.975]).tolist(),
            'unit':'participant mean; paired views retained within participant','draws':1000,'seed':123}

def probability_metrics(logits,truth):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(truth,dtype=int);p=softmax(z,axis=-1)
    confidence=p.max(-1);correct=p.argmax(-1)==y;ece=0
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        selected=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
        if selected.any():ece+=selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {'NLL':float(np.mean(logsumexp(z,axis=-1)-z[np.arange(len(y)),y])),
            'Brier':float(np.mean(np.sum((p-np.eye(5)[y])**2,axis=1))),'ECE15':float(ece),
            'mean_confidence_correct':float(confidence[correct].mean()) if correct.any() else None,
            'mean_confidence_wrong':float(confidence[~correct].mean()) if (~correct).any() else None,
            'high_confidence_wrong_count':int(((confidence>=.9)&~correct).sum()),
            'high_confidence_wrong_fraction_all':float(((confidence>=.9)&~correct).mean()),
            'high_confidence_wrong_fraction_errors':float((confidence[~correct]>=.9).mean()) if (~correct).any() else None}

def public_metrics(y,logits,subjects,records=None):
    y=np.asarray(y);z=np.asarray(logits);s=np.asarray(subjects);mask=y>=0
    result=classification_metrics(y,z.argmax(-1),s,mask)
    values=result.pop('subject_metrics');f1=result.pop('subject_macro_f1')
    result['subject_macro_f1_distribution']=distribution(list(f1.values()))
    result['mean_subject_macro_f1_bootstrap95']=clustered_mean(np.array(list(f1.values())),np.arange(len(f1)))
    result['calibration']=probability_metrics(z[mask],y[mask])
    result['stage_mean_subject_f1']=[float(np.mean([m['per_stage_f1'][i] for m in values.values()])) for i in range(5)]
    if records is not None:
        r=np.asarray(records);record_f1=[]
        for rid in sorted(set(r)):
            sel=(r==rid)&mask
            if sel.any():record_f1.append(classification_metrics(y[sel],z[sel].argmax(-1),np.zeros(sel.sum(),int))['macro_f1'])
        result['recording_macro_f1_distribution']=distribution(record_f1)
    return result

QUALITY_NAMES=('absolute_peak_uV','centered_rms_uV','DC_uV','low_frequency_fraction','high_frequency_fraction','broadband_step_rms_uV','flat_fraction','largest_step_uV','atom_rms_max_median_ratio')

def signal_quality(wave):
    x=np.asarray(wave,dtype=np.float64).reshape(-1,3000);center=x-x.mean(1,keepdims=True)
    power=abs(np.fft.rfft(center,axis=1))**2;freq=np.fft.rfftfreq(3000,.01);total=power.sum(1).clip(1e-12)
    dx=np.diff(x,axis=1);atoms=x.reshape(-1,15,200);rms=np.std(atoms,axis=2)
    return np.stack((abs(x).max(1),np.std(x,axis=1),x.mean(1),power[:,freq<.5].sum(1)/total,
                     power[:,freq>=30].sum(1)/total,np.sqrt(np.mean(dx**2,axis=1)),(abs(dx)<1e-6).mean(1),
                     abs(dx).max(1),rms.max(1)/np.median(rms,axis=1).clip(1e-8)),axis=1)

def quality_associations(y,z,subjects,quality,boundary):
    m=y>=0;correct=z.argmax(-1)==y;rows={}
    for j,name in enumerate(QUALITY_NAMES):
        q=quality[:,j];diff=[];unit=[]
        for subject in sorted(set(subjects)):
            sel=m&(subjects==subject);a=sel&correct;b=sel&~correct
            if a.any() and b.any():diff.append(q[b].mean()-q[a].mean());unit.append(subject)
        rows[name]={'correct':distribution(q[m&correct]),'wrong':distribution(q[m&~correct]),
                    'participant_wrong_minus_correct':clustered_mean(diff,unit)}
    rows['record_boundary']={'n':int((m&boundary).sum()),'error_rate':float((~correct[m&boundary]).mean()) if (m&boundary).any() else None}
    return rows

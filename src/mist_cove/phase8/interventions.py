"""Independent TRAIN-derived synthetic donor audit; historical guards untouched."""
import numpy as np
import torch
from mist_cove.phase4.donors import fit_compatibility
from mist_cove.phase6.generator import generate
from mist_cove.phase5.runtime import predict
from mist_evidence.runtime import tensor_digest


def distortions(original,donor):
    a=original.double().numpy();b=donor.double().numpy()
    ea=np.mean(a*a);eb=np.mean(b*b)
    pa=np.abs(np.fft.rfft(a-a.mean()))**2;pb=np.abs(np.fft.rfft(b-b.mean()))**2
    pa/=max(pa.sum(),1e-12);pb/=max(pb.sum(),1e-12)
    return {'energy_relative':float(abs(ea-eb)/max(ea,1e-8)),
        'spectral_l1':float(np.abs(pa-pb).sum()),
        'boundary_max':float(max(abs(a[0]-b[0]),abs(a[-1]-b[-1])))}


def donor_audit(model,task,seed,c,budget):
    # Independent generated TRAIN and calibration populations; no clinical data.
    train=generate(100,seed+c['donor_train_offset'],task)
    cal=generate(100,seed+c['donor_calibration_offset'],task)
    bank=train.base[:,0,:200].clone();cb=cal.base[:,0,:200].clone()
    envelope=fit_compatibility(bank.numpy(),cb.numpy(),role='train',provenance={
        'scope':'SYNTHETIC_TRAIN_FIXTURE','fit_calibration_disjoint':True,'source_sha256':tensor_digest({'train_bank':bank,'calibration_bank':cb})})
    draw=generate(50,seed+c['donor_eval_offset'],task)
    counts={k:{'attempted':0,'accepted':0,'rejections':{'envelope':0,'energy':0,'spectrum':0,'boundary':0},'distortions':[]} for k in ('train_baseline','bridge','oracle_neutral')}
    disagreements=[];spreads=[];covered=0
    for i,truth in enumerate(draw.intervals):
        lo,hi=truth[0];target=draw.x[i,0,lo:hi];source=draw.x[i:i+1]
        with torch.no_grad():original=model(source)[0]
        # Fixed TRAIN bank lookup by energy only; no model score or class is used.
        energy=target.square().mean();order=(bank.square().mean(1)-energy).abs().argsort(stable=True)[:3]
        candidates=[('train_baseline',bank[j]) for j in order]
        candidates += [('bridge',torch.linspace(float(target[0]),float(target[-1]),200)),
                       ('oracle_neutral',draw.base[i,0,lo:hi])]
        accepted=[]
        for name,donor in candidates:
            budget.check();v=counts[name];v['attempted']+=1;d=distortions(target,donor)
            reasons={'envelope':not envelope.accepts(donor.numpy()),'energy':d['energy_relative']>.5,
                'spectrum':d['spectral_l1']>.5,'boundary':d['boundary_max']>30.}
            for k,reject in reasons.items():v['rejections'][k]+=int(reject)
            v['distortions'].append(d)
            if any(reasons.values()):continue
            changed=source.clone();changed[0,0,lo:hi]=donor
            with torch.no_grad():after=model(changed)[0]
            v['accepted']+=1;accepted.append((after,float(original[draw.y[i]]-after[draw.y[i]])))
        if accepted:covered+=1
        if len(accepted)>=2:
            pred=[int(a.argmax()) for a,e in accepted];disagreements.append(len(set(pred))>1)
            effects=[e for a,e in accepted];spreads.append(max(effects)-min(effects))
    for values in counts.values():
        values['mean_distortions']={k:float(np.mean([d[k] for d in values['distortions']])) for k in ('energy_relative','spectral_l1','boundary_max')}
        del values['distortions']
    return {'population':counts,'evaluation_examples':50,'accepted_coverage':covered/50,
        'multiple_accepted_examples':len(spreads),'conditional_prediction_disagreement':float(np.mean(disagreements)) if disagreements else None,
        'mean_effect_spread':float(np.mean(spreads)) if spreads else None,
        'accepted_donor_robustness':'DESCRIPTIVE_SYNTHETIC_ONLY' if spreads else 'NOT TESTED',
        'neutrality':'TRAIN baseline and oracle-neutral populations contain no generated diagnostic component; bridge neutrality unverified',
        'overlap_policy':'single first diagnostic interval only; no union geometry confound',
        'clinical_validity':'NOT TESTED','envelope_center':envelope.center.tolist(),'envelope_scale':envelope.scale.tolist(),'cutoff':envelope.cutoff}

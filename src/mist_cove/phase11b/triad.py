"""Finite structural claims, measured waveform probes, and fixed-target stability.

No claim in this module identifies a physiological event or preserves its label.
"""
import numpy as np
import torch
from mist_cove.phase9.clauses import verify_clause

VIEW_NAMES=('mild_hf','drift','gain')
PROBE_NAMES=('identity','mild_hf','strong_hf','drift','transient','broadband','gain','phase')

def waveform_probes(wave,seed):
    x=np.asarray(wave,dtype=np.float32).reshape(3000);rng=np.random.default_rng(seed)
    centered=x-x.mean();rms=max(float(np.sqrt(np.mean(centered**2))),1e-8);t=np.arange(3000)/100
    phase=rng.uniform(0,2*np.pi,3)
    spectrum=np.fft.rfft(centered);rotation=rng.uniform(-np.pi,np.pi,len(spectrum));rotation[[0,-1]]=0
    phase_wave=np.fft.irfft(spectrum*np.exp(1j*rotation),n=3000)+x.mean()
    values=[x.copy(),x+.05*rms*np.sin(2*np.pi*32*t+phase[0]),
            x+.5*rms*np.sin(2*np.pi*35*t+phase[1]),x+.1*rms*np.sin(2*np.pi*.2*t+phase[2]),
            x+3*rms*np.exp(-.5*((np.arange(3000)-1500)/10)**2),
            x+.3*rms*rng.standard_normal(3000),x*1.05,phase_wave]
    return {name:np.asarray(v,dtype=np.float32) for name,v in zip(PROBE_NAMES,values)}

def distortion(original,changed,start=0,stop=3000):
    x=np.asarray(original,dtype=np.float64).reshape(3000);z=np.asarray(changed,dtype=np.float64).reshape(3000)
    target=x[start:stop];replacement=z[start:stop];outside=np.ones(3000,bool);outside[start:stop]=False
    energy=float(abs(np.sum(replacement**2)-np.sum(target**2))/max(np.sum(target**2),1e-12))
    a=np.abs(np.fft.rfft(target));b=np.abs(np.fft.rfft(replacement))
    spectral=float(np.abs(a-b).sum()/max(a.sum(),1e-12))
    jumps=[]
    if start:jumps.append(abs(float(z[start]-z[start-1])))
    if stop<3000:jumps.append(abs(float(z[stop]-z[stop-1])))
    boundary=max(jumps,default=0.)
    return {'target_rms_difference_uV':float(np.sqrt(np.mean((target-replacement)**2))),
            'nontarget_exact':bool(np.array_equal(x[outside],z[outside])),
            'boundary_jump_uV':boundary,'relative_energy_distortion':energy,'spectral_magnitude_L1':spectral,
            'historical_compatibility':bool(energy<=.5 and spectral<=.5 and boundary<=5)}

def replacements(wave):
    x=np.asarray(wave,dtype=np.float32).reshape(3000);atoms=x.reshape(15,200)
    rms=np.sqrt(np.mean((atoms-atoms.mean(1,keepdims=True))**2,axis=1));values=[];metadata=[]
    for atom in range(15):
        start=atom*200;stop=start+200
        bridge=x.copy();left=x[start-1] if start else x[stop];right=x[stop] if stop<3000 else x[start-1]
        bridge[start:stop]=np.linspace(left,right,202,dtype=np.float32)[1:-1]
        candidates=[j for j in range(15) if j!=atom and rms[j]<=3]
        donor=min(candidates,key=lambda j:(float(rms[j]),j)) if candidates else None
        for kind,z in [('bridge',bridge),('background',None if donor is None else x.copy())]:
            if z is None:
                metadata.append({'kind':kind,'atom':atom,'status':'NO_ELIGIBLE_BACKGROUND','donor':None,'output_index':None});continue
            if kind=='background':z[start:stop]=atoms[donor]
            row={'kind':kind,'atom':atom,'status':'ATTEMPTED','donor':donor if kind=='background' else None,
                 'output_index':len(values),**distortion(x,z,start,stop)}
            values.append(z);metadata.append(row)
    return np.asarray(values,dtype=np.float32)[:,None],metadata

def exact_support(truth):
    truth=np.asarray(truth,dtype=bool)
    if truth.shape!=(32768,):raise ValueError('complete fifteen-atom table required')
    universal=truth.copy()
    for bit in range(15):
        view=universal.reshape(-1,2,1<<bit);view[:,0]&=view[:,1]
    def smallest(values):
        indices=np.flatnonzero(values)
        return min((int(i) for i in indices),key=lambda i:(i.bit_count(),i)) if len(indices) else None
    return {'canonical_minimum_mask':smallest(truth),'positive_minimum_mask':smallest(universal),
            'prior_response':bool(truth[0]),'universal_prior':bool(truth.all())}

def fixed_clause_stability(summary,view_truth,view_prediction,target):
    clauses=summary['signed']['clauses'];active=[c for c in clauses if c['negative_bits']==0]
    primary=active[0] if active else None
    validity=[verify_clause(view_truth,c['positive_bits'],c['negative_bits']) for c in clauses]
    fixed=None if primary is None else verify_clause(view_truth,primary['positive_bits'],0)
    # An alternative supports the ORIGINAL full prediction only if it has no
    # withheld literals. Counterfactual-state clauses cannot count as substitutes.
    other_active=any(ok and c['negative_bits']==0 for c,ok in zip(clauses,validity))
    return {'prediction_stable':view_prediction==target,'fixed_clause_valid':fixed,
            'any_original_clause_valid':any(validity),'any_original_active_clause_valid':other_active,
            'active_substitution':bool(fixed is False and other_active),'clause_validity':validity}

def triangulation_graph(summary,truth,replacement_rows,stability,neutral_truth,confidence):
    clauses=summary['signed']['clauses'];edges=[]
    for i,c in enumerate(clauses):
        edges.append({'type':'JOINT_SUFFICIENT_CLAUSE','clause':i,'positive':c['positive'],'negative':c['negative'],
                      'domain':'INTERNAL_ALL_COMPLETIONS','active_at_original':c['negative_bits']==0})
        for j in c['negative']:edges.append({'type':'INTERFERING_LITERAL','atom':j,'clause':i,'domain':'SIGNED_SUBCUBE'})
    for i in range(len(clauses)):
        for j in range(i):edges.append({'type':'ALTERNATIVE_CLAUSES','clauses':[j,i]})
    disagreements=0;valid=0
    for row in replacement_rows:
        if row['status']!='ATTEMPTED' or not row['historical_compatibility']:continue
        atom=row['atom'];structural=not bool(truth[32767^(1<<atom)]);waveform=not row['target_retained']
        valid+=1;disagreements+=int(structural!=waveform)
        edges.append({'type':'OPERATOR_COMPARISON','atom':atom,'operator':row['kind'],
                      'structural_class_loss':structural,'waveform_class_loss':waveform,'disagrees':structural!=waveform})
    return {'nodes':15,'edges':edges,'compatible_comparisons':valid,'operator_disagreements':disagreements,
            'stability':stability,'neutral_table_disagreement_fraction':float(np.mean(truth!=neutral_truth)),
            'native_confidence':confidence,'clinical_causality_claim':False}

def feature_rankings(adapter,cache,target):
    features=adapter.features(cache,torch.ones(1,15,dtype=torch.bool)).features[0].detach().requires_grad_(True)
    logits=adapter.model.emissions(features[None])[0,0]
    rivals=logits.detach().clone();rivals[target]=-torch.inf;competitor=int(rivals.argmax())
    gradient=torch.autograd.grad(logits[target]-logits[competitor],features)[0][0]
    result={}
    for name,importance in [('native_gradient_times_feature',gradient*features.detach()[0]),('absolute_gradient',gradient.abs())]:
        scores=np.zeros(15)
        for scale,embedding in zip(adapter.graph['scales'],cache.local_embeddings):
            p=embedding.shape[1];d=embedding.shape[2];offset=scale['feature_start'];maximum=embedding[0].argmax(0)
            for wi,window in enumerate(scale['windows']):
                # Fixed projection: distribute mean feature equally and max
                # feature to its first argmax window, then equally across atoms.
                value=importance[offset:offset+d].sum()/p+importance[offset+d:offset+2*d][maximum==wi].sum()
                for atom in window['atoms']:scores[atom]+=float(value)/len(window['atoms'])
        result[name]=sorted(range(15),key=lambda j:(-scores[j],j))
    return result

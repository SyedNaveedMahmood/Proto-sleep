"""Frozen sequential synthetic audit; never opens EEG or source descriptors."""
import copy
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from torch import nn
from mist_cove.phase2.provenance import environment,file_sha,digest,git_head
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_evidence.runtime import atomic_json,atomic_torch_save,tensor_digest
from .signals import historical,harmonic,harmonic_component,relative_phase_oracle
from .features import restricted,signal_audit,template_diagnostics,patches
from .models import RestrictedProbe,WaveformControl
from .runtime import historical_proofs,frozen_proofs,execution_proofs,private_output,Budget,fit,predict,summarize,chance_reference
from .audits import pairing,witness_audit

CONFIG=Path('configs/mist_cove_phase5_identifiability_v1.json')


def gate_v2(c,budget):
    rows=[]
    ideal=torch.stack([harmonic_component(code,.731,10.) for code in range(5)])[:,None]
    def relative_range(m): return float((m.amax(0)-m.amin(0)).max()/m.max().clamp_min(1e-12))
    magnitude={'epoch_relative_range':relative_range(torch.fft.rfft(ideal.double()).abs()),
        'patch_relative_range':relative_range(torch.fft.rfft(patches(ideal).double()).abs())}
    for seed in c['seeds']:
        for condition in ['clean','competitive']:
            budget.check();data=harmonic(c['eval_examples'],seed+c['eval_seed_offset'],condition)
            from .signals import diagnostic_swap
            paired=diagnostic_swap(data,(data.y.numpy()+1)%5)
            accuracy=float((relative_phase_oracle(data.x)==data.y).float().mean())
            rows.append({'seed':seed,'condition':condition,'oracle_accuracy':accuracy,
                'geometry':list(data.x.shape),'all_finite':bool(torch.isfinite(data.x).all()),
                'paired_nuisance_exact':torch.equal(data.nuisance,paired.nuisance)})
    passed=all(v['oracle_accuracy']>=c['criteria']['new_oracle_accuracy_minimum'] and v['all_finite'] and v['paired_nuisance_exact'] for v in rows) and max(magnitude.values())<=1e-5
    return {'status':'PASS' if passed else 'FAIL','noiseless_magnitude':magnitude,'oracle_rows':rows}


def _save_latents(output,family,seed,condition,data):
    atomic_json({'family':family,'seed':seed,'condition':condition,'synthetic_latents':data.latent},output/f'latents_{family}_{seed}_{condition}.json')


def evaluate_fit(family,seed,name,model,training,evals,c,budget,feature_name=None):
    started=time.perf_counter();conditions={}
    for condition,data in evals.items():
        values=data.x if feature_name is None else restricted(data,feature_name)
        conditions[condition]=summarize(predict(model,values,budget),data.y,c)
    return {'family':family,'seed':seed,'method':name,'training':training,'conditions':conditions,
        'evaluation_seconds':time.perf_counter()-started,'regime':'end_to_end' if feature_name is None else 'restricted_probe',
        'feature_dimensions':None if feature_name is None else values.shape[1]}


def suite(family,c,output,budget,rows,diagnostics,common):
    generator=harmonic if family=='harmonic' else lambda n,s,cond:historical(n,s,family,cond)
    probes=c['new_probes'] if family=='harmonic' else c['historical_probes']
    conditions=c['new_conditions'] if family=='harmonic' else ['clean','competitive']
    for seed in c['seeds']:
        budget.check();train=generator(c['train_examples'],seed+c['train_seed_offset'],'train')
        evals={condition:generator(c['eval_examples'],seed+c['eval_seed_offset'],condition) for condition in conditions}
        for name,data in {'train':train,**evals}.items(): _save_latents(output,family,seed,name,data)
        paired=generator(c['pair_examples'],seed+c['pair_seed_offset'],'clean')
        other=generator(c['pair_examples'],seed+c['pair_seed_offset']+1,'clean')
        diagnostic={'family':family,'seed':seed,'signal':signal_audit(evals['clean']),
            'chance':chance_reference(evals['clean'].y,seed,c)}
        diagnostics.append(diagnostic)
        for name in c['waveform_controls']:
            budget.check();started=time.perf_counter();torch.manual_seed(seed);model=WaveformControl(name)
            training=fit(model,train.x,train.y,c,seed,budget)
            row=evaluate_fit(family,seed,name,model,training,evals,c,budget)
            row['pairing']=pairing(model,paired,other,seed,c,budget)
            # Full-model audits use original-waveform candidates. Coherent and
            # patch-zero are information controls, not physiological traces.
            if name=='original':
                row['witness_audit']=witness_audit(model.core,paired,other,seed)
                if family=='harmonic': common[seed]=copy.deepcopy(model.core.encoder.state_dict())
            row['run_seconds']=time.perf_counter()-started
            rows.append(row);atomic_torch_save(model.state_dict(),output/f'{family}_{seed}_{name}.pt')
            atomic_json({'status':'IN_PROGRESS','rows':rows,'diagnostics':diagnostics},output/'partial.json')
            print(family,seed,name,'clean F1',round(row['conditions']['clean']['macro_f1'],6),flush=True)
        for name in probes:
            budget.check();started=time.perf_counter();features=restricted(train,name)
            torch.manual_seed(seed);model=RestrictedProbe(features,c['probe_hidden'],c['feature_std_floor'])
            y=train.y
            if name=='permuted_patch': y=y[torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+3931))]
            training=fit(model,features,y,c,seed,budget)
            row=evaluate_fit(family,seed,name,model,training,evals,c,budget,name)
            row['run_seconds']=time.perf_counter()-started
            rows.append(row);atomic_torch_save(model.state_dict(),output/f'{family}_{seed}_{name}.pt')
            atomic_json({'status':'IN_PROGRESS','rows':rows,'diagnostics':diagnostics},output/'partial.json')
            print(family,seed,name,'clean accuracy',round(row['conditions']['clean']['accuracy'],6),flush=True)


def historical_decision(rows,c):
    checks=[]
    for family in c['historical_tasks']:
        for probe in c['historical_probes']:
            values=[next(r for r in rows if (r['family'],r['seed'],r['method'])==(family,seed,probe)) for seed in c['seeds']]
            checks.append({'family':family,'probe':probe,
                'leakage_supported_all_seeds':all(r['conditions']['clean']['leakage_lower_bound_pass'] for r in values),
                'equivalent_all_seeds':all(r['conditions']['clean']['restricted_equivalence'] for r in values)})
    leakage=any(r['leakage_supported_all_seeds'] for r in checks if r['probe'] in {'isolated_magnitude','scaled_magnitude','diagnostic_epoch_magnitude','epoch_magnitude','patch_magnitude'})
    transform=[]
    for family in c['historical_tasks']:
        margins=[]
        for seed in c['seeds']:
            value=lambda name:next(r for r in rows if (r['family'],r['seed'],r['method'])==(family,seed,name))['conditions']['clean']['macro_f1']
            margins.append({'seed':seed,'patch_minus_original':value('patch_zero')-value('original'),
                'patch_minus_coherent':value('patch_zero')-value('coherent_zero')})
        transform.append({'family':family,'seed_margins':margins,
            'patch_advantage_all_seeds':all(min(v['patch_minus_original'],v['patch_minus_coherent'])>=c['criteria']['transform_advantage_f1'] for v in margins)})
    return {'historical_phase_isolation':'FAIL' if leakage else 'NOT_ESTABLISHED','probes':checks,'transform_comparison':transform,
            'interpretation':'successful restricted decoding shows available information, not the unique cause of historical optimization differences'}


def representation_decision(rows,c):
    checks=[];k=c['criteria']
    for seed in c['seeds']:
        row=lambda method:next(r for r in rows if (r['family'],r['seed'],r['method'])==('harmonic',seed,method))
        m=row('original');coherent=row('coherent_zero')
        eq={name:all(row(name)['conditions'][cond]['restricted_equivalence'] for cond in ['clean','competitive']) for name in c['new_probes'] if name!='permuted_patch'}
        clauses={'clean_learnability':m['conditions']['clean']['macro_f1']>=k['new_clean_f1_minimum'],
            'competitive_learnability':m['conditions']['competitive']['macro_f1']>=k['new_competitive_f1_minimum'],
            'restricted_equivalence':all(eq.values()),
            'erased_confidence':m['conditions']['erased']['confidence']<=k['new_erased_confidence_maximum'],
            'coherent_erasure_gap':m['conditions']['clean']['macro_f1']-coherent['conditions']['clean']['macro_f1']>=k['transform_advantage_f1'],
            'diagnostic_pair_sensitivity':m['pairing']['diagnostic_change']['mean']>=k['diagnostic_pair_prediction_change_minimum'],
            'nuisance_pair_agreement':m['pairing']['nuisance_agreement']['mean']>=k['nuisance_pair_prediction_agreement_minimum']}
        checks.append({'seed':seed,'clauses':clauses,'restricted_equivalence_by_probe':eq,'pass':all(clauses.values())})
    return {'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL','checks':checks}


class CachedHead(nn.Module):
    def __init__(self,model): super().__init__();self.model=model
    def forward(self,packed): return self.model.decision(packed[:,:,:18],packed[:,:,18])[0]


def evidence_stage(decision,common,c,output,budget,rows):
    if decision['status']!='PASS': return {'status':'NOT_TESTED_V3_FAILED','optimizer_updates':0}
    for seed in c['seeds']:
        train=harmonic(c['train_examples'],seed+c['train_seed_offset'],'train')
        evals={cond:harmonic(c['eval_examples'],seed+c['eval_seed_offset'],cond) for cond in c['new_conditions']}
        for method in ['M1','M2','M3']:
            budget.check();torch.manual_seed(seed+402);model=make_model(method)
            model.encoder.load_state_dict(common[seed]);model.encoder.requires_grad_(False)
            before=tensor_digest(model.encoder.state_dict())
            with torch.no_grad(): f,r=model.encode_candidates(train.x)
            training=fit(CachedHead(model),torch.cat((f,r[:,:,None]),-1),train.y,c,seed,budget,updates=c['head_updates'])
            assert tensor_digest(model.encoder.state_dict())==before
            row=evaluate_fit('harmonic',seed,method,model,training,evals,c,budget)
            paired=harmonic(c['pair_examples'],seed+c['pair_seed_offset'],'clean')
            other=harmonic(c['pair_examples'],seed+c['pair_seed_offset']+1,'clean')
            row.update(regime='fixed_encoder',encoder_sha256=before,witness_audit=witness_audit(model,paired,other,seed))
            rows.append(row);atomic_torch_save(model.state_dict(),output/f'harmonic_{seed}_fixed_{method}.pt')
    # Focal localization is constant by construction for global harmonic support.
    # Do not let a saturated metric become an evidence-organization PASS.
    return {'status':'NOT_ESTABLISHED_LOCALIZATION_NONIDENTIFIABLE','comparisons_executed':True,
        'reason':'Every candidate contains the diagnostic code; a .10 localization gain cannot be identified.'}


def run(output):
    c=json.loads(CONFIG.read_text())
    historical_proof=historical_proofs();prereg=frozen_proofs();execution=execution_proofs()
    if c['scope']!='SYNTHETIC_ONLY' or c['real_data_authorized']: raise ValueError('synthetic scope required')
    subprocess.run(['git','merge-base','--is-ancestor',c['base_commit'],'HEAD'],check=True)
    output=private_output(output,c['scope']);configure_precision('cpu');torch.set_num_threads(c['cpu_threads'])
    budget=Budget(c['cpu_wall_ceiling_seconds']);rows=[];diagnostics=[];common={}
    try:
        for family in c['historical_tasks']: suite(family,c,output,budget,rows,diagnostics,common)
        v1=historical_decision(rows,c);v2={'status':'NOT_REQUIRED'};v3={'status':'NOT_TESTED'};v4={'status':'NOT_TESTED'}
        if v1['historical_phase_isolation']=='FAIL':
            v2=gate_v2(c,budget)
            print('Gate V2',v2['status'],flush=True)
            if v2['status']=='PASS':
                suite('harmonic',c,output,budget,rows,diagnostics,common)
                v3=representation_decision(rows,c);v4=evidence_stage(v3,common,c,output,budget,rows)
        historical_proofs();frozen_proofs();execution_proofs();budget.check()
        result={'status':'COMPLETE_SYNTHETIC_SEQUENTIAL_STUDY','git_head':git_head(),'config':c,'config_sha256':digest(c),
            'historical':historical_proof,'preregistration':prereg,'execution':execution,
            'environment':environment('cpu'),'rows':rows,'diagnostics':diagnostics,'template_diagnostics':template_diagnostics(),
            'gates':{'V0':'PASS','V1':v1,'V2':v2,'V3':v3,'V4':v4},
            'wall_seconds':budget.elapsed,'peak_process_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'optimizer_updates':sum(r['training']['optimizer_updates'] for r in rows),
            'example_exposures':sum(r['training']['example_exposures'] for r in rows),
            'real_data_accessed':False,'reserved_access':False,'st_access':False,'shhs_access':False,
            'clinical_or_transfer_claim':False}
        atomic_json(result,output/'results.json');return result
    except BaseException as error:
        atomic_json({'status':'INCOMPLETE_STOPPED','exception_type':type(error).__name__,'reason':str(error),
            'completed_rows':len(rows),'wall_seconds':budget.elapsed,'real_data_accessed':False},output/'STOPPED.json')
        raise

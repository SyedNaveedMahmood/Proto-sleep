"""Frozen localized-information gates; failed prerequisites prevent head fits."""
import copy
import json
import resource
from pathlib import Path
import subprocess
import time
import torch
from mist_cove.phase2.provenance import environment,git_head,digest
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase5.models import RestrictedProbe
from mist_cove.phase5.runtime import Budget,fit,predict,chance_reference
from mist_cove.phase5.experiment import CachedHead
from mist_evidence.runtime import atomic_json,atomic_torch_save,tensor_digest
from .runtime import ROOT,frozen,historical,private_output
from .generator import generate,conditions_for
from .features import features,signal_view
from .evaluation import score,evaluate
from .audit import pairs,audit

CONFIG=ROOT/'configs/mist_cove_phase6_synthetic_v1.json'


def draws(c,seed,task,all_conditions=False):
    train=generate(c['train_examples'],seed+c['train_offset'],task,'train')
    names=conditions_for(task,c['conditions']) if all_conditions else ['clean','competitive']
    evals={name:generate(c['eval_examples'] if name in {'clean','competitive'} else c['stress_examples'],seed+c['eval_offset'],task,name) for name in names}
    return train,evals


def checkpoint_progress(output,rows):atomic_json({'status':'IN_PROGRESS','rows':rows},output/'partial.json')


def leakage_stage(c,output,budget,rows):
    checks=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            budget.check();train,evals=draws(c,seed,task)
            for name in c['leakage_probes']:
                started=time.perf_counter();f=features(train,name);torch.manual_seed(seed)
                model=RestrictedProbe(f,c['probe_hidden'],c['feature_std_floor'])
                training=fit(model,f,train.y,c,seed,budget)
                conditions={key:score(predict(model,features(data,name),budget),data.y,c) for key,data in evals.items()}
                row={'task':task,'seed':seed,'stage':'VI-1','method':name,'training':training,'conditions':conditions,'run_seconds':time.perf_counter()-started}
                rows.append(row);checkpoint_progress(output,rows)
                atomic_torch_save(model.state_dict(),output/f'{task}_{seed}_leakage_{name}.pt')
                passed=all(v['equivalent_chance'] for v in conditions.values())
                checks.append({'task':task,'seed':seed,'probe':name,'pass':passed,'reliable_leakage':any(v['reliable_leakage'] for v in conditions.values())})
                print('VI-1',task,seed,name,'PASS' if passed else 'FAIL_OR_INCONCLUSIVE',flush=True)
    return {'status':'PASS' if all(v['pass'] for v in checks) else 'FAIL_OR_INCONCLUSIVE','checks':checks}


def conventional_stage(c,output,budget,rows):
    common={};checks=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            budget.check();train,evals=draws(c,seed,task,True)
            for name,data in {'train':train,**evals}.items():atomic_json({'synthetic_latents':data.latents,'condition':name},output/f'{task}_{seed}_{name}_latents.json')
            for method in c['conventional_controls']:
                started=time.perf_counter();torch.manual_seed(seed);model=make_model('M1')
                training=fit(model,signal_view(train,method),train.y,c,seed,budget)
                names=list(evals) if method=='full' else ['clean','competitive']
                conditions={name:evaluate(model,evals[name],c,budget,view=method,localize=method=='full',seed=seed) for name in names}
                row={'task':task,'seed':seed,'stage':'VI-3','method':method,'training':training,'conditions':conditions}
                if method=='full':
                    data=generate(c['pair_examples'],seed+c['pair_offset'],task,'clean')
                    row['pairs']=pairs(model,data,c,seed,budget)
                    row['audits']={name:audit(model,generate(c['pair_examples'],seed+c['pair_offset'],task,name),c,seed+c['pair_offset']) for name in ['clean','competitive','removed']+(['redundant','complementary','overlap'] if task=='multiple' else [])}
                    common[(task,seed)]=copy.deepcopy(model.encoder.state_dict())
                row['run_seconds']=time.perf_counter()-started;rows.append(row);checkpoint_progress(output,rows)
                atomic_torch_save(model.state_dict(),output/f'{task}_{seed}_{method}.pt')
                print('VI-3',task,seed,method,'clean F1',round(conditions['clean']['macro_f1'],6),'competitive',round(conditions['competitive']['macro_f1'],6),flush=True)
            for method in c['proxy_probes']:
                started=time.perf_counter();f=features(train,method);torch.manual_seed(seed)
                model=RestrictedProbe(f,c['probe_hidden'],c['feature_std_floor']);training=fit(model,f,train.y,c,seed,budget)
                conditions={name:score(predict(model,features(evals[name],method),budget),evals[name].y,c) for name in ['clean','competitive']}
                rows.append({'task':task,'seed':seed,'stage':'VI-3-proxy','method':method,'training':training,'conditions':conditions,'run_seconds':time.perf_counter()-started})
                checkpoint_progress(output,rows);atomic_torch_save(model.state_dict(),output/f'{task}_{seed}_proxy_{method}.pt')
            relevant=[v for v in rows if v['task']==task and v['seed']==seed and v['stage']=='VI-3']
            m=next(v for v in relevant if v['method']=='full');k=c['criteria']
            criteria={'clean_f1':m['conditions']['clean']['macro_f1']>=k['clean_f1'],
                'competitive_f1':m['conditions']['competitive']['macro_f1']>=k['competitive_f1'],
                'appropriate_diagnostic_pair':m['pairs']['appropriate_diagnostic_change']['mean']>=k['diagnostic_pair_appropriate'],
                'nuisance_pair_agreement':m['pairs']['nuisance_agreement']['mean']>=k['nuisance_pair_agreement'],
                'nondiagnostic_learned_controls':all(v['conditions'][cond]['equivalent_chance'] for v in relevant if v['method'] in {'removed','background','nuisance'} for cond in ['clean','competitive'])}
            checks.append({'task':task,'seed':seed,'criteria':criteria,'pass':all(criteria.values())})
    return {'status':'PASS' if all(v['pass'] for v in checks) else 'FAIL','checks':checks},common


def head_stage(gate,common,c,output,budget,rows):
    if gate['status']!='PASS':return {'status':'NOT_TESTED_VI3_FAILED','optimizer_updates':0}
    checks=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            train,evals=draws(c,seed,task,True);local=[]
            for method in c['heads']:
                budget.check();torch.manual_seed(seed+402);model=make_model(method)
                model.encoder.load_state_dict(common[(task,seed)]);model.encoder.requires_grad_(False)
                encoder_sha=tensor_digest(model.encoder.state_dict())
                with torch.no_grad():f,r=model.encode_candidates(train.x)
                training=fit(CachedHead(model),torch.cat((f,r[:,:,None]),-1),train.y,c,seed,budget,updates=c['head_updates'])
                assert tensor_digest(model.encoder.state_dict())==encoder_sha
                conditions={name:evaluate(model,data,c,budget,localize=True,seed=seed) for name,data in evals.items()}
                audits={name:audit(model,generate(c['pair_examples'],seed+c['pair_offset'],task,name),c,seed+c['pair_offset']) for name in ['clean','competitive','removed']+(['redundant','complementary','overlap'] if task=='multiple' else [])}
                row={'task':task,'seed':seed,'stage':'VI-4','method':method,'training':training,'encoder_sha256':encoder_sha,'conditions':conditions,'audits':audits}
                rows.append(row);local.append(row);checkpoint_progress(output,rows);atomic_torch_save(model.state_dict(),output/f'{task}_{seed}_head_{method}.pt')
                print('VI-4',task,seed,method,flush=True)
            ref=next(v for v in local if v['method']=='M1');m=next(v for v in local if v['method']=='M2');rms=next(v for v in local if v['method']=='M2_rms');k=c['criteria']
            clauses={'f1_penalty':m['conditions']['clean']['macro_f1']>=ref['conditions']['clean']['macro_f1']-k['f1_penalty'],
                'topk_hit':task!='single' or m['conditions']['clean']['localization']['model']['topk_hit']>=k['topk_hit'],
                'competitive_rms_gain':m['conditions']['competitive']['localization']['model']['topk_hit']>=rms['conditions']['competitive']['localization']['model']['topk_hit']+k['rms_localization_gain'],
                'oracle_control_contrast':m['audits']['clean']['oracle_minus_control_abs_true_stage_mean']>=k['oracle_control_effect_margin'],
                'reconciliation':max(v.get('max_additive_residual',0) for v in m['conditions'].values())<=k['reconciliation'] and max(v['max_reconciliation_residual'] for v in m['audits'].values())<=k['reconciliation']}
            checks.append({'task':task,'seed':seed,'criteria':clauses,'pass':all(clauses.values())})
    return {'status':'PASS' if all(v['pass'] for v in checks) else 'FAIL','checks':checks}


def run(path):
    c=json.loads(CONFIG.read_text());old=historical();proof=frozen()
    if c['scope']!='SYNTHETIC_ONLY' or c['real_data_authorized']:raise ValueError('synthetic-only configuration required')
    subprocess.run(['git','merge-base','--is-ancestor',c['base_commit'],'HEAD'],check=True)
    output=private_output(path,c['scope']);configure_precision('cpu');torch.set_num_threads(c['cpu_threads'])
    budget=Budget(c['cpu_wall_ceiling_seconds']);rows=[]
    try:
        v1=leakage_stage(c,output,budget,rows);v3={'status':'NOT_TESTED_VI1_FAILED'};v4={'status':'NOT_TESTED_VI1_FAILED','optimizer_updates':0}
        if v1['status']=='PASS':
            v3,common=conventional_stage(c,output,budget,rows)
            v4=head_stage(v3,common,c,output,budget,rows)
        assert historical()==old and frozen()==proof;budget.check()
        report={'status':'COMPLETE_SYNTHETIC_STUDY','git_head':git_head(),'config':c,'config_sha256':digest(c),
            'historical':old,'frozen':proof,'environment':environment('cpu'),'rows':rows,
            'gates':{'VI-0':'PASS','VI-1':v1,'VI-2':'COMMITTED_PRE_FIT_FREEZE','VI-3':v3,'VI-4':v4},
            'wall_seconds':budget.elapsed,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'fits':len(rows),'optimizer_updates':sum(v['training']['optimizer_updates'] for v in rows),
            'example_exposures':sum(v['training']['example_exposures'] for v in rows),
            'real_data_accessed':False,'reserved_access':False,'st_access':False,'shhs_access':False,'real_training_authorized':False}
        atomic_json(report,output/'results.json');return report
    except BaseException as error:
        atomic_json({'status':'STOPPED_INCOMPLETE','error':type(error).__name__,'reason':str(error),'completed_fits':len(rows),'wall_seconds':budget.elapsed,'real_data_accessed':False},output/'STOPPED.json');raise

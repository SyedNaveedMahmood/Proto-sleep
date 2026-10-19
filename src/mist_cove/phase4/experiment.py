"""Frozen synthetic comparison; no validation tuning, source IO or saved traces."""
import copy
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from mist_cove.experiment import metrics
from mist_cove.phase2.provenance import digest,file_sha,implementation_snapshot,environment,git_head
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.audit import jaccard_distance,selected_set
from mist_evidence.runtime import tensor_digest
from .model import make_model
from .synthetic import generate,diagnostic_intervals,localization
from .audit import group_report


def verify_preregistration():
    p=Path('experiments/mist_cove/phase4_preregistration_v2.json')
    r=json.loads(p.read_text())
    for name,h in r['sha256'].items():
        if file_sha(name)!=h:raise ValueError('preregistration changed: '+name)
    return {'sha256':file_sha(p),'files':r['sha256']}


def train_model(model,batch,c,seed,*,features=None,updates=None):
    model.train();g=torch.Generator().manual_seed(seed+711)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c['weight_decay'])
    losses=[];start=time.perf_counter()
    for step in range(updates or c['updates']):
        ix=torch.randperm(len(batch.y),generator=g)[:c['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch.x[ix]) if features is None else model.decision(features[0][ix],features[1][ix])[0]
        loss=torch.nn.functional.cross_entropy(logits,batch.y[ix])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite synthetic loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite synthetic gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite updated model')
    return {'optimizer_updates':len(losses),'training_example_exposures':len(losses)*c['batch_size'],'initial_loss':losses[0],'final_loss':losses[-1],
            'wall_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameters_with_nonzero_final_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum()>0)}


@torch.no_grad()
def selected_intervals(model,x):
    if hasattr(model,'cfg') and getattr(model.cfg,'method',None) in {'M1','M1_phase','M3'}:
        f,r=model.encode_candidates(x);scores=model.conventional_ranking(f,r)
        ix=scores.masked_fill(r<=1e-6,-float('inf')).argsort(dim=-1,descending=True,stable=True)[0,:4]
        ix=ix[r[0,ix]>1e-6]
        return [(int(j)*200,int(j)*200+400) for j in ix]
    ev=model(x,return_evidence=True)
    return [tuple(iv) for iv in ev.event_intervals[0,ev.valid[0]].tolist()]


@torch.no_grad()
def evaluate(model,batch):
    model.eval();logits=[];loc=[];residual=0.
    for start in range(0,len(batch.y),16):
        ev=model(batch.x[start:start+16],return_evidence=True);logits.append(ev.logits)
        residual=max(residual,float((ev.logits-ev.background_logits-ev.contributions.sum(1)).abs().max()))
    for j in range(len(batch.y)):
        loc.append(localization(selected_intervals(model,batch.x[j:j+1]),diagnostic_intervals(batch.components[j])))
    out=metrics(torch.cat(logits),batch.y)
    out['per_stage_accuracy']=out['per_stage_recall']
    out['localization']={k:float(np.mean([r[k] for r in loc if r[k] is not None])) if any(r[k] is not None for r in loc) else None for k in loc[0]}
    out['localization_estimand']='internal_feature_deletion_rank' if getattr(model.cfg,'method',None) in {'M1','M1_phase','M3'} else 'native_selected_event_rank'
    out['max_additive_residual']=residual
    if residual>1e-5:raise ValueError('additive reconciliation failed')
    return out


@torch.no_grad()
def audit_examples(model,batch,seed):
    model.eval();groups=[];selected=[];controls=[];stability=[];native_scores=[]
    examples=[int((batch.y==s).nonzero()[0]) for s in range(5)]
    for j in examples:
        x=batch.x[j:j+1];parts=batch.components[j];truth=diagnostic_intervals(parts);stage=int(batch.y[j])
        iv=selected_intervals(model,x)
        targets=truth or [(0,200)]
        joint=group_report(model,x,targets,parts,seed=seed+j)
        groups.append((stage,joint))
        if iv:
            selected.append((stage,group_report(model,x,[iv[0]],parts,seed=seed+j)))
            ev=model(x,return_evidence=True)
            native_scores.append(ev.contributions[0,ev.valid[0]][0].tolist() if ev.valid[0].any() else None)
            # Fixed energy-nearest non-diagnostic, same-length control rule;
            # label information is synthetic ground truth used for audit only.
            length=iv[0][1]-iv[0][0];energy=float(x[0,0,iv[0][0]:iv[0][1]].square().mean())
            candidates=[(abs(float(x[0,0,a:a+length].square().mean())-energy),a) for a in range(0,3001-length,100) if not any(a<b and a+length>lo for lo,b in truth) and (a,a+length)!=iv[0]]
            if candidates:
                _,a=min(candidates);controls.append((stage,group_report(model,x,[(a,a+length)],parts,seed=seed+j)))
        g=torch.Generator().manual_seed(seed+j+619)
        changed=x+torch.randn(x.shape,generator=g)*.01
        stability.append(jaccard_distance(set(iv),set(selected_intervals(model,changed))))
    result={'n_examples':len(examples),'tiny_noise_jaccard':float(np.mean(stability)),
            'donor_spread_max_mean':float(np.mean([v['donor_spread_max'] for _,v in groups])),
            'max_reconciliation_residual':max(abs(r) for _,g in groups for d in g['donors'] for r in d['residual']),
            'donors':{},'selected_control':{},'group_geometry_caveat':'Union bridge differs from composing primitive bridges; oracle component subtraction is separately reported.'}
    for donor in ['oracle','bridge','background']:
        values=[(s,next(v for v in g['donors'] if v['donor']==donor)) for s,g in groups]
        d={key:np.mean([v[key] for _,v in values],axis=0).tolist() for key in ['additive','full','background','fixed_witness','fixed_event_path','reselection']}
        d.update(joint_abs_true_stage=float(np.mean([abs(v['full'][s]) for s,v in values])),
          individual_mean_full_effect=np.mean([np.mean(v['individuals'],axis=0) for _,v in values],axis=0).tolist(),
          n_individual_interventions=sum(len(v['individuals']) for _,v in values),
          largest_individual_abs_true_stage=float(np.mean([max(abs(one[s]) for one in v['individuals']) for s,v in values])),
          confidence_after_mean=float(np.mean([v['confidence_after'] for _,v in values])),
          additive_full_mae=float(np.mean([np.abs(np.asarray(v['additive'])-v['full']).mean() for _,v in values])),
          group_individual_nonadditivity_l1=float(np.mean([np.abs(np.asarray(v['full'])-np.sum(v['individuals'],axis=0)).sum() for _,v in values])),
          energy_mismatch=float(np.mean([v['energy_mismatch'] for _,v in values])),spectral_distortion=float(np.mean([v['spectral_distortion'] for _,v in values])),
          boundary_jump_change_max=max(v['boundary_jump_change_max'] for _,v in values),
          elementary_unsupported_count=sum(not v['elementary_supported'] for _,v in values),physiological_validity='NOT_ESTABLISHED')
        d['selection_membership_changed_fraction']=float(np.mean([v['selection_membership_changed'] for _,v in values]))
        result['donors'][donor]=d
        selected_effect=[next(v for v in g['donors'] if v['donor']==donor)['full'][s] for s,g in selected]
        control_effect=[next(v for v in g['donors'] if v['donor']==donor)['full'][s] for s,g in controls]
        result['selected_control'][donor]={'selected_mean':float(np.mean(selected_effect)) if selected_effect else None,'control_mean':float(np.mean(control_effect)) if control_effect else None,'n_selected':len(selected_effect),'n_control':len(control_effect)}
        pairs=[(np.asarray(score),np.asarray(next(v for v in g['donors'] if v['donor']==donor)['full'])) for score,(_,g) in zip(native_scores,selected) if score is not None]
        d['native_first_event_vs_full_interval_mae']=float(np.mean([np.abs(score-effect).mean() for score,effect in pairs])) if pairs else None
        d['native_first_event_vs_full_interval_sign_agreement']=float(np.mean([np.mean(np.sign(score)==np.sign(effect)) for score,effect in pairs])) if pairs else None
        d['native_score_caveat']='One original event score; replacement can also change overlapping witnesses, global background and selection. Not necessity.'
    return result


def decisions(rows,c):
    def row(task,seed,method,regime='end_to_end'):
        return next(v for v in rows if (v['task'],v['seed'],v['method'],v['regime'])==(task,seed,method,regime))
    checks=[];k=c['criteria']
    for task in c['tasks']:
        for seed in c['seeds']:
            m1=row(task,seed,'M1');m2=row(task,seed,'M2');rms=row(task,seed,'M2_rms')
            rep=all(m1['conditions'][cond]['macro_f1']>=row(task,seed,other)['conditions'][cond]['macro_f1']+k['representation_margin'] for cond in ['clean','competitive'] for other in ['M0','M1_phase']) if task=='substitutable' else None
            one=row(task,seed,'M1','fixed_encoder');two=row(task,seed,'M2','fixed_encoder')
            cf=two['audits']['clean'];contrast=cf['selected_control']['oracle']
            organization=(two['conditions']['clean']['macro_f1']>=one['conditions']['clean']['macro_f1']-k['organization_f1_penalty'] and two['conditions']['competitive']['localization']['top1_hit']>=one['conditions']['competitive']['localization']['top1_hit']+k['localization_advantage'] and two['conditions']['competitive']['localization']['top1_hit']>=k['top1_hit_minimum'] and contrast['selected_mean'] is not None and contrast['control_mean'] is not None and contrast['selected_mean']>=contrast['control_mean']+k['selected_control_effect_margin'] and cf['donor_spread_max_mean']<=k['donor_spread_maximum'])
            plural=all(m2['audits'][cond]['donors']['oracle']['joint_abs_true_stage']>=m2['audits'][cond]['donors']['oracle']['largest_individual_abs_true_stage']+k['joint_individual_margin'] and m2['audits'][cond]['tiny_noise_jaccard']<=k['tiny_noise_jaccard_maximum'] for cond in ['redundant','correlated']) and m2['conditions']['erased']['confidence']<=k['erased_confidence_maximum']
            donor_robust=all(v['donor_spread_max_mean']<=k['donor_spread_maximum'] for v in m2['audits'].values())
            distractor=all(m2['conditions']['competitive'][key]>=rms['conditions']['competitive'][key]+k['distractor_advantage'] for key in ['macro_f1']) and m2['conditions']['competitive']['localization']['top1_hit']>=rms['conditions']['competitive']['localization']['top1_hit']+k['distractor_advantage']
            checks.append({'task':task,'seed':seed,'H4_1':rep,'H4_2':bool(organization),'H4_3':bool(plural),'donor_robustness':bool(donor_robust),'learned_vs_rms_distractor':bool(distractor)})
    return checks


def run(c,output):
    proof=verify_preregistration()
    if c!=json.loads(Path('configs/mist_cove_phase4_synthetic_v2.json').read_text()):raise ValueError('confirmatory config override prohibited; historical v1 requires its original checkout')
    output=Path(output)
    if output.exists():raise FileExistsError('preserve existing results')
    output.mkdir(parents=True);configure_precision('cpu');torch.set_num_threads(2)
    snapshot=implementation_snapshot();started=time.perf_counter();rows=[]
    for ti,task in enumerate(c['tasks']):
        for seed in c['seeds']:
            batch=generate(c['train_examples'],seed+1000+ti*1000000,task,'train',c)
            evals={cond:generate(c['eval_examples'],seed+100000+ti*1000000+ci*10000,task,cond,c) for ci,cond in enumerate(c['conditions'])}
            common=None
            for method in c['methods']:
                if time.perf_counter()-started>c['cpu_wall_ceiling_seconds']:raise TimeoutError('synthetic CPU budget exceeded')
                model_started=time.perf_counter();torch.manual_seed(seed);model=make_model(method)
                training=train_model(model,batch,c,seed)
                if method=='M1':common=copy.deepcopy(model.encoder.state_dict())
                result={'task':task,'seed':seed,'method':method,'regime':'end_to_end','training':training,
                        'encoder_digest':tensor_digest(model.encoder.state_dict()) if hasattr(model,'encoder') else None,
                        'conditions':{cond:evaluate(model,data) for cond,data in evals.items()},
                        'audits':{cond:audit_examples(model,evals[cond],seed) for cond in c['audit_conditions']}}
                result['run_wall_seconds']=time.perf_counter()-model_started
                rows.append(result);torch.save(model.state_dict(),output/f'{task}_{seed}_{method}_synthetic.pt')
                (output/'partial.json').write_text(json.dumps({'rows':rows},allow_nan=False))
                print(task,seed,method,'clean F1',result['conditions']['clean']['macro_f1'],'competition F1',result['conditions']['competitive']['macro_f1'],flush=True)
            assert common is not None
            for method in c['frozen_encoder_heads']:
                model_started=time.perf_counter();torch.manual_seed(seed+402);model=make_model(method);model.encoder.load_state_dict(common);model.encoder.requires_grad_(False)
                with torch.no_grad():features=model.encode_candidates(batch.x)
                training=train_model(model,batch,c,seed,features=features,updates=c['frozen_head_updates'])
                assert tensor_digest(model.encoder.state_dict())==tensor_digest(common)
                rows.append({'task':task,'seed':seed,'method':method,'regime':'fixed_encoder','training':training,
                    'encoder_digest':tensor_digest(common),'common_encoder_provenance':'same task/seed final TRAIN-only M1; pretraining head discarded',
                    'conditions':{cond:evaluate(model,data) for cond,data in evals.items()},
                    'audits':{cond:audit_examples(model,evals[cond],seed) for cond in c['audit_conditions']}})
                rows[-1]['run_wall_seconds']=time.perf_counter()-model_started
                print(task,seed,method,'fixed encoder complete',flush=True)
    if implementation_snapshot()!=snapshot:raise ValueError('implementation changed during confirmatory execution')
    verify_preregistration()
    report={'status':'CONFIRMATORY_SYNTHETIC_COMPLETE','config':c,'config_sha256':digest(c),'preregistration':proof,'implementation':snapshot,
            'git_head':git_head(),'environment':environment('cpu'),'rows':rows,'decisions':decisions(rows,c),
            'wall_seconds':time.perf_counter()-started,'peak_process_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'real_data_accessed':False,'reserved_access':False,'st_access':False,'shhs_access':False,
            'interpretation':'Artificial phase codes and controlled nuisance, not clinical EEG or transfer evidence.'}
    with (output/'results.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report

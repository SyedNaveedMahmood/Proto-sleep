"""New sealed synthetic states. Never reads the unsealed historical weights."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import resource
import time

import torch
from torch import nn

from mist_cove.phase2.provenance import digest, environment, file_sha
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase5.models import RestrictedProbe
from mist_cove.phase5.runtime import Budget, fit, predict
from mist_cove.phase6.generator import generate, compose
from mist_cove.phase6.evaluation import location
from mist_cove.experiment import metrics
from mist_evidence.runtime import atomic_json, atomic_torch_save, tensor_digest, rng_state, restore_rng, load_trusted
from .integrity import ROOT, BASE, snapshot, require_scope, git

CONFIG = ROOT/'configs/mist_cove_phase7_prospective_v1.json'
MANIFEST = ROOT/'experiments/mist_cove/phase7_prospective_freeze_v2.json'


def proof():
    snapshot(ROOT)
    r=json.loads(MANIFEST.read_text())
    for name, expected in r['files'].items():
        if file_sha(ROOT/name)!=expected:raise ValueError('prospective frozen file mismatch: '+name)
    return {'manifest_sha256':file_sha(MANIFEST),'inventory_sha256':digest(r['files'])}


def output_directory(path,scope):
    require_scope(scope)
    path=Path(path);allowed=(ROOT/'mist_transfer_runs/mist_cove/phase7_prospective_v1').resolve()
    if allowed not in path.resolve().parents:raise ValueError('new private prospective output required')
    if path.exists():raise FileExistsError('preserve completed/incomplete outputs')
    path.mkdir(parents=True);return path


def matched(n,seed,task,c,*,strength=1.,condition='clean',labels=None):
    clean=generate(n,seed,task,condition,labels=labels)
    extra=generate(n,seed+c['extra_nuisance_offset'],task,'competitive')
    if condition=='eventless':
        strength=0.;extra.nuisance_intervals=[[] for _ in extra.y]
    added=extra.nuisance*strength
    base=clean.base+added
    competitive=replace(clean,x=compose(base,clean.events),base=base,nuisance=clean.nuisance+added,
        nuisance_intervals=[a+b for a,b in zip(clean.nuisance_intervals,extra.nuisance_intervals)],
        condition='paired_added_nuisance',latents=[{'original':a,'added':b,'strength':strength} for a,b in zip(clean.latents,extra.latents)])
    return clean,competitive


def oracle_mask(draw,*,contain=False):
    return torch.tensor([[any((j*200<=a and j*200+400>=b) if contain else
        (j*200<b and j*200+400>a) for a,b in truth) for j in range(14)] for truth in draw.intervals])


def pooled(f,mask=None):
    if mask is None:return torch.cat((f.mean(1),f.amax(1)),1)
    if not mask.any(1).all():raise ValueError('oracle pooling requires nonempty diagnostic truth')
    mean=(f*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
    return torch.cat((mean,f.masked_fill(~mask[:,:,None],-float('inf')).amax(1)),1)


def head(model,p):
    weights=torch.cat((model.background_head.weight,model.event_head.weight),1)
    return p@weights.T+model.background_head.bias


def encode(model,x):
    with torch.no_grad():
        pieces=[model.encode_candidates(v) for v in x.split(64)]
    return torch.cat([v[0] for v in pieces]),torch.cat([v[1] for v in pieces])


def optimizer(model,c):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c['weight_decay'])


def steps(model,opt,g,x,y,c,start,end,budget):
    model.train()
    for step in range(start,end):
        budget.check();ix=torch.randperm(len(y),generator=g)[:c['batch_size']]
        opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x[ix]),y[ix])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite model')
    return float(loss.detach())


def seal(path,model,opt,g,step,identity):
    path=Path(path)
    if path.exists() or path.with_suffix('.seal.json').exists():raise FileExistsError('preserve checkpoint')
    state={'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
        'batch_rng':g.get_state(),'step':step,'model_digest':tensor_digest(model.state_dict())}
    atomic_torch_save(state,path)
    value={'identity':identity,'sha256':file_sha(path),'step':step,'model_digest':state['model_digest']}
    atomic_json(value,path.with_suffix('.seal.json'))
    creation={'checkpoint_sha256':value['sha256'],'seal_sha256':file_sha(path.with_suffix('.seal.json')),
              'identity':identity,'step':step,'model_digest':state['model_digest']}
    atomic_json(creation,path.with_suffix('.creation.json'))
    return creation


def restore(path,creation,identity,model,opt,g):
    path=Path(path)
    if creation['identity']!=identity or file_sha(path)!=creation['checkpoint_sha256'] or file_sha(path.with_suffix('.seal.json'))!=creation['seal_sha256']:
        raise ValueError('creation proof mismatch before checkpoint deserialization')
    bound=json.loads(path.with_suffix('.seal.json').read_text())
    if bound['identity']!=identity or bound['sha256']!=creation['checkpoint_sha256']:raise ValueError('seal identity mismatch')
    state=load_trusted(path)
    if state['identity']!=identity or state['model_digest']!=creation['model_digest'] or tensor_digest(state['model'])!=creation['model_digest'] or state['step']!=creation['step']:
        raise ValueError('checkpoint payload mismatch')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);g.set_state(state['batch_rng']);restore_rng(state['rng'])
    return state['step']


def train_checkpoint(x,y,c,seed,path,identity,budget):
    started=time.perf_counter();torch.manual_seed(seed);model=make_model('M1')
    opt=optimizer(model,c);g=torch.Generator().manual_seed(seed+711)
    loss=steps(model,opt,g,x,y,c,0,c['updates'],budget)
    before=rng_state();creation=seal(path,model,opt,g,c['updates'],identity)
    # Load only the state just created, against its in-memory creation proof.
    restored=make_model('M1');restored_opt=optimizer(restored,c);restored_g=torch.Generator()
    assert restore(path,creation,identity,restored,restored_opt,restored_g)==c['updates']
    assert tensor_digest(model.state_dict())==tensor_digest(restored.state_dict())
    restore_rng(before)
    return restored,{'updates':c['updates'],'exposures':c['updates']*c['batch_size'],'seconds':time.perf_counter()-started,
        'final_loss':loss,'parameters':sum(p.numel() for p in model.parameters()),'creation':creation}


class LinearProbe(nn.Module):
    def __init__(self,f):
        super().__init__();self.register_buffer('mean',f.mean(0));self.register_buffer('std',f.std(0).clamp_min(.01));self.linear=nn.Linear(f.shape[1],5)
    def forward(self,f):return self.linear((f-self.mean)/self.std)


def score(logits,y):
    r=metrics(logits,y);p=logits.softmax(-1);pred=p.argmax(-1)
    r.update(entropy=float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),
             false_confidence_0_9=float(((pred!=y)&(p.max(-1).values>=.9)).float().mean()),
             mean_true_margin=float((logits.gather(1,y[:,None])[:,0]-logits.masked_fill(nn.functional.one_hot(y,5).bool(),-float('inf')).max(-1).values).mean()))
    return r


def location_batch(selected,draw):
    records=[location(v,t,n) for v,t,n in zip(selected,draw.intervals,draw.nuisance_intervals)]
    return {k:sum(v[k] for v in records if v[k] is not None)/sum(v[k] is not None for v in records)
            if any(v[k] is not None for v in records) else None for k in records[0]}


def feature_paths(model,a,b,mask):
    pa,pb=pooled(a),pooled(b);la,lb=head(model,pa),head(model,pb)
    diag=torch.where(mask[:,:,None],b,a);other=torch.where(mask[:,:,None],a,b)
    ld,ln=head(model,pooled(diag)),head(model,pooled(other))
    weights=torch.cat((model.background_head.weight,model.event_head.weight),1)
    parts=(pa-pb)[:,:,None]*weights.T
    error=max(float((parts.sum(1)-(la-lb)).abs().max()),float(((la-ld)+(ld-lb)-(la-lb)).abs().max()))
    if error>1e-5:raise ValueError('feature path reconciliation failed')
    return {'diagnostic_first_abs_mean':float((la-ld).abs().mean()),'nondiagnostic_second_abs_mean':float((ld-lb).abs().mean()),
        'nondiagnostic_first_abs_mean':float((la-ln).abs().mean()),'diagnostic_second_abs_mean':float((ln-lb).abs().mean()),
        'interaction_abs_mean':float(((la-ld)-(ln-lb)).abs().mean()),'max_reconciliation_error':error,
        'signed_head_feature_logit_delta_mean':parts.mean(0).tolist(),'off_manifold_internal_intervention':True}


def analyze(model,clean,comp,train_features,c,budget,private):
    a,ar=encode(model,clean.x);b,br=encode(model,comp.x)
    la,lb=head(model,pooled(a)),head(model,pooled(b));y=clean.y
    mask=oracle_mask(clean);contain=oracle_mask(clean,contain=True)
    with torch.no_grad():
        ra=model.conventional_ranking(a,ar);rb=model.conventional_ranking(b,br)
    ia=ra.argsort(dim=1,descending=True,stable=True)[:,:4];ib=rb.argsort(dim=1,descending=True,stable=True)[:,:4]
    iva=[[(j*200,j*200+400) for j in row] for row in ia.tolist()];ivb=[[(j*200,j*200+400) for j in row] for row in ib.tolist()]
    def hit(iv,truth):return any(max(0,min(b,d)-max(a,cc))/(max(b,d)-min(a,cc))>=.25 for a,b in iv for cc,d in truth)
    ha=torch.tensor([hit(v,t) for v,t in zip(iva,clean.intervals)]);hb=torch.tensor([hit(v,t) for v,t in zip(ivb,comp.intervals)])
    correct_a,correct_b=la.argmax(1)==y,lb.argmax(1)==y
    local_a=pooled(a,contain)[:,:18];local_b=pooled(b,contain)[:,:18]
    mu=train_features.mean((0,1));sd=train_features.std((0,1)).clamp_min(.01)
    clip=lambda f:torch.maximum(torch.minimum(f,mu+c['clip_z']*sd),mu-c['clip_z']*sd)
    def summaries(f):return {'feature_norm_mean':float(f.norm(dim=-1).mean()),'oracle_candidate_norm_mean':float(f.norm(dim=-1)[mask].mean()),
        'nuisance_candidate_norm_mean':float(f.norm(dim=-1)[~mask].mean()),'nondiagnostic_max_winner_fraction':float((~mask).gather(1,f.argmax(1)).float().mean())}
    # Detailed generated outputs remain private, never public waveform arrays.
    atomic_torch_save({'clean_logits':la,'competitive_logits':lb,'probabilities_clean':la.softmax(-1),'probabilities_competitive':lb.softmax(-1),
        'labels':y,'clean_features':a,'competitive_features':b,'oracle_mask':mask,'clean_intervals':clean.intervals,
        'competitive_latents':comp.latents},private)
    result={'clean':score(la,y),'competitive':score(lb,y),'transitions':{
        'correct_to_wrong':float((correct_a&~correct_b).float().mean()),'wrong_to_correct':float((~correct_a&correct_b).float().mean()),
        'confidence_increased':float((lb.softmax(-1).max(-1).values>la.softmax(-1).max(-1).values).float().mean()),
        'localized_but_wrong':float((hb&~correct_b).float().mean()),'lost_localization_and_wrong':float((ha&~hb&~correct_b).float().mean())},
        'localization_clean':location_batch(iva,clean),
        'localization_competitive':location_batch(ivb,comp),
        'features_clean':summaries(a),'features_competitive':summaries(b),
        'diagnostic_candidate_drift':float((a-b).norm(dim=-1)[mask].mean()),'nondiagnostic_candidate_drift':float((a-b).norm(dim=-1)[~mask].mean()),
        'pooled_drift':float((pooled(a)-pooled(b)).norm(dim=-1).mean()),'paths':feature_paths(model,a,b,mask),
        'oracle_pooling_competitive':score(head(model,pooled(b,contain)),y),'blind_clipped_pooling_competitive':score(head(model,pooled(clip(b))),y)}
    # Predeclared first one per class; exact component deletion, not donors.
    audits=[]
    for label in range(5):
        i=int((y==label).nonzero()[0]);events=comp.events[i:i+1];baseline=comp.base[i:i+1]
        original=lb[i];joint=original-predict(model,baseline,budget)[0];individual=[]
        for j in range(2 if comp.task=='multiple' else 1):
            keep=torch.ones(1,2);keep[:,j]=0
            individual.append((original-predict(model,compose(baseline,events,keep),budget)[0]).tolist())
        audits.append({'label':label,'joint_effect':joint.tolist(),'individual_effects':individual,'kind':'exact_generator_component_deletion'})
    result['oracle_group_effects']=audits
    return result,{'local':(local_a,local_b),'pooled':(pooled(a),pooled(b))}


def run_diagnostics(path):
    c=json.loads(CONFIG.read_text());frozen=proof();output=output_directory(path,c['scope'])
    configure_precision('cpu');torch.set_num_threads(c['cpu_threads']);budget=Budget(c['cpu_wall_ceiling_seconds']);rows=[];fits=0
    try:
        for task in c['tasks']:
            for seed in c['seeds']:
                train=generate(c['train_examples'],seed+c['train_offset'],task,'train')
                identity={'experiment':c['version'],'task':task,'seed':seed,'config_sha256':file_sha(CONFIG),'implementation':frozen,
                    'environment':environment('cpu'),'train_digest':tensor_digest({'x':train.x,'y':train.y})}
                reuse=json.loads(MANIFEST.read_text())['reuse_baseline']
                if (task,seed)==('single',123):
                    creation_path=ROOT/reuse['creation_path']
                    if file_sha(creation_path)!=reuse['creation_sha256']:raise ValueError('reused prospective creation proof changed')
                    creation=json.loads(creation_path.read_text());prior=creation['identity']
                    for key in ('experiment','task','seed','config_sha256','environment','train_digest'):
                        if prior[key]!=identity[key]:raise ValueError('reused prospective identity mismatch')
                    if prior['implementation']!=reuse['original_implementation']:raise ValueError('reused implementation mismatch')
                    model=make_model('M1');opt=optimizer(model,c);g=torch.Generator()
                    checkpoint=creation_path.with_name(creation_path.name.removesuffix('.creation.json')+'.pt')
                    assert restore(checkpoint,creation,prior,model,opt,g)==c['updates']
                    training={'updates':c['updates'],'exposures':c['updates']*c['batch_size'],'seconds':None,'parameters':3593,
                              'creation':creation,'reused_completed_fit_from_preserved_failed_audit':True}
                else:
                    model,training=train_checkpoint(train.x,train.y,c,seed,output/f'{task}_{seed}_P0.pt',identity,budget)
                fits+=1
                tf,_=encode(model,train.x)
                clean,comp=matched(c['diagnostic_examples'],seed+c['diagnostic_development_offset'],task,c)
                analysis,eval_features=analyze(model,clean,comp,tf,c,budget,output/f'{task}_{seed}_details.pt')
                base,competition=matched(c['probe_train_examples'],seed+c['probe_train_offset'],task,c)
                fa,_=encode(model,base.x);fb,_=encode(model,competition.x);use=torch.arange(len(base.y))%2==1
                mixed=torch.where(use[:,None,None],fb,fa);pmask=oracle_mask(base,contain=True)
                probe_features={'local':pooled(mixed,pmask)[:,:18],'pooled':pooled(mixed)};probes={}
                for kind,f in probe_features.items():
                    for nonlinear in (False,True):
                        torch.manual_seed(seed+211);probe=RestrictedProbe(f,c['probe_hidden'],.01) if nonlinear else LinearProbe(f)
                        training_probe=fit(probe,f,base.y,c,seed,budget);fits+=1
                        probes[kind+('_nonlinear' if nonlinear else '_linear')]={name:score(predict(probe,data,budget),clean.y) for name,data in zip(('clean','competitive'),eval_features[kind])}
                        probes[kind+('_nonlinear' if nonlinear else '_linear')]['training']=training_probe
                sweep={}
                for strength in c['strengths']:
                    _,draw=matched(c['diagnostic_examples'],seed+c['diagnostic_development_offset'],task,c,strength=strength)
                    sweep[str(strength)]=score(predict(model,draw.x,budget),draw.y)
                conditions={}
                for condition in ('removed','nuisance_swap','position_shift','eventless'):
                    a,b=matched(c['diagnostic_examples'],seed+c['diagnostic_development_offset'],task,c,condition=condition)
                    conditions[condition]=score(predict(model,b.x,budget),b.y)
                conditions['diagnostic_only']=score(predict(model,clean.diagnostic,budget),clean.y)
                row={'task':task,'seed':seed,'training':training,'analysis':analysis,'probes':probes,'strength_sweep':sweep,'conditions':conditions}
                rows.append(row);atomic_json({'status':'IN_PROGRESS','fits':fits,'rows':rows},output/'partial.json')
                print(task,seed,'clean',analysis['clean']['macro_f1'],'competitive',analysis['competitive']['macro_f1'],flush=True)
        assert fits==c['permitted_fits'] and proof()==frozen;budget.check()
        result={'status':'COMPLETE_PROSPECTIVE_DIAGNOSTICS','experiment':c['version'],'git_head':git(ROOT,'rev-parse','HEAD').decode().strip(),'frozen':frozen,'config':c,'rows':rows,
            'fits':fits,'optimizer_updates':fits*c['updates'],'example_exposures':fits*c['updates']*c['batch_size'],
            'wall_seconds':budget.elapsed,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'historical_checkpoint_verification':'BLOCKED','historical_weights_loaded':0,'real_data_accessed':False}
        atomic_json(result,output/'results.json');return result
    except BaseException as error:
        atomic_json({'status':'STOPPED','completed_fits':fits,'error':repr(error),'wall_seconds':budget.elapsed},output/'STOPPED.json');raise

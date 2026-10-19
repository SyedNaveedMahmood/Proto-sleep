"""Preregistered prospective R1/R2 controls and collectively gated heads."""
import copy
from dataclasses import replace
import json
import resource
import time
import torch
from mist_cove.phase4.model import MorphModel, MorphConfig, make_model
from mist_cove.phase4.audit import decompose
from mist_cove.phase5.experiment import CachedHead
from mist_cove.phase5.runtime import Budget, fit, predict, wilson
from mist_cove.phase6.generator import generate, compose, conditions_for, burst
from mist_cove.phase6.evaluation import evaluate, selected
from mist_cove.phase2.provenance import digest, file_sha, environment
from mist_cove.phase2.training import configure_precision
from mist_evidence.runtime import atomic_json, tensor_digest
from .integrity import ROOT, snapshot, git
from .prospective import (matched, encode, pooled, oracle_mask, head, score, steps, optimizer,
    seal, restore, output_directory)

CONFIG=ROOT/'configs/mist_cove_phase7_robustness_v1.json'
MANIFEST=ROOT/'experiments/mist_cove/phase7_robustness_freeze_v1.json'
DIAGNOSTICS=ROOT/'reports/mist_cove/phase7_v1/prospective_diagnostics_v2.json'


class ClippedModel(MorphModel):
    def __init__(self,mu,sd):
        super().__init__(MorphConfig(method='M1'))
        self.register_buffer('clip_mean',mu.clone());self.register_buffer('clip_std',sd.clamp_min(.01).clone())
    def encode_candidates(self,x):
        f,r=super().encode_candidates(x)
        return torch.maximum(torch.minimum(f,self.clip_mean+3*self.clip_std),self.clip_mean-3*self.clip_std),r


def proof():
    snapshot(ROOT);value=json.loads(MANIFEST.read_text())
    for p,h in value['files'].items():
        if file_sha(ROOT/p)!=h:raise ValueError('robustness source freeze mismatch: '+p)
    return {'sha256':file_sha(MANIFEST),'inventory_sha256':digest(value['files'])}


def load_p0(row,c):
    task,seed=row['task'],row['seed'];record=row['training']['creation'];identity=record['identity']
    dirname='diagnostic_v1' if (task,seed)==('single',123) else 'diagnostic_corrected_v2'
    path=ROOT/f'mist_transfer_runs/mist_cove/phase7_prospective_v1/{dirname}/{task}_{seed}_P0.pt'
    if identity['task']!=task or identity['seed']!=seed or identity['environment']!=environment('cpu'):raise ValueError('prospective P0 identity/environment mismatch')
    model=make_model('M1');opt=optimizer(model,c);g=torch.Generator()
    assert restore(path,record,identity,model,opt,g)==300
    return model


def paired_effects(model,c,task,seed,budget):
    result={}
    for condition in ('clean','competitive'):
        draw=generate(c['pair_examples'],seed+c['pair_offset'],task,condition)
        changed=generate(c['pair_examples'],seed+c['pair_offset'],task,condition,labels=(draw.y+1)%5)
        other=generate(c['pair_examples'],seed+c['pair_offset']+1,task,condition)
        replacement=compose(other.base,draw.events)
        a=predict(model,draw.x,budget).argmax(1);b=predict(model,changed.x,budget).argmax(1);d=predict(model,replacement,budget).argmax(1)
        result[condition]={'appropriate_diagnostic_change':float(((b!=a)&(b==changed.y)).float().mean()),
            'nuisance_agreement':float((d==a).float().mean()),'independent_groups':len(a)}
    return result


def summarize_model(model,c,task,seed,budget):
    clean,added=matched(c['evaluation_examples'],seed+c['locked_offset'],task,c)
    legacy=generate(c['evaluation_examples'],seed+c['locked_offset'],task,'competitive')
    result={name:score(predict(model,draw.x,budget),draw.y) for name,draw in [('clean',clean),('added_nuisance',added),('competitive',legacy)]}
    # Privilege is explicit, not a deployable competitor or guaranteed bound.
    f,_=encode(model,legacy.x)
    result['R3_privileged_oracle_pooling']=score(head(model,pooled(f,oracle_mask(legacy,contain=True))),legacy.y)
    result['R4_privileged_nuisance_removed']=score(predict(model,compose(legacy.background+legacy.noise,legacy.events),budget),legacy.y)
    logits=predict(model,legacy.base,budget);result['nuisance_only']=score(logits,legacy.y)
    result['nuisance_only']['accuracy_ci99']=wilson(int((logits.argmax(1)==legacy.y).sum()),len(legacy.y),2.576)
    # Original five-class metrics plus frozen K4 internal localization for M1.
    ec={**c,'criteria':{'wilson_z':2.576,'restricted_equivalence_low':.15,'restricted_equivalence_high':.25,'iou_hit':.25,'reconciliation':1e-5}}
    result['competitive_localization']=evaluate(model,legacy,ec,budget,localize=True)['localization']
    stress={}
    for condition in conditions_for(task,['removed','weak','repeated','correlated','position_shift','artifact','eventless','redundant','complementary','adjacent','overlap']):
        draw=generate(c['stress_examples'],seed+c['locked_offset'],task,condition)
        stress[condition]=score(predict(model,draw.x,budget),draw.y)
        stress[condition]['localization']=evaluate(model,draw,ec,budget,localize=True)['localization']
    for strength in (0,.25,.5,1,2):
        _,draw=matched(c['stress_examples'],seed+c['locked_offset'],task,c,strength=strength)
        stress['added_strength_'+str(strength)]=score(predict(model,draw.x,budget),draw.y)
        stress['added_strength_'+str(strength)]['localization']=evaluate(model,draw,ec,budget,localize=True)['localization']
    return {'conditions':result,'stress':stress,'pairs':paired_effects(model,c,task,seed,budget)}


def qualifies(row,c):
    k=c['criteria'];v=row['evaluation']['conditions'];p=row['evaluation']['pairs'];lo,hi=v['nuisance_only']['accuracy_ci99']
    clauses={'clean':v['clean']['macro_f1']>=k['clean_f1'],'combined_competitive':v['competitive']['macro_f1']>=k['competitive_f1'],
        'added_competitive':v['added_nuisance']['macro_f1']>=k['competitive_f1'],
        'diagnostic_pairs':all(a['appropriate_diagnostic_change']>=k['pair_appropriate'] for a in p.values()),
        'nuisance_pairs':all(a['nuisance_agreement']>=k['nuisance_agreement'] for a in p.values()),
        'no_collapse':all(not v[name]['collapsed'] for name in ['clean','competitive','added_nuisance']),
        'nuisance_equivalence':.15<=lo and hi<=.25}
    return {'pass':all(clauses.values()),'clauses':clauses}


def collective(rows,c):
    primary=[r for r in rows if r['method']==c['primary']]
    expected={(t,s) for t in c['tasks'] for s in c['seeds']}
    passed=len(primary)==len(expected) and {(r['task'],r['seed']) for r in primary}==expected and all(r['gate_clauses']['pass'] for r in primary)
    return {'status':'PASS' if passed else 'FAIL','primary':c['primary'],
            'checks':[{'task':r['task'],'seed':r['seed'],**r['gate_clauses']} for r in primary]}


def controlled_audit(model,draw,index,seed):
    """Same construction for both exact component deletions; no donor claim."""
    from mist_cove.phase6.generator import rng
    model.eval();ev=draw.events[index:index+1];base=draw.base[index:index+1];truth=draw.intervals[index]
    controls=torch.zeros_like(ev);sites=[];prng=rng(seed,96,index)
    for j,(a,b) in enumerate(truth):
        candidates=prng.permutation(list(range(0,2801,200))).tolist()
        lo=next(v for v in candidates if all(v+200<=cc or v>=dd for cc,dd in truth+sites))
        rms=float(ev[0,j,0,a:b].square().mean().sqrt())
        controls[0,j,0,lo:lo+200]=burst(30.,float(prng.uniform(0,6.283185307179586)),rms)
        sites.append((lo,lo+200))
    source=compose(base+controls.sum(1),ev)
    diag_removed=base+controls.sum(1);ctrl_removed=compose(base,ev)
    d=decompose(model,source,diag_removed,truth);n=decompose(model,source,ctrl_removed,sites)
    original_selection={tuple(v) for v in selected(model,source)[0]}
    generator=torch.Generator().manual_seed(seed+31071)
    perturbed=source+torch.randn(source.shape,generator=generator)*.01
    noisy_selection={tuple(v) for v in selected(model,perturbed)[0]}
    with torch.no_grad():
        original=model(source)[0];individual=[]
        for j in range(len(truth)):
            keep=torch.ones(1,2);keep[:,j]=0
            individual.append((original-model(compose(base+controls.sum(1),ev,keep))[0]).tolist())
    # Keep decompose's signed estimands unchanged and distinct.
    return {'diagnostic':d,'nondiagnostic':n,'individual_diagnostic_effects':individual,
            'same_component_intervention_construction':True,'diagnostic_intervals':truth,'control_intervals':sites,
            'tiny_noise_set_jaccard_distance':1-len(original_selection&noisy_selection)/max(1,len(original_selection|noisy_selection)),
            'construction':'exact known component removal for both groups; augmented synthetic controls',
            'donor_robustness':'NOT TESTED','physiological_validity':'NOT TESTED'}


def head_stage(gate,encoders,c,output,budget):
    if gate['status']!='PASS':return {'status':'NOT_TESTED_VII4_FAILED','fits':0,'updates':0,'rows':[]}
    rows=[]
    for task in c['tasks']:
        for seed in c['seeds']:
            a=generate(c['train_examples'],seed+c['train_offset'],task,'train');b=generate(c['train_examples'],seed+c['train_offset'],task,'competitive')
            mix=torch.where((torch.arange(len(a.y))%2==1)[:,None,None],b.x,a.x)
            local=[]
            for method in ('M1','M2','M3','M2_rms'):
                torch.manual_seed(seed+402);model=make_model(method);model.encoder.load_state_dict(encoders[(task,seed)]);model.encoder.requires_grad_(False)
                encoder_hash=tensor_digest(model.encoder.state_dict());f,r=encode(model,mix)
                cached=CachedHead(model);packed=torch.cat((f,r[:,:,None]),-1);opt=optimizer(model,c);g=torch.Generator().manual_seed(seed+711)
                started=time.perf_counter();loss=steps(cached,opt,g,packed,a.y,c,0,c['head_updates'],budget)
                identity={'experiment':c['version'],'task':task,'seed':seed,'method':method,'frozen_encoder_sha256':encoder_hash,
                          'config_sha256':file_sha(CONFIG),'implementation':proof(),'environment':environment('cpu'),
                          'train_digest':tensor_digest({'features':packed,'y':a.y})}
                creation=seal(output/f'{task}_{seed}_head_{method}.pt',model,opt,g,c['head_updates'],identity)
                training={'optimizer_updates':c['head_updates'],'example_exposures':c['head_updates']*c['batch_size'],'training_seconds':time.perf_counter()-started,
                          'final_loss':loss,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'creation':creation}
                assert tensor_digest(model.encoder.state_dict())==encoder_hash
                ec={**c,'criteria':{'wilson_z':2.576,'restricted_equivalence_low':.15,'restricted_equivalence_high':.25,'iou_hit':.25,'reconciliation':1e-5}}
                evaluations={}
                for condition in ('clean','competitive','removed','eventless'):
                    draw=generate(c['evaluation_examples'],seed+c['locked_offset'],task,condition)
                    evaluations[condition]=evaluate(model,draw,ec,budget,localize=True)
                draw=generate(100,seed+c['pair_offset'],task,'competitive');audits=[]
                for label in range(5):audits.append(controlled_audit(model,draw,int((draw.y==label).nonzero()[0]),seed))
                row={'task':task,'seed':seed,'method':method,'encoder_sha256':encoder_hash,'training':training,'conditions':evaluations,'audits':audits}
                rows.append(row);local.append(row)
                atomic_json({'status':'IN_PROGRESS','rows':rows},output/'heads_partial.json')
            assert len({v['encoder_sha256'] for v in local})==1
    # Honest lossless results; acceptance is evaluated by the public collector.
    return {'status':'COMPLETE','fits':len(rows),'updates':len(rows)*c['head_updates'],'rows':rows}


def run(path):
    c=json.loads(CONFIG.read_text());frozen=proof();output=output_directory(path,c['scope'])
    configure_precision('cpu');torch.set_num_threads(c['cpu_threads']);budget=Budget(c['cpu_wall_ceiling_seconds']);rows=[];encoders={};fits=0
    try:
        original=json.loads(DIAGNOSTICS.read_text())
        for prior in original['rows']:
            task,seed=prior['task'],prior['seed'];p0=load_p0(prior,c)
            train=generate(c['train_examples'],seed+c['train_offset'],task,'train');competitive=generate(c['train_examples'],seed+c['train_offset'],task,'competitive')
            f,_=encode(p0,train.x);mu,sd=f.mean((0,1)),f.std((0,1)).clamp_min(.01)
            mixed=torch.where((torch.arange(len(train.y))%2==1)[:,None,None],competitive.x,train.x)
            for method in ('P0','R1','R2'):
                if method=='P0':model=p0;training={'additional_updates':0,'original_prospective_creation':prior['training']['creation']}
                else:
                    torch.manual_seed(seed);model=make_model('M1') if method=='R1' else ClippedModel(mu,sd)
                    opt=optimizer(model,c);g=torch.Generator().manual_seed(seed+711)
                    identity={'experiment':c['version'],'method':method,'task':task,'seed':seed,'implementation':frozen,
                        'config_sha256':file_sha(CONFIG),'environment':environment('cpu'),'train_digest':tensor_digest({'x':mixed,'y':train.y}),
                        'p0_creation_sha256':digest(prior['training']['creation'])}
                    started=time.perf_counter();steps(model,opt,g,mixed,train.y,c,0,c['updates'],budget)
                    creation=seal(output/f'{task}_{seed}_{method}.pt',model,opt,g,c['updates'],identity);fits+=1
                    training={'updates':c['updates'],'exposures':c['updates']*c['batch_size'],'seconds':time.perf_counter()-started,'creation':creation,
                        'parameters':sum(p.numel() for p in model.parameters()),'fixed_clip_buffer_values':36 if method=='R2' else 0}
                evaluation=summarize_model(model,c,task,seed,budget)
                row={'task':task,'seed':seed,'method':method,'training':training,'evaluation':evaluation}
                row['gate_clauses']=qualifies(row,c);rows.append(row)
                if method=='R1':encoders[(task,seed)]=copy.deepcopy(model.encoder.state_dict())
                atomic_json({'status':'IN_PROGRESS','fits':fits,'rows':rows},output/'partial.json')
                print(task,seed,method,'clean',evaluation['conditions']['clean']['macro_f1'],'combined',evaluation['conditions']['competitive']['macro_f1'],'gate',row['gate_clauses']['pass'],flush=True)
        gate=collective(rows,c)
        heads=head_stage(gate,encoders,c,output,budget)
        assert proof()==frozen;budget.check()
        total=fits+heads['fits']
        assert total<=c['maximum_new_fits']
        result={'status':'COMPLETE_PROSPECTIVE_ROBUSTNESS','git_head':git(ROOT,'rev-parse','HEAD').decode().strip(),'frozen':frozen,'config':c,
            'rows':rows,'gate_vii_4':gate,'heads':heads,'new_fits':total,'optimizer_updates':fits*c['updates']+heads['updates'],
            'example_exposures':(fits*c['updates']+heads['updates'])*c['batch_size'],'wall_seconds':budget.elapsed,
            'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'historical_weights_loaded':0,'real_data_accessed':False}
        atomic_json(result,output/'results.json');return result
    except BaseException as error:
        atomic_json({'status':'STOPPED','completed_new_fits':fits,'error':repr(error),'wall_seconds':budget.elapsed},output/'STOPPED.json');raise

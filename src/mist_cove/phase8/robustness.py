"""Bounded locked comparison, calibration, reliability, failure and donor audits."""
import copy
import json
import resource
import time
import torch
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.provenance import file_sha,environment
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget,predict
from mist_cove.phase6.generator import generate,compose
from mist_cove.phase7.prospective import steps,optimizer,seal,restore,encode,oracle_mask,head
from mist_cove.phase7.robustness import paired_effects
from mist_cove.phase6.evaluation import location
from mist_evidence.runtime import atomic_json,atomic_torch_save,tensor_digest
from .provenance import CONFIG,PRIVATE,ROOT,freeze_proof,output_directory,trusted_reference
from .generator import TAXONOMY,STRENGTHS,intervene,training_draw,information_loss
from .pooling import TrimmedModel,pool
from .evaluation import reliability,gate,paired_accuracy_interval
from .calibration import fit_temperature,rejection_threshold,audit
from .representation import compare
from .evidence import matched_heads
from .interventions import donor_audit


def fail_analysis(model,draw,removed):
    with torch.no_grad():
        f,r=model.encode_candidates(draw.x);logits=model.decision(f,r)[0];conf=logits.softmax(1).amax(1);correct=logits.argmax(1)==draw.y
        rank=model.conventional_ranking(f,r);indices=rank.argsort(dim=1,descending=True,stable=True)[:,:4]
    records=[location([(j*200,j*200+400) for j in ix],t,n) for ix,t,n in zip(indices.tolist(),draw.intervals,draw.nuisance_intervals)]
    hit=torch.tensor([v['topk_hit']==1 for v in records]);false=torch.tensor([v['false_nuisance_fraction']>0 for v in records])
    with torch.no_grad():oracle=head(model,pool(f,'oracle',oracle_mask(draw,contain=True)))
    return {'definitions':'topK any IoU>=.25; nuisance selection overlaps nuisance without diagnostic hit; confidence>=.9; categories overlap',
        'localized_but_wrong':int((hit&~correct).sum()),'nuisance_localized_wrong':int((false&~correct).sum()),
        'correct_without_diagnostic_hit':int((correct&~hit).sum()),'high_confidence_error':int(((conf>=.9)&~correct).sum()),
        'deletion_without_confidence_reduction':int((removed.softmax(1).amax(1)>=conf).sum()),
        'oracle_correct_full_wrong':int(((oracle.argmax(1)==draw.y)&~correct).sum()),
        'class_collapse':float(torch.bincount(logits.argmax(1),minlength=5).max()/len(draw.y))>=.98,
        'n':len(draw.y),'localization':{k:sum(v[k] for v in records)/len(records) for k in ('top1_hit','topk_hit','precision','coverage','mean_best_iou','false_nuisance_fraction')},
        'first_error_index':int((~correct).nonzero()[0]) if (~correct).any() else None}


def run(path):
    c=json.loads(CONFIG.read_text());frozen=freeze_proof('confirmatory');out=output_directory(path,c['scope'])
    configure_precision('cpu');torch.set_num_threads(2);budget=Budget(c['confirmatory_budget_seconds'])
    rows=[];models={};training_records={};provenance=[];fit_count=0
    try:
        # All candidates train before any locked score is inspected.
        for task in c['tasks']:
            for seed in c['seeds']:
                for method in ('P0','R1','R2'):
                    budget.check();m,p=trusted_reference(task,seed,method);models[task,seed,method]=m;provenance.append(p)
                for method in ('B1','A1','T1'):
                    x,y=training_draw(c['train_examples'],seed+c['train_offset'],task,method)
                    torch.manual_seed(seed);m=TrimmedModel() if method=='T1' else make_model('M1');opt=optimizer(m,c);g=torch.Generator().manual_seed(seed+711)
                    identity={'experiment':c['version'],'method':method,'task':task,'seed':seed,'implementation':frozen,
                        'config_sha256':file_sha(CONFIG),'environment':environment('cpu'),'train_digest':tensor_digest({'x':x,'y':y})}
                    started=time.perf_counter();loss=steps(m,opt,g,x,y,c,0,c['updates'],budget);train_seconds=time.perf_counter()-started
                    checkpoint=out/f'{task}_{seed}_{method}.pt';started=time.perf_counter();creation=seal(checkpoint,m,opt,g,c['updates'],identity)
                    checkpoint_seconds=time.perf_counter()-started
                    m2=TrimmedModel() if method=='T1' else make_model('M1');o2=optimizer(m2,c);g2=torch.Generator()
                    started=time.perf_counter();restore(checkpoint,creation,identity,m2,o2,g2);restore_seconds=time.perf_counter()-started
                    if tensor_digest(m2.state_dict())!=tensor_digest(m.state_dict()):raise ValueError('exact checkpoint recovery failed')
                    training_records[task,seed,method]={'updates':c['updates'],'exposures':c['updates']*16,'train_seconds':train_seconds,
                        'checkpoint_write_seconds':checkpoint_seconds,'checkpoint_restore_seconds':restore_seconds,'final_loss':loss,
                        'parameters':sum(p.numel() for p in m.parameters()),'effective_trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
                        'encoder_sha256':tensor_digest(m.encoder.state_dict()),'creation':creation}
                    models[task,seed,method]=m2;fit_count+=1;print('fit',task,seed,method,flush=True)
                    atomic_json({'status':'TRAINING','completed_fits':fit_count},out/'progress.json')
        for task in c['tasks']:
            for seed in c['seeds']:
                clean=generate(c['evaluation_examples'],seed+c['locked_offset'],task)
                legacy=generate(c['evaluation_examples'],seed+c['locked_offset'],task,'competitive')
                added=intervene(clean,'high_frequency',1.)
                for method in c['methods']:
                    budget.check();m=models[task,seed,method];started=time.perf_counter()
                    conditions={};private={};logits_clean=None
                    for name,draw in [('clean',clean),('competitive',legacy),('added',added)]:
                        logits=predict(m,draw.x,budget);conditions[name]=reliability(logits,draw.y);private[name]=logits
                    # Separate scalar calibration / rejection fit, no locked outcomes.
                    cx,cy=training_draw(c['calibration_examples'],seed+c['calibration_offset'],task,'B1')
                    cl=predict(m,cx,budget);temperature=fit_temperature(cl,cy,role='SYNTHETIC_CALIBRATION')
                    threshold=rejection_threshold(cl/temperature,role='SYNTHETIC_CALIBRATION')
                    calibration={k:audit(v,clean.y,temperature,threshold) for k,v in private.items()}
                    loss_draw=generate(c['stress_examples'],seed+c['loss_offset'],task,'competitive');loss={}
                    for kind in ('intact','weak','partial','removed','unrelated','high_energy_remaining','background'):
                        xx=information_loss(loss_draw,kind);ll=predict(m,xx,budget);loss[kind]=audit(ll,loss_draw.y,temperature,threshold)
                    stress_clean=generate(c['stress_examples'],seed+c['stress_offset'],task);base_logits=predict(m,stress_clean.x,budget)
                    strengths={}
                    for kind in TAXONOMY:
                        sweep={}
                        for strength in STRENGTHS:
                            d=intervene(stress_clean,kind,strength);ll=predict(m,d.x,budget)
                            sweep[str(strength)]={'metrics':reliability(ll,d.y),'paired_accuracy_delta':paired_accuracy_interval(base_logits,ll,d.y,seed)}
                        strengths[kind]=sweep
                    sensitivity={}
                    for name in ('position_shift','overlap','adjacent','redundant','complementary','weak'):
                        d=generate(c['stress_examples'],seed+c['stress_offset'],task,name)
                        sensitivity[name]=reliability(predict(m,d.x,budget),d.y)
                    # Fixed independent secondary draw; no second candidate selection.
                    independent=generate(c['stress_examples'],seed+c['independent_offset'],task,'competitive')
                    independent_metrics=reliability(predict(m,independent.x,budget),independent.y)
                    pairs=paired_effects(m,c,task,seed,budget)
                    audit_draw=generate(100,seed+c['failure_offset'],task,'competitive')
                    removed=predict(m,audit_draw.base,budget);failures=fail_analysis(m,audit_draw,removed)
                    a,_=encode(m,stress_clean.x);corrupt=intervene(stress_clean,'transient',1.);b,_=encode(m,corrupt.x)
                    representation=compare(m,stress_clean,corrupt,a,b)
                    donors=donor_audit(m,task,seed,c,budget) if method==c['primary'] else None
                    atomic_torch_save({'logits':private,'labels':clean.y,'calibration_logits':cl,'calibration_labels':cy},out/f'{task}_{seed}_{method}_details.pt')
                    row={'task':task,'seed':seed,'method':method,'conditions':conditions,'pairs':pairs,'strengths':strengths,
                        'sensitivity':sensitivity,'independent_competitive':independent_metrics,'calibration':calibration,'information_loss':loss,
                        'failures':failures,'representation_transient':representation,'donors':donors,
                        'training':training_records.get((task,seed,method),{'new_updates':0,'reused_prospective':True,'parameters':3593}),
                        'evaluation_wall_seconds':time.perf_counter()-started,'inference_batch_size':64}
                    rows.append(row);atomic_json({'status':'EVALUATING','rows':rows},out/'partial.json')
                    print('evaluation',task,seed,method,'clean',conditions['clean']['macro_f1'],'competitive',conditions['competitive']['macro_f1'],flush=True)
        decision=gate(rows,c);heads={'status':'NOT TESTED','reason':'PRIMARY_ROBUSTNESS_GATE_FAILED','fits':0,'rows':[]}
        if decision['status']=='PASS':
            head_rows=[]
            for task in c['tasks']:
                for seed in c['seeds']:
                    x,y=training_draw(c['train_examples'],seed+c['head_train_offset'],task,'B1')
                    ms,rs=matched_heads(copy.deepcopy(models[task,seed,c['primary']].encoder.state_dict()),x,y,c,seed,budget,gate=decision)
                    for name,m in ms.items():
                        d=generate(c['evaluation_examples'],seed+c['head_eval_offset'],task,'competitive')
                        from mist_cove.phase6.evaluation import evaluate
                        ec={**c,'criteria':{'wilson_z':2.576,'restricted_equivalence_low':.15,'restricted_equivalence_high':.25,'iou_hit':.25,'reconciliation':1e-5}}
                        from mist_cove.phase7.robustness import controlled_audit
                        head_rows.append({'task':task,'seed':seed,'head':name,'receipt':rs[name],
                            'evaluation':evaluate(m,d,ec,budget,localize=True),
                            'audits':[controlled_audit(m,d,int((d.y==k).nonzero()[0]),seed) for k in range(5)]})
            heads={'status':'COMPLETE','fits':len(head_rows),'rows':head_rows}
        budget.check()
        if freeze_proof('confirmatory')!=frozen:raise ValueError('confirmatory source freeze changed')
        result={'status':'COMPLETE','frozen':frozen,'config':c,'rows':rows,'gate':decision,'heads':heads,'provenance':provenance,
            'new_representation_fits':fit_count,'new_head_fits':heads['fits'],'optimizer_updates':fit_count*c['updates']+heads['fits']*c['head_updates'],
            'example_exposures':(fit_count*c['updates']+heads['fits']*c['head_updates'])*16,
            'wall_seconds':budget.elapsed,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'real_data_accessed':False,'historical_phase6_deserializations':0,'git_implementation':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        atomic_json(result,out/'results.json');return result
    except BaseException as e:
        atomic_json({'status':'STOPPED','error':repr(e),'completed_fits':fit_count,'evaluated_rows':len(rows),'wall_seconds':budget.elapsed},out/'STOPPED.json');raise

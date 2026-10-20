"""Prespecified frozen-reference diagnosis; conclusions remain conditional."""
import json
import resource
import time
import torch
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.provenance import file_sha
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget,fit,predict
from mist_cove.phase7.prospective import encode,head,oracle_mask,feature_paths,LinearProbe
from mist_cove.phase6.generator import generate
from mist_evidence.runtime import atomic_json
from .provenance import CONFIG,PRIVATE,freeze_proof,output_directory,trusted_reference
from .generator import TAXONOMY,STRENGTHS,intervene
from .pooling import pool
from .representation import compare,separation
from .evaluation import reliability,paired_accuracy_interval


def run(path):
    c=json.loads(CONFIG.read_text());frozen=freeze_proof('diagnostic');out=output_directory(path,c['scope'])
    configure_precision('cpu');torch.set_num_threads(2);budget=Budget(c['diagnostic_budget_seconds']);rows=[];proofs=[]
    try:
        for task in c['tasks']:
            for seed in c['seeds']:
                model,proof=trusted_reference(task,seed,'P0');proofs.append(proof)
                clean=generate(c['diagnostic_examples'],seed+c['diagnostic_offset'],task);a,ar=encode(model,clean.x)
                baseline=head(model,pool(a));train=generate(500,seed+c['probe_train_offset'],task)
                train_corrupt=intervene(train,'high_frequency',1.);ta,_=encode(model,train.x);tb,_=encode(model,train_corrupt.x)
                # Fit only new generated TRAIN features, never diagnostic draws.
                mixed=torch.where((torch.arange(len(train.y))%2==0)[:,None,None],ta,tb)
                local=pool(mixed,'oracle',oracle_mask(train,contain=True))[:,:18];torch.manual_seed(seed+403)
                probe=LinearProbe(local);training=fit(probe,local,train.y,{**c,'updates':150},seed,budget)
                conditions={}
                for kind in TAXONOMY:
                    sweep={}
                    for strength in STRENGTHS:
                        budget.check();draw=intervene(clean,kind,strength);b,br=encode(model,draw.x);logits=head(model,pool(b))
                        row={'metrics':reliability(logits,draw.y),'paired_accuracy_delta':paired_accuracy_interval(baseline,logits,draw.y,seed),
                            'representation':compare(model,clean,draw,a,b)}
                        if strength==1.:
                            row['pooling']={method:reliability(head(model,pool(b,method,oracle_mask(clean,contain=True))),draw.y)
                                for method in ('mean','max','meanmax','trimmed','bounded','oracle')}
                            row['paths']=feature_paths(model,a,b,oracle_mask(clean))
                            row['privileged_train_probe']=reliability(predict(probe,pool(b,'oracle',oracle_mask(clean,contain=True))[:,:18],budget),draw.y)
                        sweep[str(strength)]=row
                    conditions[kind]=sweep
                eventless=generate(10,seed+c['diagnostic_offset'],task,'eventless');fe,_=encode(model,eventless.x)
                rows.append({'task':task,'seed':seed,'method':'P0','conditions':conditions,'probe_training':training,
                    'clean':reliability(baseline,clean.y),'eventless_feature_norm':float(fe.norm(dim=-1).mean())})
                atomic_json({'rows':rows,'status':'IN_PROGRESS'},out/'partial.json');print('diagnostic',task,seed,flush=True)
        result={'status':'COMPLETE','frozen':frozen,'rows':rows,'provenance':proofs,'new_probe_fits':len(rows),
            'optimizer_updates':len(rows)*150,'exposures':len(rows)*150*16,'wall_seconds':budget.elapsed,
            'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'real_data_accessed':False}
        if freeze_proof('diagnostic')!=frozen:raise ValueError('source changed during diagnostic run')
        atomic_json(result,out/'results.json');return result
    except BaseException as e:
        atomic_json({'status':'STOPPED','error':repr(e),'completed_rows':len(rows),'wall_seconds':budget.elapsed},out/'STOPPED.json');raise

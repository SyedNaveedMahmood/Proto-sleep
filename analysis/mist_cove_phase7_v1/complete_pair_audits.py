"""Complete descriptive per-pair exports; no fit or gate mutation is possible."""
import json
from pathlib import Path
import resource
import signal
import time
import torch
from mist_cove.phase2.provenance import file_sha,digest,environment
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase5.runtime import Budget,predict
from mist_cove.phase7.integrity import ROOT
from mist_cove.phase7.prospective import matched,encode,pooled,optimizer,restore,output_directory
from mist_cove.phase7.robustness import ClippedModel,proof,load_p0
from mist_cove.phase6.generator import compose
from mist_evidence.runtime import atomic_json,atomic_torch_save

MANIFEST=ROOT/'experiments/mist_cove/phase7_pair_audit_freeze_v1.json'


def verify():
    value=json.loads(MANIFEST.read_text())
    for p,h in value['files'].items():
        if file_sha(ROOT/p)!=h:raise ValueError('pair-audit freeze mismatch: '+p)
    return file_sha(MANIFEST)


def per_example(logits,y):
    p=logits.softmax(-1)
    return {'logits':logits,'probabilities':p,'prediction':p.argmax(1),
        'loss':torch.nn.functional.cross_entropy(logits,y,reduction='none'),
        'margin':logits.gather(1,y[:,None])[:,0]-logits.masked_fill(torch.nn.functional.one_hot(y,5).bool(),-float('inf')).max(1).values,
        'confidence':p.max(1).values,'entropy':-(p*p.clamp_min(1e-12).log()).sum(1)}


def main():
    def timeout(*_):raise TimeoutError('180-second pair-audit completion ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    configure_precision('cpu');torch.set_num_threads(2);frozen=verify();original=proof()
    output=output_directory(ROOT/'mist_transfer_runs/mist_cove/phase7_prospective_v1/pair_audits_v1','SYNTHETIC_ONLY')
    budget=Budget(180);records=[];started=time.perf_counter()
    diag=json.loads((ROOT/'reports/mist_cove/phase7_v1/prospective_diagnostics_v2.json').read_text())
    rows=[json.loads(v) for v in (ROOT/'reports/mist_cove/phase7_v1/robustness_runs_v1.jsonl').read_text().splitlines()]
    c=json.loads((ROOT/'configs/mist_cove_phase7_robustness_v1.json').read_text())
    try:
        for row in rows:
            task,seed,method=row['task'],row['seed'],row['method']
            if method=='P0':
                prior=next(v for v in diag['rows'] if v['task']==task and v['seed']==seed);model=load_p0(prior,c)
                creation=prior['training']['creation']
            else:
                creation=row['training']['creation'];identity=creation['identity']
                if identity['environment']!=environment('cpu'):raise ValueError('environment changed')
                model=make_model('M1') if method=='R1' else ClippedModel(torch.zeros(18),torch.ones(18))
                opt=optimizer(model,c);g=torch.Generator()
                path=ROOT/f'mist_transfer_runs/mist_cove/phase7_prospective_v1/robustness_v1/{task}_{seed}_{method}.pt'
                assert restore(path,creation,identity,model,opt,g)==300
            model.eval();a,b=matched(500,seed+700000,task,c);exports={};summaries={}
            fa,_=encode(model,a.x);fb,_=encode(model,b.x)
            feature={'candidate_drift_l2':(fa-fb).norm(dim=-1),'pooled_drift_l2':(pooled(fa)-pooled(fb)).norm(dim=-1),
                     'clean_candidate_norm':fa.norm(dim=-1),'competitive_candidate_norm':fb.norm(dim=-1)}
            for condition,draw in [('clean',a),('competitive',b)]:
                before=predict(model,draw.x,budget);removed=predict(model,draw.base,budget)
                joint=before-removed;individual=[]
                for member in range(1 if task=='single' else 2):
                    keep=torch.ones(len(draw.y),2);keep[:,member]=0
                    individual.append(before-predict(model,compose(draw.base,draw.events,keep),budget))
                individual=torch.stack(individual,1)
                exports[condition]={**per_example(before,draw.y),'joint_effect':joint,'individual_effects':individual,
                    'labels':draw.y,'diagnostic_intervals':draw.intervals,'native_event_scores':torch.zeros_like(individual),
                    'background_effect':joint,'fixed_event_path_effect':torch.zeros_like(joint),'reselection_effect':torch.zeros_like(joint)}
                true=joint.gather(1,draw.y[:,None])[:,0].abs()
                one=individual.gather(2,draw.y[:,None,None].expand(-1,individual.shape[1],1))[:,:,0].abs().amax(1)
                summaries[condition]={'examples':len(draw.y),'signed_joint_mean':joint.mean(0).tolist(),
                    'signed_individual_mean':individual.mean(0).tolist(),'absolute_joint_mean':float(joint.abs().mean()),
                    'true_class_absolute_joint_mean':float(true.mean()),'true_class_max_individual_mean':float(one.mean()),
                    'joint_exceeds_largest_individual_fraction':float((true>one).float().mean()),
                    'native_scores_are_zero':True,'physiological_validity':'NOT ESTABLISHED','donor_robustness':'NOT TESTED'}
            path=output/f'{task}_{seed}_{method}_pairs.pt'
            atomic_torch_save({'experiment':'phase7-pair-audit-completion-v1','checkpoint_creation':creation,'features':feature,'pairs':exports},path)
            atomic_json({'artifact_sha256':file_sha(path),'checkpoint_creation_sha256':digest(creation),'freeze_sha256':frozen},path.with_suffix('.seal.json'))
            records.append({'task':task,'seed':seed,'method':method,'conditions':summaries,
                'candidate_drift_l2_mean':float(feature['candidate_drift_l2'].mean()),'pooled_drift_l2_mean':float(feature['pooled_drift_l2'].mean()),
                'private_artifact_sha256':file_sha(path),'private_seal_sha256':file_sha(path.with_suffix('.seal.json'))})
        assert verify()==frozen and proof()==original;budget.check()
        report={'status':'COMPLETE_DESCRIPTIVE_POST_LOCK_EXPORT','freeze_sha256':frozen,'rows':records,
            'model_draw_pairs':18*500,'condition_specific_examples':18*500*2,'new_fits':0,'new_optimizer_updates':0,
            'wall_seconds':time.perf_counter()-started,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'primary_gate_remains':'FAIL','additive_head_fits':0,'historical_weight_deserializations':0,'real_data_accessed':False}
        atomic_json(report,output/'results.json')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    except BaseException as error:
        atomic_json({'status':'STOPPED','completed_model_audits':len(records),'error':repr(error),'wall_seconds':budget.elapsed},output/'STOPPED.json');raise


if __name__=='__main__':main()

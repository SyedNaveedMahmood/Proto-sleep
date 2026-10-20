"""Descriptive completion of requested strength/localization and group endpoints.

Registered after locked outcomes; adds no fit and cannot change any gate.
"""
import json
from pathlib import Path
import resource
import time
import torch
from collect import ROOT,PUBLIC,PRIVATE,sha,publish
from mist_cove.phase2.training import configure_precision
from mist_cove.phase8.provenance import trusted_reference,output_directory
from mist_cove.phase8.generator import TAXONOMY,STRENGTHS,intervene
from mist_cove.phase8.pooling import TrimmedModel
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate,compose
from mist_cove.phase6.evaluation import location
from mist_cove.phase7.prospective import restore,optimizer,encode,oracle_mask
from mist_evidence.runtime import atomic_json,tensor_digest
from mist_cove.phase5.runtime import Budget,predict


@torch.no_grad()
def local_summary(model,draw,clean_features,clean_logits):
    f,r=encode(model,draw.x);decision=model.decision(f,r);logits=decision[0]
    rank=model.conventional_ranking(f,r);ix=rank.argsort(dim=1,descending=True,stable=True)[:,:4]
    records=[location([(j*200,j*200+400) for j in indices],truth,nuisance) for indices,truth,nuisance in zip(ix.tolist(),draw.intervals,draw.nuisance_intervals)]
    mask=oracle_mask(draw);norm=clean_features.norm(dim=-1)*f.norm(dim=-1)
    cosine=(clean_features*f).sum(-1)/norm.clamp_min(1e-12)
    truth_features=f[mask];other_features=f[~mask]
    center_a=truth_features.mean(0);center_b=other_features.mean(0)
    within=(truth_features-center_a).square().sum(-1).mean()+(other_features-center_b).square().sum(-1).mean()
    pooled_before=model.decision(clean_features,r)[2];pooled_after=decision[2]
    weights=torch.cat((model.background_head.weight,model.event_head.weight),1)
    signed=(pooled_after-pooled_before)[:,:,None]*weights.T
    residual=float((signed.sum(1)-(logits-clean_logits)).abs().max())
    # Identical finite arithmetic, allow scale-dependent FP32 cancellation bounds.
    if residual>3e-5:raise ValueError('pooled-feature arithmetic mismatch')
    return {'prediction_change':float((logits.argmax(1)!=clean_logits.argmax(1)).float().mean()),
        'diagnostic_feature_cosine':float(cosine[mask].mean()),'diagnostic_feature_drift':float((f-clean_features).norm(dim=-1)[mask].mean()),
        'pooled_drift':float((pooled_after-pooled_before).norm(dim=-1).mean()),
        'diagnostic_nondiagnostic_fisher_ratio':float((center_a-center_b).square().sum()/within.clamp_min(1e-12)),
        'localization':{k:sum(v[k] for v in records)/len(records) for k in ('top1_hit','topk_hit','precision','coverage','mean_best_iou','false_nuisance_fraction')},
        'pooled_head_arithmetic_residual':residual,'signed_logit_change_mean':signed.sum(1).mean(0).tolist()}


@torch.no_grad()
def group_summary(model,draw):
    before=predict(model,draw.x);joint=before-predict(model,draw.base);individual=[]
    for j in range(2 if draw.task=='multiple' else 1):
        keep=torch.ones(len(draw.y),2);keep[:,j]=0
        individual.append(before-predict(model,compose(draw.base,draw.events,keep)))
    idx=torch.arange(len(draw.y));true_joint=joint[idx,draw.y];true_ind=torch.stack([v[idx,draw.y] for v in individual],1)
    largest=true_ind.abs().amax(1);excess=true_joint.abs()-largest
    return {'construction':'exact generator component deletion, not waveform donors or native additive evidence',
        'mean_abs_true_joint':float(true_joint.abs().mean()),'mean_largest_abs_individual':float(largest.mean()),
        'joint_exceeds_largest_fraction':float((excess>0).float().mean()),
        'joint_exceeds_largest_by_0_25_fraction':float((excess>.25).float().mean()),
        'joint_true_effect_quantiles':torch.quantile(true_joint,torch.tensor([0.,.1,.25,.5,.75,.9,1.])).tolist(),
        'joint_minus_sum_individual_abs_mean':float((joint-sum(individual)).abs().mean()),
        'n':len(draw.y)}


def main():
    manifest=ROOT/'experiments/mist_cove/phase8_descriptive_completion_freeze_v1.json'
    frozen=json.loads(manifest.read_text())
    import subprocess
    if subprocess.check_output(['git','show',f'HEAD:{manifest.relative_to(ROOT)}'],cwd=ROOT)!=manifest.read_bytes():raise ValueError('commit appendix freeze before execution')
    for relative,expected in frozen['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('appendix source mismatch')
    result_path=PRIVATE/'confirmatory_v1/results.json'
    if sha(result_path)!=frozen['private_result_sha256']:raise ValueError('completed campaign receipt changed')
    original=json.loads(result_path.read_text());c=original['config'];configure_precision('cpu');torch.set_num_threads(2)
    out=output_directory(PRIVATE/'descriptive_completion_v1');budget=Budget(frozen['remaining_budget_seconds']);rows=[]
    for row in original['rows']:
        task,seed,method=row['task'],row['seed'],row['method'];budget.check()
        if method in ('P0','R1','R2'):model,proof=trusted_reference(task,seed,method)
        else:
            creation=row['training']['creation'];path=PRIVATE/f'confirmatory_v1/{task}_{seed}_{method}.pt'
            if json.loads(path.with_suffix('.creation.json').read_text())!=creation:raise ValueError('creation sidecar mismatch')
            model=TrimmedModel() if method=='T1' else make_model('M1');opt=optimizer(model,c);g=torch.Generator()
            restore(path,creation,creation['identity'],model,opt,g);model.eval()
            if tensor_digest(model.encoder.state_dict())!=row['training']['encoder_sha256']:raise ValueError('encoder mismatch')
        clean=generate(c['stress_examples'],seed+c['stress_offset'],task);a,r=encode(model,clean.x)
        with torch.no_grad():baseline=model.decision(a,r)[0]
        conditions={}
        for kind in TAXONOMY:
            conditions[kind]={}
            for strength in STRENGTHS:
                budget.check();draw=intervene(clean,kind,strength)
                conditions[kind][str(strength)]=local_summary(model,draw,a,baseline)
        groups={}
        for condition in ('clean','competitive','redundant','complementary'):
            if task=='single' and condition in ('redundant','complementary'):continue
            draw=generate(c['stress_examples'],seed+c['stress_offset'],task,condition);groups[condition]=group_summary(model,draw)
        weights=torch.cat((model.background_head.weight,model.event_head.weight),1).detach()
        rows.append({'task':task,'seed':seed,'method':method,'strengths':conditions,'groups':groups,
            'head_direction_sensitivity':{'per_logit_l2_gradient_norm':weights.norm(dim=1).tolist(),
                'per_logit_l1_gradient_norm':weights.abs().sum(1).tolist(),'interpretation':'exact affine pooled-feature sensitivity; not waveform causality'}})
        atomic_json({'status':'IN_PROGRESS','rows':rows},out/'partial.json');print('appendix',task,seed,method,flush=True)
    result={'status':'COMPLETE','classification':'DESCRIPTIVE_AFTER_LOCKED_RESULTS','freeze_sha256':sha(manifest),'rows':rows,'new_fits':0,'new_optimizer_updates':0,
        'wall_seconds':budget.elapsed,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'real_data_accessed':False,'gate_changed':False}
    atomic_json(result,out/'results.json');publish(PUBLIC/'descriptive_diagnostics_summary.json',{**result,'private_receipt_sha256':sha(out/'results.json')})

if __name__=='__main__':main()

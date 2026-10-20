"""Private synthetic trace previews plus public aggregate appendix charts."""
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import ROOT,PUBLIC,PRIVATE,publish,sha
from mist_cove.phase8.provenance import output_directory
from mist_cove.phase2.training import configure_precision
from mist_cove.phase4.model import make_model
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import restore,optimizer
from mist_cove.phase6.evaluation import selected


def main():
    configure_precision('cpu');torch.set_num_threads(2)
    original=json.loads((PRIVATE/'confirmatory_v1/results.json').read_text());c=original['config'];out=output_directory(PRIVATE/'synthetic_examples_v1');count=0
    for row in original['rows']:
        if row['method']!='B1':continue
        task,seed=row['task'],row['seed'];creation=row['training']['creation'];checkpoint=PRIVATE/f'confirmatory_v1/{task}_{seed}_B1.pt'
        if json.loads(checkpoint.with_suffix('.creation.json').read_text())!=creation:raise ValueError('creation sidecar mismatch')
        m=make_model('M1');restore(checkpoint,creation,creation['identity'],m,optimizer(m,c),torch.Generator());m.eval()
        d=generate(100,seed+c['failure_offset'],task,'competitive')
        with torch.no_grad():logits=m(d.x)
        indices=[('prespecified_class_'+str(k),int((d.y==k).nonzero()[0])) for k in range(5)]
        wrong=(logits.argmax(1)!=d.y).nonzero()
        if len(wrong):indices.append(('deterministic_first_error',int(wrong[0])))
        for name,i in indices:
            chosen=selected(m,d.x[i:i+1])[0];fig,ax=plt.subplots(figsize=(11,3));t=np.arange(3000)/100
            ax.plot(t,d.x[i,0],lw=.6,color='black')
            for a,b in d.intervals[i]:ax.axvspan(a/100,b/100,color='green',alpha=.3)
            for a,b in chosen:ax.axvspan(a/100,b/100,color='orange',alpha=.15)
            ax.set(title=f'Generated {task}/{seed}: {name}; code={int(d.y[i])}, prediction={int(logits[i].argmax())}; green truth/orange internal K4',xlabel='Seconds',ylabel='Synthetic uV')
            fig.tight_layout();fig.savefig(out/f'{task}_{seed}_{name}.png');plt.close(fig);count+=1
    r=json.loads((PUBLIC/'descriptive_diagnostics_summary.json').read_text());paths=[]
    fig,ax=plt.subplots(figsize=(8,4));methods=['P0','R1','R2','B1','A1','T1']
    for j,m in enumerate(methods):
        values=np.array([v['groups']['competitive']['joint_true_effect_quantiles'] for v in r['rows'] if v['method']==m and v['task']=='multiple'])
        # Mean per-seed quantiles are descriptive; not a pooled quantile estimator.
        mean=values.mean(0);ax.plot([j,j],[mean[1],mean[5]],lw=3);ax.scatter(j,mean[3],marker='o')
    ax.set(xticks=range(6),xticklabels=methods,ylabel='True-code full joint component logit effect',title='Multiple-task competitive: mean seed median and 10–90% quantiles')
    fig.tight_layout();p=PUBLIC/'figures/15_group_effect_quantiles.png';fig.savefig(p);plt.close(fig);paths.append(p)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,('topk_hit','false_nuisance_fraction')):
        for m in ('P0','B1','T1'):
            values=np.array([[v['strengths']['high_frequency'][s]['localization'][metric] for s in ('0.0','0.5','1.0','2.0')] for v in r['rows'] if v['method']==m])
            ax.errorbar([0,.5,1,2],values.mean(0),yerr=values.std(0,ddof=1),label=m,marker='o',capsize=2)
        ax.set(title=metric,xlabel='Added high-frequency strength',ylabel='Fraction',ylim=(0,1.05));ax.legend()
    fig.tight_layout();p=PUBLIC/'figures/16_strength_localization.png';fig.savefig(p);plt.close(fig);paths.append(p)
    publish(PUBLIC/'descriptive_figure_manifest.json',{'input_sha256':sha(PUBLIC/'descriptive_diagnostics_summary.json'),
        'figures':{p.name:sha(p) for p in paths},'private_synthetic_example_figures':count,
        'example_selection':'first generated draw per class plus first error, each task/seed primary B1',
        'public_waveform_exports':0,'real_data_accessed':False})

if __name__=='__main__':main()

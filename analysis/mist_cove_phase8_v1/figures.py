"""Deterministic aggregate figures; never opens a model or generated waveform."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import PUBLIC,sha,publish


def main():
    paths={name:PUBLIC/name for name in ('robustness_summary.json','diagnostic_summary.json','calibration_summary.json','evidence_summary.json','execution_receipt.json')}
    receipts={k:sha(v) for k,v in paths.items()};r=json.loads(paths['robustness_summary.json'].read_text());d=json.loads(paths['diagnostic_summary.json'].read_text());cal=json.loads(paths['calibration_summary.json'].read_text());e=json.loads(paths['evidence_summary.json'].read_text());comp=json.loads(paths['execution_receipt.json'].read_text())
    rows=r['rows'];methods=r['config']['methods'];out=PUBLIC/'figures';out.mkdir(exist_ok=False);figures=[]
    plt.rcParams.update({'font.size':9,'figure.dpi':120,'savefig.dpi':150})
    def save(name,fig):
        fig.tight_layout();fig.savefig(out/(name+'.png'),metadata={'Software':'MIST-COVE aggregate renderer v1'});plt.close(fig);figures.append(name+'.png')
    fig,ax=plt.subplots(figsize=(8,4))
    ix=np.arange(len(methods))
    for j,name in enumerate(('clean','competitive')):
        means=[np.mean([v['conditions'][name]['macro_f1'] for v in rows if v['method']==m]) for m in methods]
        sd=[np.std([v['conditions'][name]['macro_f1'] for v in rows if v['method']==m],ddof=1) for m in methods]
        ax.bar(ix+(j-.5)*.35,means,.35,yerr=sd,label=name,capsize=3)
    ax.set(xticks=ix,xticklabels=methods,ylabel='Fixed-five Macro-F1',ylim=(0,1.08),title='Generated task/seed mean and descriptive SD');ax.legend();save('01_clean_corrupted',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,('single','multiple')):
        for method in methods:
            vs=[v for v in rows if v['method']==method and v['task']==task]
            values=np.array([[v['strengths']['high_frequency'][s]['metrics']['macro_f1'] for s in ('0.0','0.5','1.0','2.0')] for v in vs])
            ax.errorbar([0,.5,1,2],values.mean(0),yerr=values.std(0,ddof=1),label=method,marker='o',capsize=2)
        ax.set(title=task,xlabel='Added HF strength (40 uV RMS units)',ylabel='Macro-F1',ylim=(0,1.05))
    axes[1].legend(fontsize=7);save('02_strength',fig)
    kinds=['transient','impulse','broadband','overlapping','disjoint','boundary','crossing_candidate']
    matrix=np.array([[np.mean([v['strengths'][k]['1.0']['metrics']['macro_f1'] for v in rows if v['method']==m]) for k in kinds] for m in methods])
    fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ax.set(yticks=range(len(methods)),yticklabels=methods,xticks=range(len(kinds)),xticklabels=kinds,title='Strength1 artifact/geometry mean F1');ax.tick_params(axis='x',rotation=30);fig.colorbar(im,ax=ax);save('03_artifact_types',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for m in methods:ax.plot(range(5),np.mean([v['conditions']['competitive']['per_stage_recall'] for v in rows if v['method']==m],0),marker='o',label=m)
    ax.set(xticks=range(5),xlabel='Synthetic class code (not clinical sleep stage)',ylabel='Recall',ylim=(0,1.05));ax.legend();save('04_class_recall',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for m in methods:
        vs=[v for v in rows if v['method']==m];x=[v['conditions']['clean']['macro_f1'] for v in vs];y=[v['conditions']['competitive']['macro_f1'] for v in vs]
        ax.scatter(x,y,label=m);ax.scatter(np.mean(x),np.mean(y),marker='X',s=110)
    ax.axhline(.7,color='gray',ls='--');ax.set(xlabel='Clean Macro-F1',ylabel='Combined competitive Macro-F1',title='Pareto coordinates: points task/seed, X means');ax.legend();save('05_pareto',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,mode in zip(axes,('raw','temperature_scaled')):
        for m in methods:
            bins=[v['calibration']['competitive'][mode]['reliability_bins'] for v in cal['rows'] if v['method']==m];x=[];y=[]
            for j in range(15):
                count=sum(b[j]['count'] for b in bins)
                if count:x.append(sum(b[j]['count']*(b[j]['confidence'] or 0) for b in bins)/count);y.append(sum(b[j]['count']*(b[j]['accuracy'] or 0) for b in bins)/count)
            ax.plot(x,y,marker='.',label=m)
        ax.plot([0,1],[0,1],'k--');ax.set(title=mode,xlabel='Mean confidence',ylabel='Generated accuracy',xlim=(0,1),ylim=(0,1))
    axes[1].legend(fontsize=7);save('06_calibration',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(methods,[np.mean([v['conditions']['competitive']['false_confidence_0_9'] for v in rows if v['method']==m]) for m in methods]);ax.set(ylabel='Fraction incorrect with confidence >= .9',title='Original combined corruption');save('07_incorrect_high_confidence',fig)
    kinds=list(d['rows'][0]['conditions']);fig,ax=plt.subplots(figsize=(10,4))
    for group in ('diagnostic_drift','pooled_drift'):ax.plot(range(len(kinds)),[np.mean([v['conditions'][k]['1.0']['representation'][group] for v in d['rows']]) for k in kinds],marker='.',label=group)
    ax.set(xticks=range(len(kinds)),xticklabels=kinds,ylabel='Feature-space L2 drift',title='Frozen P0 matched strength1');ax.tick_params(axis='x',rotation=60);ax.legend();save('08_candidate_drift',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for metric in ('topk_hit','false_nuisance_fraction'):ax.plot(methods,[np.mean([v['failures']['localization'][metric] for v in rows if v['method']==m]) for m in methods],marker='o',label=metric)
    ax.set(ylabel='Fraction',ylim=(0,1.05),title='Internal candidate-deletion ranking; independent 100-draw audit');ax.legend();save('09_localization',fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for method in ('meanmax','trimmed','bounded','oracle'):ax.plot(kinds,[np.mean([v['conditions'][k]['1.0']['pooling'][method]['macro_f1'] for v in d['rows']]) for k in kinds],marker='.',label=method)
    ax.set(ylabel='Macro-F1',title='Frozen P0 unchanged-head pooling; oracle is privileged');ax.tick_params(axis='x',rotation=60);ax.legend();save('10_oracle_pooling',fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for feature in ('pooled_separation','privileged_local_separation'):ax.plot(kinds,[np.mean([v['conditions'][k]['1.0']['representation'][feature]['ratio'] for v in d['rows']]) for k in kinds],label=feature,marker='.')
    ax.set(ylabel='Between/within squared-distance ratio',title='Frozen P0 feature separability');ax.tick_params(axis='x',rotation=60);ax.legend();save('11_separability',fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for m in ('P0','B1','A1','T1'):
        values=[s['paired_accuracy_delta']['mean'] for v in rows if v['method']==m for k,levels in v['strengths'].items() for strength,s in levels.items() if strength=='1.0']
        ax.hist(values,bins=np.linspace(-1,1,21),histtype='step',label=m,linewidth=1.5)
    ax.set(xlabel='Paired generated accuracy delta vs matched clean',ylabel='Correlated task/seed/condition summaries',title='Descriptive intervention-effect distribution (not independent trials)');ax.legend();save('12_intervention_effects',fig)
    fig,ax=plt.subplots(figsize=(8,4));reasons=['envelope','energy','spectrum','boundary'];populations=['train_baseline','bridge','oracle_neutral']
    for j,pop in enumerate(populations):
        counts=[sum(v['donors']['population'][pop]['rejections'][k] for v in e['donors']) for k in reasons]
        ax.bar(np.arange(4)+(j-1)*.25,counts,.25,label=pop)
    ax.set(xticks=range(4),xticklabels=reasons,ylabel='Rejected attempts (reasons overlap)',title='Independent synthetic donor compatibility');ax.legend();save('13_donor_rejection',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4));new=['B1','A1','T1']
    axes[0].bar(new,[sum(v['training']['train_seconds'] for v in rows if v['method']==m) for m in new]);axes[0].set(ylabel='Summed measured CPU training seconds')
    axes[1].bar(methods,[3593]*6);axes[1].set(ylabel='Declared trainable parameters',title='Encoder3408 + head185');save('14_compute_parameters',fig)
    publish(PUBLIC/'figure_manifest.json',{'inputs':receipts,'figures':{name:sha(out/name) for name in figures},'scope':'AGGREGATE_SYNTHETIC_ONLY'})

if __name__=='__main__':main()

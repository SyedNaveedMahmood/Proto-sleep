"""Six aggregate scientific figures; no waveforms or participant identifiers."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .provenance import PUBLIC
from .statistics import rows


def run():
    rr=rows();figdir=PUBLIC/'figures';figdir.mkdir(exist_ok=True);seeds=(123,456,789)
    colors={'S_R0':'#377eb8','S_R1':'#4daf4a','L_R0':'#984ea3','L_R1':'#e41a1c'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for size in ('S','L'):
        for route in ('R0','R1'):
            label=size+'_'+route
            for k in seeds:
                a=[rr[size,route,k,t]['development']['clean']['metrics'] for t in (300,1000,3000)]
                ax[0].plot([300,1000,3000],[v['mean_subject_macro_f1'] for v in a],color=colors[label],alpha=.5,marker='.',label=label if k==123 else None)
                ax[1].plot([300,1000,3000],[v['per_stage_recall'][1] for v in a],color=colors[label],alpha=.5,marker='.')
    ax[0].set(ylabel='Mean participant Macro-F1',xlabel='Optimizer updates',title='All three seeds; reused TRAIN10/5')
    ax[1].set(ylabel='N1 recall',xlabel='Optimizer updates',title='More exposure recovers N1');ax[0].legend(ncol=2)
    save(fig,'01_optimization_and_N1')
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for size in ('S','L'):
        for route in ('R0','R1'):
            label=size+'_'+route
            for k in seeds:
                r=rr[size,route,k,3000];f=r['development']['clean']['metrics']['mean_subject_macro_f1']
                ax[0].scatter(r['counts']['trainable'],f,color=colors[label],label=label if k==123 else None)
                ax[1].scatter(r['trajectory_wall_to_milestone'],f,color=colors[label])
    ax[0].set(xlabel='Active trainable parameters',ylabel='Mean participant Macro-F1',title='Fixed3,000updates');ax[0].legend()
    ax[1].set(xlabel='Measured trajectory CPU seconds',ylabel='Mean participant Macro-F1',title='Includes milestone evaluation');save(fig,'02_capacity_and_compute')
    fig,ax=plt.subplots(2,2,figsize=(9,7))
    for axis,(size,route) in zip(ax.flat,[(s,r) for s in ('S','L') for r in ('R0','R1')]):
        cm=sum(np.array(rr[size,route,k,3000]['development']['clean']['metrics']['confusion_matrix']) for k in seeds);cm=cm/cm.sum(1,keepdims=True)
        axis.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):axis.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color='white' if cm[i,j]>.55 else 'black')
        axis.set(xticks=range(5),yticks=range(5),xticklabels=['W','N1','N2','N3','R'],yticklabels=['W','N1','N2','N3','R'],xlabel='Predicted',ylabel='Scored',title=size+'_'+route+'; seeds pooled descriptively')
    save(fig,'03_stage_confusions')
    u=json.loads((PUBLIC/'uncertainty_summary.json').read_text())['contrasts'];views=list(u['routing_large'])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for label,offset,color in [('routing_small',-.12,'#377eb8'),('routing_large',.12,'#984ea3')]:
        a=[u[label][v] for v in views];means=np.array([r['mean'] for r in a]);bounds=np.array([r['participant_cluster_bootstrap_95_percent'] for r in a])
        ax[0].errorbar(np.arange(len(views))+offset,means,yerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',label=label,color=color)
    ax[0].axhline(0,color='gray');ax[0].set(xticks=range(len(views)),xticklabels=['clean','mild','strong','global','drift','gain','broad'],ylabel='R1 minus R0 participant Macro-F1',title='Participant-cluster95% intervals');ax[0].legend()
    for j,(s,r) in enumerate([(s,r) for s in ('S','L') for r in ('R0','R1')]):
        a=[rr[s,r,k,3000]['development']['clean']['metrics']['calibration'] for k in seeds]
        ax[1].scatter([j]*3,[v['ECE15'] for v in a],label=s+'_'+r,color=colors[s+'_'+r])
    ax[1].set(xticks=range(4),xticklabels=['S_R0','S_R1','L_R0','L_R1'],ylabel='ECE15',title='Calibration; three seeds per family');save(fig,'04_routing_and_calibration')
    e=json.loads((PUBLIC/'explanation_validity.json').read_text());fig,ax=plt.subplots(1,3,figsize=(12,3.8))
    for j,(s,r) in enumerate([(s,r) for s in ('S','L') for r in ('R0','R1')]):
        for step,delta,marker in [(300,-.12,'x'),(3000,.12,'o')]:
            aa=[v['all'] for v in e['new_states'] if (v['capacity'],v['routing'],v['step'])==(s,r,step)]
            for axis,key in zip(ax,('displayed_nonempty_compact_active','global_stable_local_changed','correct')):
                axis.scatter([j+delta]*3,[a[key]/25 for a in aa],marker=marker,color=colors[s+'_'+r],label=str(step) if j==0 else None)
    for axis,title in zip(ax,('Displayed compact active support','Global stable / local changed','25-case panel accuracy')):
        axis.set(xticks=range(4),xticklabels=['S_R0','S_R1','L_R0','L_R1'],ylim=(-.03,1.03),ylabel='Fraction of25cases',title=title)
    ax[0].legend(title='Updates');save(fig,'05_support_and_global_dependence')
    original=e['primary_original'];fig,ax=plt.subplots(1,2,figsize=(10,3.8));vv=list(original['views']);pos=np.arange(3)
    ax[0].bar(pos-.15,[original['views'][v]['prediction_stable']/150 for v in vv],.3,label='Prediction stable')
    ax[0].bar(pos+.15,[original['views'][v]['fixed_clause_valid']/150 for v in vv],.3,label='Original clause valid')
    ax[0].set(xticks=pos,xticklabels=vv,ylim=(0,1.05),title='Original P3B150new primary cases',ylabel='Fraction of150');ax[0].legend()
    values=[original['neutral']['any_table_change'],original['neutral']['positive_minimum_changed'],original['physical']['bridge']['historically_compatible'],original['physical']['bridge']['operator_disagreements']]
    ax[1].bar(range(4),values);ax[1].set(xticks=range(4),xticklabels=['neutral\ncase changes','minimum\nmask changes','compatible\nbridges','operator\ndisagreements'],ylabel='Counts; denominators differ',title='Neutral150cases; bridges2,250attempts');save(fig,'06_original_stability_and_operators')

if __name__=='__main__':run()

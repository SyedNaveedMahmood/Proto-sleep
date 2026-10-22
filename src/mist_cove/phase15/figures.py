"""Six aggregate, data-derived Phase XV multi-panel figures."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .governance import PUBLIC,OLD_PUBLIC,LABEL,read,new


STAGES=['W','N1','N2','N3','REM']
ERRORS=['correct_REM','REM_to_W','REM_to_N1','REM_to_N2']
COLORS=['#238b8d','#ca6937','#825ea8','#d1a034']


def _save(fig,name):
    path=PUBLIC/'figures'/name;path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
    return str(path.relative_to(PUBLIC.parent.parent.parent))


def run():
    st=read(OLD_PUBLIC/'st_classification_metrics.json')['metrics']
    sc=read(OLD_PUBLIC/'sc_classification_metrics.json')['metrics']
    rem=read(PUBLIC/'rem_forensics.json');wave=read(PUBLIC/'rem_wave_diagnostics.json')['summary']
    stats=read(PUBLIC/'rem_statistics.json');comparison=read(PUBLIC/'explanation_comparison.json')
    stability=read(PUBLIC/'stability_results.json');operator=read(PUBLIC/'operator_inheritance.json')
    quality=read(PUBLIC/'quality_results.json');cost=read(PUBLIC/'explanation_execution.json')
    made=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    cm=np.array(rem['confusion_matrix']);im=ax[0].imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
    ax[0].set(xticks=range(5),yticks=range(5),xticklabels=STAGES,yticklabels=STAGES,xlabel='Predicted',ylabel='Scored',title='ST full-grid confusion (row normalized)')
    fig.colorbar(im,ax=ax[0],fraction=.045)
    x=np.arange(5);ax[1].bar(x-.18,sc['per_stage_recall'],.36,label='SC TRAIN in sample',color='#238b8d')
    ax[1].bar(x+.18,st['per_stage_recall'],.36,label='ST reused',color='#ca6937')
    ax[1].set(xticks=x,xticklabels=STAGES,ylim=(0,1),ylabel='Recall',title='Cross-study stage recall');ax[1].legend(fontsize=7)
    counts=cm[4];ax[2].bar(STAGES,counts,color='#825ea8');ax[2].set(ylabel='Scored REM epochs',title='REM predicted-stage counts')
    made.append(_save(fig,'01_REM_confusion_and_transfer.png'))
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained');x=np.arange(4)
    metrics=[('EEG_Fpz_Cz','sigma_fraction','Fpz-Cz sigma fraction'),('EEG_Pz_Oz','sigma_fraction','Pz-Oz sigma fraction'),
             ('EEG_Fpz_Cz','spectral_entropy','Fpz-Cz spectral entropy'),('EOG','excursion_count','EOG excursion count'),
             ('ST_EMG','within_night_rank','ST EMG within-night rank')]
    for a,(group,key,title) in zip(ax.flat[:5],metrics):
        vals=[wave[s][group][key]['median'] for s in ERRORS]
        a.bar(x,vals,color=COLORS);a.set(xticks=x,xticklabels=['correct','to W','to N1','to N2'],title=title)
    rates=[rem['taxonomy'][s]['within_2_epochs_of_transition']/rem['taxonomy'][s]['n'] for s in ERRORS]
    ax.flat[5].bar(x,rates,color=COLORS);ax.flat[5].set(xticks=x,xticklabels=['correct','to W','to N1','to N2'],title='Within 2 scored epochs of transition',ylim=(0,.4))
    made.append(_save(fig,'02_REM_multichannel_and_transitions.png'))
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    ece=[stats['stage_calibration'][str(j)]['ECE15'] for j in range(5)]
    ax[0].bar(STAGES,ece,color='#238b8d');ax[0].axhline(stats['full_grid_ECE15'],color='#ca6937',ls='--',label='pooled');ax[0].set(title='15-bin calibration error',ylabel='ECE');ax[0].legend(fontsize=7)
    b=stats['full_grid_calibration_bins'];ax[1].scatter([r['confidence'] for r in b if r['n']],[r['accuracy'] for r in b if r['n']],s=[max(5,r['n']/80) for r in b if r['n']],color='#825ea8')
    ax[1].plot([0,1],[0,1],color='grey',ls='--');ax[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Accuracy',title='ST reliability (bubble ∝ bin n)')
    w=stats['REM_warning_indicators'];names=list(w);ax[2].bar(np.arange(len(names))-.18,[w[k]['sensitivity'] for k in names],.36,label='Sensitivity',color='#ca6937')
    ax[2].bar(np.arange(len(names))+.18,[w[k]['false_warning_rate'] for k in names],.36,label='False warning',color='#238b8d')
    ax[2].set(xticks=np.arange(len(names)),xticklabels=['margin','entropy','high conf','transition'],ylim=(0,1),title='Fixed REM error indicators');ax[2].legend(fontsize=7)
    made.append(_save(fig,'03_calibration_and_warning_indicators.png'))
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    panels=comparison['panels'];pn=['E1','E2','E3'];ax[0].bar(pn,[panels[p]['cases'] for p in pn],color='#d0d0d0',label='all')
    ax[0].bar(pn,[panels[p]['eligible_cases'] for p in pn],color='#238b8d',label='eligible');ax[0].set(title='Proxy-eligible denominators',ylabel='Cases');ax[0].legend(fontsize=7)
    methods=['exact_positive','sigma_matched','RMS_matched','gradient_matched','random_matched'];vals=[panels['E2']['methods']['0'][m]['mean_precision'] for m in methods]
    ax[1].bar(range(len(methods)),vals,color=['#825ea8','#ca6937','#238b8d','#d1a034','#777777']);ax[1].set(xticks=range(len(methods)),xticklabels=['exact','sigma','RMS','gradient','random'],ylabel='Mean interval precision',title='Matched 2-s atoms, E2 spindle proxy')
    dif=panels['E2']['paired_differences']['exact_minus_sigma_matched'];ax[2].errorbar([0],[dif['person_mean_delta']],yerr=[[dif['person_mean_delta']-dif['bootstrap95'][0]],[dif['bootstrap95'][1]-dif['person_mean_delta']]],fmt='o',color='#ca6937')
    ax[2].axhline(0,color='black',lw=.8);ax[2].set(xlim=(-.5,.5),xticks=[0],xticklabels=['exact − sigma'],ylabel='Participant mean precision difference',title='1000 person-bootstrap draws')
    made.append(_save(fig,'04_explanation_sigma_challenge.png'))
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    e1=panels['E1'];names=['nonmonotone_cases','active_compact_signed','neutral_changed_tables','prior_response','whole_epoch_positive']
    ax[0].bar(range(len(names)),[e1[n]/e1['cases'] for n in names],color='#825ea8');ax[0].set(xticks=range(len(names)),xticklabels=['nonmono','compact','neutral Δ','prior','whole'],ylabel='Fraction of E1',title='Exact circuit properties')
    views=stability['views'];v=list(views);xx=np.arange(3)
    ax[1].bar(xx-.18,[views[n]['prediction_stable']/100 for n in v],.36,label='prediction',color='#238b8d')
    ax[1].bar(xx+.18,[views[n]['fixed_primary_clause_valid']/100 for n in v],.36,label='same clause',color='#ca6937')
    ax[1].set(xticks=xx,xticklabels=v,ylim=(0,1),title='100 E1 nuisance cases');ax[1].legend(fontsize=7)
    phy=operator['phase14_physical'];ax[2].bar(['bridge','background'],[phy[k]['compatible']/phy[k]['requests'] for k in ('bridge','background')],color='#d1a034')
    ax[2].set(ylim=(0,.3),ylabel='Compatible / requested',title='Inherited ST physical replacements')
    made.append(_save(fig,'05_circuits_stability_and_operators.png'))
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    ax[0].bar(['REM recall','REM error','high-conf wrong/all'],[rem['REM_recall'],1-rem['REM_recall'],stats['high_confidence_wrong_scored']/rem['scored_epochs']],color=['#238b8d','#ca6937','#825ea8'])
    ax[0].set(ylim=(0,1),title='Historical classifier reliability')
    q=cost['head_queries_attempted_including_successes'];s=stability['new_head_queries_attempted_including_successes']
    ax[1].bar(['base+neutral','nuisance'],[q/1e6,s/1e6],color=['#825ea8','#238b8d']);ax[1].axhline(38,color='#ca6937',ls='--',label='budget');ax[1].set(ylabel='Million cached-head states',title='New explanation cost');ax[1].legend(fontsize=7)
    negative=comparison['exact_sigma_disagreement_E2'];ax[2].bar(['sigma wins','ties','exact wins'],[negative['sigma_higher'],negative['tie'],negative['exact_higher']],color=['#ca6937','#777','#238b8d']);ax[2].set(ylabel='Eligible E2 cases',title='Proxy alignment disagreement')
    made.append(_save(fig,'06_scientific_decision_and_cost.png'))
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    person=rem['participant_REM_distribution']['per_person_recall']
    ax[0,0].bar(['25th','median','75th'],[person['q25'],person['median'],person['q75']],color='#238b8d')
    ax[0,0].axhline(person['mean'],color='#ca6937',ls='--',label='person mean')
    ax[0,0].set(ylim=(0,1),ylabel='REM recall',title='22-person variability (aggregate quantiles)');ax[0,0].legend(fontsize=7)
    paired=stats['paired_condition_REM_recall_difference'];mu=paired['mean_temazepam_minus_placebo'];ci=paired['person_bootstrap_95CI']
    ax[0,1].errorbar([0],[mu],yerr=[[mu-ci[0]],[ci[1]-mu]],fmt='o',color='#825ea8');ax[0,1].axhline(0,color='grey',lw=.8)
    ax[0,1].set(xlim=(-.5,.5),xticks=[0],xticklabels=['temazepam − placebo'],ylabel='REM recall difference',title='Within-person paired nights, n=22')
    q=quality['by_REM_outcome'];xx=np.arange(4)
    ax[0,2].bar(xx-.18,[q[s]['by_channel_flag']['EEG Fpz-Cz|step']['fraction'] for s in ERRORS],.36,color='#ca6937',label='Fpz step')
    ax[0,2].bar(xx+.18,[q[s]['by_channel_flag']['EEG Pz-Oz|extreme']['fraction'] for s in ERRORS],.36,color='#238b8d',label='Pz extreme')
    ax[0,2].set(xticks=xx,xticklabels=['correct','to W','to N1','to N2'],ylabel='Fixed flag fraction',title='Signal-quality candidates');ax[0,2].legend(fontsize=7)
    hist=panels['E1']['positive_support_atom_histogram'];ax[1,0].bar(range(16),[hist[str(k)] for k in range(16)],color='#825ea8')
    ax[1,0].set(xlabel='Required physical 2-s atoms',ylabel='E1 cases',title='Exact-positive support size')
    ax[1,1].scatter([wave[s]['EEG_Fpz_Cz']['sigma_fraction']['median'] for s in ERRORS],
                    [wave[s]['EEG_Pz_Oz']['sigma_fraction']['median'] for s in ERRORS],s=80,c=COLORS)
    for s,label in zip(ERRORS,['correct','to W','to N1','to N2']):
        ax[1,1].annotate(label,(wave[s]['EEG_Fpz_Cz']['sigma_fraction']['median'],wave[s]['EEG_Pz_Oz']['sigma_fraction']['median']),xytext=(4,3),textcoords='offset points',fontsize=7)
    ax[1,1].set(xlabel='Fpz-Cz median sigma fraction',ylabel='Pz-Oz median sigma fraction',title='Cross-lead spectral summaries')
    disagreement=comparison['exact_sigma_disagreement_E2']
    ax[1,2].bar(['correct stage','wrong stage'],[disagreement['correct_exact_minus_sigma_mean'],
                    disagreement['wrong_exact_minus_sigma_mean']],color=['#238b8d','#ca6937'])
    ax[1,2].axhline(0,color='grey',lw=.8);ax[1,2].set(ylabel='Exact − sigma mean precision',title='Proxy disagreement by model correctness')
    made.append(_save(fig,'07_participants_quality_support_and_disagreement.png'))
    new(PUBLIC/'figure_manifest.json',{'status':LABEL,'figures':made,'source':'public aggregate results only',
        'raw_EEG_in_figures':False,'principal_figures':len(made),'human_event_validation':'NOT_TESTED'})
    print(json.dumps({'figures':len(made),'files':made}))


if __name__=='__main__':run()

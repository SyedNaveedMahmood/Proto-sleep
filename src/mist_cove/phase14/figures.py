"""Aggregate-only figures and a separate ignored private waveform atlas."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import PUBLIC,PRIVATE,read_json,write_new,ledger


def run():
    sc=read_json(PUBLIC/'sc_proxy_aggregates.json');st=read_json(PUBLIC/'st_proxy_results.json');sa=read_json(PUBLIC/'sc_alignment_results.json');ta=read_json(PUBLIC/'st_explanation_results.json')
    clas=[read_json(PUBLIC/'sc_classification_metrics.json'),read_json(PUBLIC/'st_classification_metrics.json')];context=read_json(PUBLIC/'context_results.json')
    out=PUBLIC/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=170);plt.close(fig)
    colors=['#377eb8','#e41a1c'];labels=['SC TRAIN','ST frozen']
    fig,ax=plt.subplots(1,2,figsize=(11,4));channels=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental'];contract=read_json(PUBLIC/'channel_contract.json')['audited']
    for j,role in enumerate(('SC_TRAIN','ST_EVAL')):
        n=29 if j==0 else 44;ax[0].bar(np.arange(4)+(j-.5)*.3,[contract[role][k]['recordings']/n for k in channels],.3,label=labels[j],color=colors[j])
    ax[0].set(xticks=range(4),xticklabels=['Fpz-Cz','Pz-Oz','EOG','EMG'],ylabel='Fraction of verified recordings',ylim=(0,1.1),title='73authenticated pairs; native channel availability');ax[0].legend()
    for j,d in enumerate((sc,st)):
        keys=['centered_RMS_uV','EOG_RMS_uV'];m=[d['signal_distributions'][k]['stage_balanced_person_mean']['mean'] for k in keys]
        ax[1].bar(np.arange(2)+(j-.5)*.3,m,.3,color=colors[j],label=labels[j])
    ax[1].set(xticks=range(2),xticklabels=['Fpz-Cz RMS','EOG RMS'],ylabel='µV; equal-stage/person means',title='Signal scale; EMG acquisition not pooled');save(fig,'01_channels_and_signal_scale')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for axis,kind in zip(ax,('spindle','slow_wave')):
        for j,d in enumerate((sc,st)):
            rr=[d['proxies'][channel+'|'+kind]['person_mean_rate'] for channel in ('EEG Fpz-Cz','EEG Pz-Oz')]
            means=np.array([r['mean'] for r in rr]);bounds=np.array([r['bootstrap95'] for r in rr]);axis.errorbar(np.arange(2)+(j-.5)*.12,means,yerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',label=labels[j],color=colors[j])
        axis.set(xticks=range(2),xticklabels=['Fpz-Cz','Pz-Oz'],ylabel='Candidates per eligible hour',title=kind+' proxies; person-bootstrap95%');axis.legend()
    save(fig,'02_spindle_and_slow_wave_rates')
    fig,ax=plt.subplots(1,2,figsize=(11,4));kinds=['spindle','spindle_strict','slow_wave','slow_wave_relaxed']
    for j,d in enumerate((sc,st)):
        ax[0].bar(np.arange(4)+(j-.5)*.3,[d['EEG_cross_channel_concordance'][k]['pooled_F1'] for k in kinds],.3,color=colors[j],label=labels[j])
        methods=['EEG Fpz-Cz|spindle','EEG Fpz-Cz|slow_wave'];m=[d['proxies'][k]['duration_seconds']['mean'] for k in methods]
        ax[1].bar(np.arange(2)+(j-.5)*.3,m,.3,color=colors[j],label=labels[j])
    ax[0].set(xticks=range(4),xticklabels=['SP','SP strict','SW','SW relaxed'],ylim=(0,1),ylabel='Cross-channel event F1',title='Signal-view agreement; not human accuracy');ax[0].legend()
    ax[1].set(xticks=range(2),xticklabels=['Spindle','Slow wave'],ylabel='Mean event duration (seconds)',title='Fpz-Cz proxy morphology');save(fig,'03_cross_channel_and_detector_sensitivity')
    fig,ax=plt.subplots(1,2,figsize=(11,4));keys=['EOG horizontal|excursion','EMG submental|low','EMG submental|high']
    for j,d in enumerate((sc,st)):
        ax[0].bar(np.arange(3)+(j-.5)*.3,[d['proxies'][k]['person_mean_rate']['mean'] for k in keys],.3,color=colors[j],label=labels[j])
        kk=['EEG Fpz-Cz|flat','EEG Fpz-Cz|step','EEG Fpz-Cz|clip'];ax[1].bar(np.arange(3)+(j-.5)*.3,[d['proxies'][k]['person_mean_rate']['mean'] for k in kk],.3,color=colors[j],label=labels[j])
    ax[0].set(xticks=range(3),xticklabels=['EOG excursion','EMG low','EMG high'],ylabel='Proxy intervals per eligible hour',title='EMG: SCenvelope versus STband RMS');ax[0].legend()
    ax[1].set(xticks=range(3),xticklabels=['Flat','Step','Clip'],ylabel='Possible artifacts per hour',title='Quality algorithms are not expert labels');save(fig,'04_EOG_EMG_and_quality')
    fig,ax=plt.subplots(1,2,figsize=(12,4));methods=['exact_positive','random_matched','position_matched','RMS_matched','sigma_matched','gradient_matched']
    for axis,d,title in zip(ax,[sa['strata']['development_primary'],ta['strata']['primary']],['SC: 5 development people','ST: frozen panel']):
        endpoint=d['proxy_concordance'].get('EEG Pz-Oz|spindle',{});mm=endpoint.get('methods',{});values=[mm.get(m+'_precision_like',{}).get('mean',np.nan) for m in methods]
        axis.bar(range(len(methods)),values,color='#377eb8');axis.set(xticks=range(len(methods)),xticklabels=['Exact','Random','Position','RMS','Sigma','Gradient'],ylabel='Selected-time overlap with proxy',title=title+f"; eligible n={endpoint.get('event_eligible_nonempty_cases',0)}")
        bounds=np.array([mm[m+'_precision_like']['bootstrap95'] for m in methods]);values=np.array(values)
        axis.errorbar(range(len(methods)),values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt='none',color='#222222',capsize=3)
    save(fig,'05_proxy_overlap_and_matched_controls')
    fig,ax=plt.subplots(1,2,figsize=(11,4));names=['Compact active','Whole-epoch positive','Prior response','Global-only stable']
    for j,d in enumerate((sa['strata']['primary'],ta['strata']['primary'])):
        keys=['displayed_compact_active','whole_epoch_positive','prior_response','global_stable_local_changed'];ax[0].bar(np.arange(4)+(j-.5)*.3,[d[k]/d['cases'] for k in keys],.3,color=colors[j],label=labels[j])
        namesv=list(d['views']);ax[1].plot(range(3),[d['views'][v]['prediction_stable']/d['views'][v]['cases'] for v in namesv],'-o',color=colors[j],label=labels[j]+' prediction')
        ax[1].plot(range(3),[d['views'][v]['fixed_clause_valid']/max(1,d['views'][v]['fixed_clause_eligible']) for v in namesv],'--x',color=colors[j],label=labels[j]+' original clause')
    ax[0].set(xticks=range(4),xticklabels=['Compact','Whole','Prior','Global'],ylabel='Fraction of audit cases',title='Structural support, not physiology');ax[0].legend()
    ax[1].set(xticks=range(3),xticklabels=['MildHF','Drift','Gain'],ylim=(0,1.05),ylabel='Stability fraction',title='Prediction versus unchanged explanation');ax[1].legend(fontsize=7)
    save(fig,'06_structure_and_explanation_stability')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for axis,d,title in zip(ax,clas,labels):
        cm=np.array(d['metrics']['confusion_matrix'],dtype=float);cm=cm/cm.sum(1,keepdims=True)
        axis.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):axis.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color='white' if cm[i,j]>.55 else 'black')
        axis.set(xticks=range(5),yticks=range(5),xticklabels=['W','N1','N2','N3','R'],yticklabels=['W','N1','N2','N3','R'],xlabel='Predicted',ylabel='Scored',title=title+'; row-normalized')
    save(fig,'07_SC_ST_stage_confusions')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for j,d in enumerate(clas):
        cal=d['metrics']['calibration'];ax[0].bar(np.arange(3)+(j-.5)*.3,[cal[k] for k in ('NLL','Brier','ECE15')],.3,color=colors[j],label=labels[j])
        ax[1].bar(np.arange(5)+(j-.5)*.3,np.array(d['metrics']['support'])/sum(d['metrics']['support']),.3,color=colors[j],label=labels[j])
    ax[0].set(xticks=range(3),xticklabels=['NLL','Brier','ECE15'],title='Frozen historical model; no recalibration');ax[0].legend()
    ax[1].set(xticks=range(5),xticklabels=['W','N1','N2','N3','R'],ylabel='Scored-stage prevalence',title='Different recording length and study selection');save(fig,'08_calibration_and_prevalence')
    fig,ax=plt.subplots(1,2,figsize=(11,4));condition=read_json(PUBLIC/'st_condition_aggregates.json')['metrics'];kk=['accuracy','macro_F1','confidence'];values=np.array([condition[k]['mean'] for k in kk]);bounds=np.array([condition[k]['bootstrap95'] for k in kk])
    ax[0].errorbar(range(3),values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt='o');ax[0].axhline(0,color='gray');ax[0].set(xticks=range(3),xticklabels=['Accuracy','Macro-F1','Confidence'],ylabel='Temazepam minus placebo',title='Withinperson descriptive95% intervals')
    for j,d in enumerate((sa['strata']['primary'],ta['strata']['primary'])):
        ax[1].bar(np.arange(2)+(j-.5)*.3,[d['neutral_changed_tables']/d['neutral_cases'],d['neutral_minimum_changed']/d['neutral_cases']],.3,color=colors[j],label=labels[j])
    ax[1].set(xticks=range(2),xticklabels=['Any truth-table change','Minimum-positive change'],ylabel='Fraction of cases',title='Zero versus .01neutral');ax[1].legend();save(fig,'09_paired_conditions_and_neutral_sensitivity')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for j,arm in enumerate(('C0','C1')):
        rr=[r for r in context['rows'] if r['arm']==arm]
        ax[0].scatter([j]*3,[r['evaluation']['all']['mean_subject_macro_f1'] for r in rr],label=arm,color=colors[j]);ax[0].scatter([j+.15]*3,[r['evaluation']['all']['per_stage_recall'][1] for r in rr],marker='x',color=colors[j])
        ax[1].scatter([r['fit_wall_seconds'] for r in rr],[r['evaluation']['all']['mean_subject_macro_f1'] for r in rr],label=arm,color=colors[j])
    ax[0].set(xticks=[.075,1.075],xticklabels=['C0','C1'],ylabel='F1(dot) / N1recall(cross)',title='All3SCoptimization seeds; matched18,000updates')
    ax[1].set(xlabel='Measured fit CPU seconds',ylabel='Mean participant Macro-F1',title='Equal parameter count; no STselection');ax[1].legend();save(fig,'10_context_and_compute')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for axis,key,title in zip(ax,('BDD_nodes','displayed_clause_state_coverage'),('Exact BDD complexity','Displayed clause response-state coverage')):
        boxes=[]
        for d,label in zip((sa['strata']['primary'],ta['strata']['primary']),labels):
            v=d[key];boxes.append(dict(label=label,med=v['median'],q1=v['q25'],q3=v['q75'],whislo=v['min'],whishi=v['max'],fliers=[]))
        axis.bxp(boxes,showfliers=False);axis.set(title=title,ylabel='Nonterminal nodes' if key=='BDD_nodes' else 'Fraction of positive response states')
    save(fig,'11_graph_complexity_and_clause_coverage')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for j,d in enumerate(clas):ax[0].bar(np.arange(5)+(j-.5)*.3,d['metrics']['per_stage_f1'],.3,label=labels[j],color=colors[j])
    ax[0].set(xticks=range(5),xticklabels=['W','N1','N2','N3','REM'],ylabel='Pooled class F1',ylim=(0,1),title='Complete scored grids; SC is in-sample');ax[0].legend()
    for seed in (123,456,789):
        rr={r['arm']:r for r in context['rows'] if r['seed']==seed}
        ax[1].plot([0,1],[rr[a]['evaluation']['transition']['accuracy'] for a in ('C0','C1')],'-o',label=str(seed))
    ax[1].set(xticks=[0,1],xticklabels=['C0 center','C1 context'],ylabel='SC transition accuracy',title='All 3 seeds: context transition gate FAIL');ax[1].legend(title='Seed')
    save(fig,'12_stage_errors_and_context_transitions')


def private_atlas():
    from matplotlib.backends.backend_pdf import PdfPages
    from .edf_reader import MultiReader
    from .event_schema import event_intervals
    from .common import measured
    out=PRIVATE/'atlas';out.mkdir(exist_ok=True);provenance=[];missing=[];source=MultiReader()
    categories=['EEG Fpz-Cz|spindle','EEG Fpz-Cz|slow_wave','EEG Fpz-Cz|alpha','EEG Fpz-Cz|theta',
                'EEG Fpz-Cz|HF','EEG Fpz-Cz|flat','EEG Fpz-Cz|clip','EEG Fpz-Cz|step',
                'EOG horizontal|excursion','EMG submental|low','EMG submental|high']
    with measured('private_waveform_atlas') as costs:
        with PdfPages(out/'AUTOMATED_PROXY_ATLAS_PRIVATE.pdf') as pdf:
            for role in ('SC_TRAIN','ST_EVAL'):
                coords=read_json(PRIVATE/(role+'_audit_coordinates.json'))['samples'];events={};selected={}
                for r in source.records(role):events[r['recording']]=read_json(PRIVATE/role/(r['recording']+'_events.json'))
                for category in categories:
                    chosen=next((i for i,c in enumerate(coords) if event_intervals(events[c['recording_id']]['events'].get(category,[]),c['epoch'])),None)
                    if chosen is None:missing.append({'role':role,'category':category,'status':'NOT_OBSERVED_IN_FROZEN_AUDIT_CACHE'})
                    else:selected[category]=chosen
                with np.load(PRIVATE/(role+'_audit_inputs.npz'),allow_pickle=False) as archive:
                    for r in source.records(role):
                        cases=[(cat,i) for cat,i in selected.items() if coords[i]['recording_id']==r['recording']]
                        if not cases:continue
                        with source.open(r,values=True) as (reader,header):
                            for category,i in cases:
                                c=coords[i];emg,identity=source.read(r,reader,'EMG submental',c['epoch']*30,(c['epoch']+1)*30)
                                fig,axes=plt.subplots(4,1,figsize=(12,8),sharex=True)
                                for axis,channel in zip(axes[:3],('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal')):
                                    axis.plot(np.arange(3000)/100,archive[channel][i],linewidth=.6);axis.set_ylabel(channel+' µV')
                                axes[3].plot(np.arange(len(emg))/identity['native_fs'],emg,linewidth=.7,marker='.' if identity['native_fs']==1 else None)
                                axes[3].set_ylabel('EMG native µV');axes[3].set_xlabel('Physical epoch seconds; EMG nativeHz='+str(identity['native_fs']))
                                intervals=event_intervals(events[r['recording']]['events'][category],c['epoch'])
                                for axis in axes:
                                    for start,stop in intervals:axis.axvspan(start/100,stop/100,color='orange',alpha=.15)
                                fig.suptitle('PRIVATE AUTOMATED PROXY: '+category+'; not expert truth');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
                                provenance.append({'role':role,'coordinate':c,'category':category,'EMG_source':identity,'proxy_source_sha256':events[r['recording']]['source_sha256'],
                                    'selection':'first observed category within already frozen audit cache; visualization-only, not representative inference'})
        write_new(out/'provenance.json',{'examples':provenance,'missing_categories':missing})
        receipt={'private_figures':len(provenance),'missing_category_strata':len(missing),'raw_waveform_publication':False,'source_access':source.access.counts,
                 'SC_EMG':'plotted native1Hz envelope without upsampling','ST_EMG':'plotted native100Hz raw signal','event_labels':'AUTOMATED_PROXIES_ONLY'}
        write_new(PUBLIC/'private_atlas_receipt.json',receipt);costs.update(**source.access.counts);source.access.close()

if __name__=='__main__':run();private_atlas()

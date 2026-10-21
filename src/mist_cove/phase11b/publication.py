"""Reproducible aggregate summaries and figures. No raw waveform publication."""
import json
import time
from pathlib import Path
import numpy as np
from .campaign import PUBLIC,PRIVATE,write_new
from .historical import ROOT,canonical
from .security import sha256
from .evaluation import distribution,clustered_mean,QUALITY_NAMES
from mist_transfer.metrics import classification_metrics

def read(name):return json.loads((PUBLIC/name).read_text())

def derived_summaries():
    started=time.perf_counter();rows=[json.loads(v) for v in (PRIVATE/'audit_rows.jsonl').read_text().splitlines()];primary=[r for r in rows if r['primary']]
    data=np.load(PRIVATE/'train_predictions.npz',allow_pickle=False);y=data['truth'];z=data['logits'];subjects=data['subjects'];records=data['records'];q=data['quality'];correct=z.argmax(1)==y
    transition=np.zeros(len(y),bool);transition[1:]=(records[1:]==records[:-1])&(y[1:]>=0)&(y[:-1]>=0)&(y[1:]!=y[:-1])
    per_stage=[]
    for stage in range(5):
        m=y==stage;qualities={}
        for j,name in enumerate(QUALITY_NAMES):
            differences=[];units=[]
            for s in sorted(set(subjects)):
                a=m&(subjects==s)&correct;b=m&(subjects==s)&~correct
                if a.any() and b.any():differences.append(q[b,j].mean()-q[a,j].mean());units.append(s)
            qualities[name]=clustered_mean(differences,units)
        selected=[r for r in primary if r['coordinate']['stage']==stage]
        per_stage.append({'stage':stage,'full_train_scored':int(m.sum()),'error_count':int((m&~correct).sum()),
            'preceded_by_stage_transition':{'n':int((m&transition).sum()),'errors':int((m&transition&~correct).sum())},
            'not_preceded_by_scored_transition':{'n':int((m&~transition).sum()),'errors':int((m&~transition&~correct).sum())},
            'within_stage_wrong_minus_correct_quality':qualities,
            'primary_global_only_stable':sum(r['native']['prediction']==r['feature_ablations']['global_only']['prediction'] for r in selected),
            'primary_high_confidence_wrong':sum(not r['correct'] and r['native']['confidence']>=.9 for r in selected),
            'primary_nontrivial_positive_minimum_compact':sum(0<r['summary']['positive_minimum_mask'].bit_count()<=4 for r in selected)})
    detail={'stage_taxonomy':per_stage,'primary_exact_compact_positive_count':sum(0<r['summary']['positive_minimum_mask'].bit_count()<=4 for r in primary),
            'primary_nonmonotone_examples':sum(r['summary']['nonmonotone_edges']['response_lost_when_adding_atom']>0 for r in primary),
            'primary_nonmonotone_edge_fraction':distribution([r['summary']['nonmonotone_edges']['fraction'] for r in primary]),
            'primary_multiple_displayed_clauses':sum(len(r['summary']['signed']['clauses'])>1 for r in primary),
            'stage_transition_is_ambiguity_truth':False,'quality_associations_are_causal':False}
    write_new(PUBLIC/'mechanistic_details.json',detail)
    historical=json.loads((ROOT/'reports/p3b/results.json').read_text())
    old=next(v['metrics'] for v in historical['per_seed'] if v['variant']=='raw_context' and v['seed']==123);new=read('val_descriptive_metrics.json')['metrics']
    write_new(PUBLIC/'historical_reproduction.json',{'selected_seed':123,'published_mean_subject_macro_f1':old['mean_subject_macro_f1'],
        'current_mean_subject_macro_f1':new['mean_subject_macro_f1'],'exact_mean_subject_score_match':old['mean_subject_macro_f1']==new['mean_subject_macro_f1'],
        'confusion_matrix_match':old['confusion_matrix']==new['confusion_matrix'],'interpretation':'same reused development cohort; numerical reproduction, not external validity'})
    new_data=np.load(PRIVATE/'new_model_data.npz',allow_pickle=False);n=int(new_data['n_fit']);ny=new_data['y'][n:];ns=new_data['subjects'][n:]
    paired={}
    for view in ('clean','local_hf','strong_hf','mild_hf'):
        deltas=[]
        for seed in (123,456,789):
            arm=[]
            for name in ('M1','M2'):
                logits=np.load(PRIVATE/(name+'_'+str(seed)+'_evaluation.npz'),allow_pickle=False)[view+'_logits'];metric=classification_metrics(ny,logits.argmax(1),ns)
                arm.append(metric['subject_macro_f1'])
            deltas.append([arm[1][s]-arm[0][s] for s in sorted(arm[0])])
        paired[view]={'mean_seed_difference_by_participant':clustered_mean(np.mean(deltas,axis=0),np.arange(5)),
                      'seeds_kept_within_participant':True,'independent_subjects':5}
    write_new(PUBLIC/'model_paired_uncertainty.json',paired)
    write_new(PUBLIC/'aggregate_analysis_resource.json',{'CPU_wall_seconds':time.perf_counter()-started,'neural_forwards':0,'EEG_file_reads':0})

def resources():
    stages={};entries=[('model_authentication.json','wall_seconds'),('trained_adapter_correctness.json','wall_seconds'),
        ('sample_selection.json','wall_seconds'),('source_pilot.json','wall_seconds'),('train_descriptive_metrics.json','wall_seconds'),
        ('val_descriptive_metrics.json','wall_seconds'),('local_hf_followup.json','total_wall_seconds_including_calibration'),
        ('new_model_data_receipt.json','wall_seconds'),('model_comparison.json','wall_seconds'),('new_model_operator_audit.json','wall_seconds'),
        ('information_loss_metrics.json','wall_seconds'),('aggregate_analysis_resource.json','CPU_wall_seconds')]
    for name,key in entries:stages[name]=read(name)[key]
    audit=read('audit_execution.json');stages['audit_execution.json']=audit['resources']['wall_seconds']
    # Generated trained-adapter checks used 338 extra head evaluations and 86
    # extra cache examples beyond the provisional runner initialization.
    cached=audit['resources']['cached_head_queries_including_prior_checks']+338+read('new_model_operator_audit.json')['cached_additive_head_queries']+300
    native=audit['resources']['native_full_example_forwards']+79984+21796+150+read('model_comparison.json')['full_example_forwards']+read('new_model_operator_audit.json')['native_full_example_forwards']+300
    caches=audit['resources']['feature_cache_example_forwards']+86+150
    source={k:0 for k in ('descriptor_files','waveform_file_reads','annotation_hash_reads','waveform_bytes','physical_epochs_decoded','source_io_seconds')}
    for name in ('sample_selection.json','source_pilot.json','audit_execution.json','train_descriptive_metrics.json','val_descriptive_metrics.json','new_model_data_receipt.json'):
        for k,v in read(name)['source_access'].items():source[k]+=v
    receipt={'stage_CPU_wall_seconds':stages,'total_scientific_CPU_wall_seconds':sum(stages.values()),'scientific_CPU_ceiling_seconds':7200,
        'cached_head_queries':cached,'cached_head_query_ceiling':50000000,'native_full_example_forwards':native,'feature_cache_example_forwards':caches,
        'combined_full_and_cache_examples':native+caches,'full_example_ceiling':250000,'core_new_neural_fits':9,'scientific_optimizer_updates':2700,
        'clean_training_exposures':43200,'paired_nuisance_training_exposures':43200,'GPU_seconds':0,'source_access':source,
        'distinct_approved_recordings':37,'distinct_physical_source_epochs':101780,
        'peak_measured_process_RSS_KiB':max(audit['resources']['peak_RSS_KiB'],read('train_descriptive_metrics.json')['peak_RSS_KiB'],read('val_descriptive_metrics.json')['peak_RSS_KiB']),
        'table_queries_not_training_exposures':True,'accounting_correction':'Add338 generated adapter head queries and86 generated cache examples omitted from initial audit runner offset; no experiment rerun',
        'generated_new_model_engineering_pilot':read('new_model_generated_pilot.json'),
        'CUDA_status':'NOT_RUN: occupancy probe19percent utilization; no allocation, no interrupted process; patched research environment CPU only',
        'excluded_from_scientific_wall':['dependency installation','test suites','literature/source review','report rendering','interactive engineering time']}
    if not (receipt['total_scientific_CPU_wall_seconds']<=7200 and cached<=50000000 and native+caches<=250000):raise ValueError('aggregate resource ceiling exceeded')
    write_new(PUBLIC/'resource_receipts.json',receipt)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    stages=['Wake','N1','N2','N3','REM'];train=read('train_descriptive_metrics.json')['metrics'];val=read('val_descriptive_metrics.json')['metrics'];cir=read('circuit_metrics.json');triad=read('triad_metrics.json')
    directory=PUBLIC/'figures';directory.mkdir(exist_ok=True)
    def finish(fig,name,title):
        fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(directory/name);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for a,m,title in [(ax[0,0],train,'TRAIN: in-sample'),(ax[0,1],val,'Reused development-VAL')]:
        cm=np.array(m['confusion_matrix']);a.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues');a.set(xticks=range(5),xticklabels=stages,yticks=range(5),yticklabels=stages,title=title,xlabel='Predicted',ylabel='Scored stage')
    x=np.arange(5);ax[0,2].bar(x-.15,train['per_stage_f1'],.3,label='TRAIN');ax[0,2].bar(x+.15,val['per_stage_f1'],.3,label='Dev-VAL');ax[0,2].set(xticks=x,xticklabels=stages,ylim=(0,1),title='Stage F1');ax[0,2].legend()
    for a,key,title in [(ax[1,0],'subject_macro_f1_distribution','Participant variability'),(ax[1,1],'recording_macro_f1_distribution','Recording variability')]:
        stats=[]
        for label,m in [('TRAIN',train),('Dev-VAL',val)]:
            d=m[key];stats.append({'label':label,'med':d['median'],'q1':d['q25'],'q3':d['q75'],'whislo':d['min'],'whishi':d['max'],'fliers':[]})
        a.bxp(stats);a.set(ylim=(0,1),title=title,ylabel='Fixed-five Macro-F1')
    ax[1,2].bar(['TRAIN errors','Dev-VAL errors'],[train['calibration']['high_confidence_wrong_fraction_errors'],val['calibration']['high_confidence_wrong_fraction_errors']]);ax[1,2].set(ylim=(0,1),title='Fraction of errors with confidence ≥.9')
    finish(fig,'01_classification.png','Original trained model: classification, imbalance and variability')
    primary=cir['primary'];extended=cir['extended_inclusive'];fig,ax=plt.subplots(2,3,figsize=(14,8));rows=[json.loads(v) for v in (PRIVATE/'audit_rows.jsonl').read_text().splitlines()];p=[r for r in rows if r['primary']]
    ax[0,0].hist([r['summary']['BDD_nonterminal_nodes'] for r in p],bins=15);ax[0,0].set(title='Exact circuit complexity',xlabel='ROBDD nonterminal nodes',ylabel='Primary epochs')
    ax[0,1].hist([2*r['summary']['positive_minimum_mask'].bit_count() for r in p],bins=np.arange(1,33,2));ax[0,1].set(title='Exact positive support minimum',xlabel='Retained seconds')
    groups=[primary['correct'],primary['incorrect']];ax[0,2].bar(['Correct (99)','Incorrect (51)'],[g['compact_active']/g['n'] for g in groups]);ax[0,2].set(ylim=(0,1),title='Compact active summary coverage')
    ax[1,0].bar(['Local-only stable','Global-only stable','Whole-epoch minimum','Empty target response'],[primary['rates'][k]['count']/150 for k in ('local_only_prediction_stable','global_only_prediction_stable','whole_epoch_positive_minimum','canonical_empty_target_response')]);ax[1,0].tick_params(axis='x',rotation=25);ax[1,0].set(ylim=(0,1),title='Different internal dependence questions')
    ax[1,1].bar(['Signed four-clause','Positive four-clause','Exact ROBDD'],[primary['signed_true_mask_coverage']['mean'],primary['positive_only_true_mask_coverage']['mean'],1]);ax[1,1].tick_params(axis='x',rotation=15);ax[1,1].set(ylim=(0,1),title='Mean true-mask coverage')
    ax[1,2].bar(['Primary150','Extended750'],[primary['classification']['accuracy'],extended['classification']['accuracy']]);ax[1,2].set(ylim=(0,1),title='Early-epoch selection sensitivity (nested)')
    finish(fig,'02_circuits.png','Exactness, compactness and correctness are separate properties')
    fig,ax=plt.subplots(2,3,figsize=(14,8));views=list(triad['stability']);x=np.arange(3)
    for offset,key,label in [(-.25,'prediction_stable','Prediction'),(0,'fixed_clause_valid','Same active clause'),(.25,'any_original_active_clause_valid','Any original active clause')]:ax[0,0].bar(x+offset,[triad['stability'][v][key]['count']/150 for v in views],.25,label=label)
    ax[0,0].set(xticks=x,xticklabels=views,ylim=(0,1),title='View stability');ax[0,0].legend(fontsize=7)
    names=list(triad['probes']);ax[0,1].bar(names,[triad['probes'][v]['metrics']['accuracy'] for v in names]);ax[0,1].tick_params(axis='x',rotation=50);ax[0,1].set(ylim=(0,1),title='Waveform sensitivity; label preservation unknown')
    ax[0,2].bar(['Bridge compatible','Background eligible'],[33/2250,0]);ax[0,2].set(ylim=(0,1),title='Intervention coverage (2,250 requests each)')
    ax[1,0].hist([r['triad']['neutral_table_disagreement_fraction'] for r in p],bins=15);ax[1,0].set(title='Zero versus .01 neutral',xlabel='Fraction of mask responses changed')
    ax[1,1].bar(['Compatible comparisons','Disagreements'],[triad['compatible_operator_comparisons'],triad['operator_disagreements']]);ax[1,1].set(title='Internal versus bridge effects')
    info=read('information_loss_metrics.json');names=list(info['states']);ax[1,2].bar(names,[info['states'][v]['confidence']['mean'] for v in names]);ax[1,2].tick_params(axis='x',rotation=35);ax[1,2].set(ylim=(0,1),title='Confidence under internal information loss')
    finish(fig,'03_triad.png','TRIAD preserves operator disagreement and unavailable interventions')
    fig,ax=plt.subplots(2,3,figsize=(14,8));arms=['M0','M1','M2'];models=[read(a+'_'+str(s)+'_metrics.json') for a in arms for s in (123,456,789)]
    for a,view,title in [(ax[0,0],'clean','Clean internal-dev F1'),(ax[0,1],'local_hf','Local high-frequency F1'),(ax[0,2],'strong_hf','Global high-frequency F1')]:
        for i,arm in enumerate(arms):a.scatter([i]*3,[r['evaluation'][view]['metrics']['macro_f1'] for r in models if r['arm']==arm])
        a.set(xticks=range(3),xticklabels=arms,ylim=(0,1),title=title)
    comparison=read('model_comparison.json');x=np.arange(3);ax[1,0].bar(x-.15,[r['M2_minus_M1']['clean'] for r in comparison['paired']],.3,label='Clean');ax[1,0].bar(x+.15,[r['M2_minus_M1']['local_hf'] for r in comparison['paired']],.3,label='Local HF');ax[1,0].axhline(.02,color='gray',ls='--');ax[1,0].set(xticks=x,xticklabels=['123','456','789'],title='M2−M1: all three primary gates failed');ax[1,0].legend()
    for offset,key,label in [(-.15,'mean_absolute_local_logit_contribution','Local'),(.15,'mean_absolute_global_logit','Global')]:ax[1,1].bar(np.arange(3)+offset,[np.mean([r['evaluation']['clean'][key] for r in models if r['arm']==arm]) for arm in arms],.3,label=label)
    ax[1,1].set(xticks=range(3),xticklabels=arms,title='Reported logit magnitudes; not causal effects');ax[1,1].legend()
    costs=read('resource_receipts.json');ax[1,2].bar(['Exact audit','Full TRAIN','Full Dev-VAL','Nine fits + eval'],[costs['stage_CPU_wall_seconds'][k] for k in ('audit_execution.json','train_descriptive_metrics.json','val_descriptive_metrics.json','model_comparison.json')]);ax[1,2].tick_params(axis='x',rotation=20);ax[1,2].set(title='Scientific CPU wall cost',ylabel='Seconds')
    finish(fig,'04_models_compute.png','Nine fixed fits: small-capacity development evidence, not a promoted replacement')

if __name__=='__main__':
    derived_summaries();resources();figures()

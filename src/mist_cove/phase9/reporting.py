"""Aggregate-only offline publication; no model states or per-mask tables."""
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np
from .experiments import ROOT,PUBLIC,PRIVATE,config,update,frozen,plan_proof
from mist_cove.phase2.provenance import file_sha
from mist_cove.phase5.runtime import wilson

FORBIDDEN={'weights','state_dict','model','optimizer','rng','batch_rng','logits','probabilities','margins','waveforms','events','labels','x','y','nodes'}


def privacy_check(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN:raise ValueError('private state or full response table forbidden in publication')
        for v in value.values():privacy_check(v)
    elif isinstance(value,(list,tuple)):
        for v in value:privacy_check(v)


def publish_json(path,value):
    privacy_check(value);payload=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n';publish_bytes(path,payload)


def publish_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.wic-')
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:os.unlink(tmp)


def verified_receipts():
    result={}
    for wp in range(7,14):
        p=PRIVATE/'core_v1'/f'wp{wp:02}_receipt.json';r=json.loads(p.read_text())
        if r['status']!='COMPLETE' or r['real_data_accessed'] is not False:raise ValueError('complete synthetic receipt required')
        result[wp]=r
    for wp in (8,9):
        for row in result[wp]['rows']:
            p=PRIVATE/'core_v1'/f"{row['task']}_{row['seed']}_{row['method']}.pt";v=row['creation']
            if file_sha(p)!=v['checkpoint_sha256'] or file_sha(p.with_suffix('.seal.json'))!=v['seal_sha256'] or json.loads(p.with_suffix('.creation.json').read_text())!=v:
                raise ValueError('prospective checkpoint integrity mismatch')
    for group in result[7]['groups']+result[8]['circuit_groups']:
        if file_sha(ROOT/group['private_circuits_file'])!=group['private_circuits_sha256']:raise ValueError('exact circuit archive mismatch')
    frozen();plan_proof();return result


def circuit_statistics(groups):
    stats={}
    for name in ('P0','E0','E1'):
        rows=[r for g in groups if g['predictor']==name for r in g['rows']];correct=[r for r in rows if r['correct']];wrong=[r for r in rows if not r['correct']]
        def compact(v):return sum(r['compact_active_clean_found'] for r in v)
        nodes=[v['nonterminal_nodes'] for r in rows for v in r['circuits']]
        stats[name]={'examples':len(rows),'correct':len(correct),'wrong':len(wrong),'compact_correct':compact(correct),
            'compact_correct_fraction':compact(correct)/len(correct) if correct else None,
            'compact_correct_wilson95_descriptive':wilson(compact(correct),len(correct),1.96) if correct else None,
            'compact_all_case_fraction':compact(rows)/len(rows) if rows else None,'compact_wrong_fraction':compact(wrong)/len(wrong) if wrong else None,
            'compact_gate':'PASS' if correct and compact(correct)/len(correct)>=.8 else 'FAIL',
            'exact_BDDs':len(nodes),'mean_BDD_nonterminals':float(np.mean(nodes)) if nodes else None,'max_BDD_nonterminals':max(nodes,default=0),
            'mean_signed_summary_coverage':float(np.mean([r['signed_summary']['coverage'] for r in rows])) if rows else None,
            'mean_positive_summary_coverage':float(np.mean([r['positive_summary']['coverage'] for r in rows])) if rows else None,
            'native_three_view_universal_fraction':float(np.mean([r['native']['universal_valid'] for r in rows])) if rows else None,
            'greedy_clean_universal_fraction':float(np.mean([r['greedy'].get('universal_positive_valid',False) for r in rows])) if rows else None,
            'mean_neutral_truth_disagreement':float(np.mean([r['neutral_response_disagreement'] for r in rows])) if rows else None,
            'clean_prior_only':sum(r['clean_summary']['prior_only'] for r in rows),'signed_no_support':sum(r['signed_summary']['status']=='NO_SUPPORT' for r in rows),
            'zero_mask_target_response':sum(r['zero_mask_response'] for r in rows),
            'nonmonotone_examples':sum(r['nonmonotone_edges']['response_lost_when_adding_atom']>0 for r in rows),
            'negative_clause_count':sum(bool(v['negative']) for r in rows for v in r['signed_summary']['clauses']),
            'signed_clause_count':sum(len(r['signed_summary']['clauses']) for r in rows),
            'max_signed_literals':max((v['signed_literals'] for r in rows for v in r['signed_summary']['clauses']),default=0),
            'retained_seconds_mean':float(np.mean([v['retained_seconds'] for r in rows for v in r['signed_summary']['clauses']])) if any(r['signed_summary']['clauses'] for r in rows) else None}
    return stats


def render_figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .experiments import audit_draw
    from .circuits import compile_robdd
    c=config();out=PUBLIC/'figures';out.mkdir(exist_ok=False);files={};plt.rcParams.update({'font.size':9,'figure.dpi':120,'savefig.dpi':150})
    def save(name,fig):
        fig.tight_layout();p=out/name;fig.savefig(p,metadata={'Software':'MIST-WIC aggregate renderer v1'});plt.close(fig);files[name]=file_sha(p)
    group=result[7]['groups'][0];row=group['rows'][0];draw,_=audit_draw(group['task'],group['seed'],c)
    clauses=row['clean_summary']['clauses'];cl=next((v for v in clauses if v['active_at_original_full_mask']),None)
    fig,ax=plt.subplots(figsize=(12,3.8));ax.plot(np.arange(3000)/100,draw.x[0,0].numpy(),lw=.7,color='black')
    for j in range(15):
        if cl and j in cl['positive']:ax.axvspan(j*2,j*2+2,color='green',alpha=.25)
        if cl and j in cl['negative']:ax.axvspan(j*2,j*2+2,color='red',alpha=.25)
        ax.axvline(j*2,color='gray',lw=.4)
    for j in range(14):ax.plot([j*2,j*2+4],[-105-j%2*8]*2,color='blue',lw=1.2,alpha=.6)
    ax.set(xlabel='Physical seconds (15 disjoint atoms)',ylabel='Generated uV',title=f"Frozen first audit example {group['task']}/{group['seed']}, generation index {row['generation_index']}; target code {row['target_class']}\nGreen: retained clause atoms; red: withheld; blue: overlapping native windows; synthetic only")
    save('01_availability_clause.png',fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5));ids=np.arange(32768);a=(ids&1)>0;b=(ids&2)>0;cc=(ids&4)>0
    for ax,(label,truth) in zip(axes,[('A OR B',a|b),('A AND B',a&b),('(A OR B) AND NOT C',(a|b)&~cc)]):
        graph=compile_robdd(truth);levels={}
        for i,(var,lo,hi) in enumerate(graph['nodes'][2:],2):levels.setdefault(var,[]).append(i)
        pos={0:(0.,-1),1:(1.,-1)}
        for var,nodes in levels.items():
            for j,i in enumerate(nodes):pos[i]=((j+1)/(len(nodes)+1),3-var)
        for i,(var,lo,hi) in enumerate(graph['nodes'][2:],2):
            for child,bit,color in ((lo,'0','gray'),(hi,'1','green')):
                xx,yy=pos[i];cx,cy=pos[child];ax.plot([xx,cx],[yy,cy],color=color,ls='--' if bit=='0' else '-');ax.text((xx+cx)/2,(yy+cy)/2,bit,color=color,fontsize=8)
        for i,(xx,yy) in pos.items():
            labelnode=str(i) if i<2 else 'ABC'[graph['nodes'][i][0]]
            ax.text(xx,yy,labelnode,ha='center',va='center',bbox={'boxstyle':'round','facecolor':'white','edgecolor':'black'})
        ax.set(title=label,xlim=(-.1,1.1),ylim=(-1.5,3.5));ax.axis('off')
    fig.suptitle('Exact generated toy ROBDDs; established Boolean infrastructure, not clinical circuits');save('02_toy_circuits.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));arms=['C0','C1','C2'];xx=np.arange(3)
    for ax,task in zip(axes,c['tasks']):
        for j,condition in enumerate(('clean','competitive')):
            values=np.array([[r['evaluation']['conditions'][condition]['raw']['macro_f1'] for r in result[9]['rows'] if r['task']==task and r['method']==arm] for arm in arms])
            ax.bar(xx+(j-.5)*.35,values.mean(1),.35,yerr=values.std(1,ddof=1),capsize=3,label=condition)
        ax.set(xticks=xx,xticklabels=arms,title=task,ylabel='Generated fixed-five Macro-F1',ylim=(0,1.06));ax.legend()
    save('03_clean_competitive.png',fig)
    fig,ax=plt.subplots(figsize=(9,4));names=['P0','E0','E1'];xx=np.arange(3)
    for j,(key,label) in enumerate([('prediction_stability','full prediction stable'),('fixed_all_case','fixed active clause, all cases'),('substitution_all_case','substitute active route, all cases')]):
        ax.bar(xx+(j-1)*.23,[result[10]['decisions'][m][key] for m in names],.23,label=label)
    ax.plot(xx,[result[10]['decisions'][m]['fixed_active_conditional'] for m in names],'ko',label='fixed clause conditional on stable prediction')
    ax.axhline(.8,color='gray',ls='--');ax.set(xticks=xx,xticklabels=names,ylabel='Fraction',ylim=(0,1.05),title='Independent finite views; bounded unchanged clause family');ax.legend(fontsize=8);save('04_stability.png',fig)
    groups=result[7]['groups']+result[8]['circuit_groups'];stats=circuit_statistics(groups);fig,axes=plt.subplots(1,2,figsize=(11,4));xx=np.arange(3)
    axes[0].bar(xx-.17,[stats[m]['mean_signed_summary_coverage'] for m in names],.34,label='signed, three-view truth')
    axes[0].bar(xx+.17,[stats[m]['mean_positive_summary_coverage'] for m in names],.34,label='positive-only, same truth')
    axes[0].set(xticks=xx,xticklabels=names,ylabel='Mean bounded-summary coverage',ylim=(0,1.05));axes[0].legend(fontsize=8)
    rows=[r for g in groups for r in g['rows']];durations=[]
    for key in ('native','greedy','WIC active clean'):
        if key=='WIC active clean':values=[min((v['retained_seconds'] for v in r['clean_summary']['clauses'] if v['active_at_original_full_mask']),default=np.nan) for r in rows]
        else:values=[r[key].get('retained_seconds',np.nan) for r in rows]
        durations.append([v for v in values if np.isfinite(v)])
    axes[1].boxplot(durations,labels=['native union','greedy canonical','WIC active clean']);axes[1].set(ylabel='Retained physical seconds',title='Different guarantees; duration alone is not quality');save('05_compactness_coverage.png',fig)
    fig,ax=plt.subplots(figsize=(10,4.5));ops=['identity','exact_diagnostic_removal','exact_nuisance_removal','historical_bridge','historical_independent_baseline']
    for i,name in enumerate(ops):
        vs=[v for r in result[11]['rows'] for v in r['records'] if v['operator']==name]
        x=np.mean([v['target_removal_fraction'] for v in vs]);y=np.mean([v['nontarget_max_residual'] for v in vs]);accept=np.mean([v['physical_compatibility']['accepted'] for v in vs])
        ax.scatter(x,y,s=110,c=[accept],vmin=0,vmax=1,cmap='RdYlGn',edgecolors='black',marker='D' if name=='exact_nuisance_removal' else 'o')
        offsets=[(-70,20),(-220,20),(-220,60),(-250,-25),(-260,15)]
        ax.annotate(name.replace('_',' ')+f' ({sum(v["physical_compatibility"]["accepted"] for v in vs)}/{len(vs)} compatible)',(x,y),xytext=offsets[i],textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'gray'})
    ax.set(xlabel='Fraction of declared target removed (identity targets diagnostic)',ylabel='Mean maximum nontarget change (uV)',yscale='symlog',ylim=(-.1,80),xlim=(-.1,1.1),title='Waveform operator contracts: erasure, preservation and compatibility\nInternal availability is a feature intervention and is excluded from this waveform plane')
    save('06_operator_contract.png',fig)
    return files


def build_results(receipts):
    c=config();groups=receipts[7]['groups']+receipts[8]['circuit_groups'];stats=circuit_statistics(groups)
    failed=json.loads((PRIVATE/'core_v1/STOPPED_WP08.json').read_text());scientific_seconds=sum(receipts[i]['wall_seconds'] for i in range(7,13))+failed['wall_seconds']
    if scientific_seconds>c['scientific_wall_seconds']:raise ValueError('scientific ceiling exceeded')
    fits=receipts[8]['new_fits']+receipts[9]['new_fits'];updates=receipts[8]['optimizer_updates']+receipts[9]['optimizer_updates']
    if fits>42 or updates>9000:raise ValueError('fit/update contract exceeded')
    delta=receipts[9]['gate']['checks'];seed_means=[np.mean([v['competitive_delta_C2_minus_C1'] for v in delta if v['seed']==s]) for s in c['seeds']]
    rng=np.random.default_rng(91233);boot=np.array(seed_means)[rng.integers(0,3,size=(500,3))].mean(1)
    resources={'neural_fits':fits,'optimizer_updates':updates,'preserved_E0_fits_counted_once':1,'reference_reproduction_fits':0,
        'cached_head_training_examples':sum(r['training']['cached_feature_examples'] for r in receipts[8]['rows']),
        'head_fit_full_network_training_examples':0,'paired_training_base_examples':sum(r['training']['base_examples'] for r in receipts[9]['rows']),
        'paired_training_full_view_forwards':sum(r['training']['view_forwards'] for r in receipts[9]['rows']),
        'paired_training_retained_query_head_forwards':sum(r['training']['head_query_forwards'] for r in receipts[9]['rows']),
        'completed_scientific_availability_queries':sum(g['head_mask_queries'] for g in groups)+receipts[10]['head_mask_queries']+receipts[12]['additional_sanity_head_mask_queries'],
        'reporting_recovery_duplicate_availability_queries':40*32768,
        'scientific_wall_seconds_including_failed_attempt':scientific_seconds,'ceiling_seconds':3600,
        'per_track_wall_seconds':{str(i):receipts[i]['wall_seconds'] for i in range(7,13)},'failed_reporting_seconds':failed['wall_seconds'],
        'peak_scientific_process_rss_KiB':max(receipts[i]['peak_rss_KiB'] for i in range(7,13)),
        'measured_feature_cache_bytes_per_example_view':1064,'Boolean_mask_grid_bytes':32768*15,
        'scalar_calibration_fits':6+len(receipts[8]['rows'])+18,
        'timing_scope':'scientific timers include generation, fits, evaluation, circuit analysis and private IO; exclude imports, CI, setup, reporting and WP13 audit; no kernel/core-time claim',
        'query_scope':'availability queries count masked-head examples; native ranking and full-view head calls are not mislabeled as extra full-network exposures'}
    result={'version':'mist-wic-phase9-results-v1','status':'COMPLETE_BOUNDED_SCIENTIFIC_STUDY','plan':plan_proof(),'config':c,
        'circuit_statistics':stats,'track_A1':{'groups':groups[:len(receipts[7]['groups'])]},
        'track_A2':{k:receipts[8][k] for k in ('rows','circuit_groups','eligibility','gate')},
        'track_B':{k:receipts[9][k] for k in ('rows','gate')},'stability':{k:receipts[10][k] for k in ('rows','decisions')},
        'operators':{k:receipts[11][k] for k in ('rows','oracle_correctness','donor_attempts','accepted_donors','historical_guard')},
        'reliability':{k:receipts[12][k] for k in ('rows','sanity_controls')},'correctness':receipts[13],
        'incremental_C2_minus_C1_seed_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
        'uncertainty_scope':'three optimization-seed clusters preserve both tasks; generated-draw paired intervals in evaluation/pairs; no participant or clinical-population significance',
        'resources':resources,'private_receipt_sha256':{f'WP{i:02}':file_sha(PRIVATE/'core_v1'/f'wp{i:02}_receipt.json') for i in range(7,14)},
        'external_implementations':'NOT RUN','clinical_validity':'NOT TESTED','historical_phase6_deserializations':0,'real_data_accessed':False}
    return result


def report_markdown(result):
    c=result['config'];stats=result['circuit_statistics'];res=result['resources'];b=result['track_B'];heads=result['track_A2'];cpu=result['correctness']['cpu'];cuda=result['correctness']['cuda']
    lines=['# MIST-WIC Phase IX — complete bounded research report','', '**Decision:** exact finite response circuits and compact clean summaries are supported within the declared synthetic operator. Clean additive utility, independent-view stability and incremental C2-over-C1 robustness all fail their complete preregistered criteria. These dimensions are reported separately.','', '## Identity, safety and execution','', f"Authoritative plan commit `{result['plan']['plan_commit']}`, blob `{result['plan']['plan_blob']}`. The existing Phase IX branch was fetched and fast-forwarded from the clean Phase VIII checkout; initial HEAD/blob matched exactly. The plan is unchanged. Scientific freeze: `702e90802e1ebbaa1b39940ad6de25ebd20dacd1`; reporting-only continuation: `31c4a19`. Six authenticated P0 references were restored; no reproduction was needed and no unauthenticated Phase VI state was deserialized. Historical III–VIII artifacts remain unchanged.",'', 'The initial A2 attempt fitted/sealed E0 single/123 and completed its circuits before an undefined constant-confidence correlation caused strict JSON to stop. The corrected exporter uses null for undefined correlation. The sealed fit was restored with zero additional optimizer updates; preserved margins and circuit hashes match exactly. Failed artifacts remain local and their 7.251684 seconds and duplicate queries are counted. No scientific model, seed, loss weight, generator, threshold or endpoint changed.','', 'No real TRAIN/VAL EEG, data-bearing annotations/descriptors, real fitted statistics/banks, RESERVED, ST, SHHS, legacy NPZ, external cohort or adaptation was accessed. All data here are generated. Finite certificates are not physiological causes, clinical sleep events or universal adversarial guarantees.','', '## Track A1: exact finite circuits and readable summaries','', '| Predictor | Audit cases correct/wrong | Compact correct cases | Compact all cases | Mean/max BDD nodes | Signed/positive summary coverage | Prior-only / no-support joint |','|---|---:|---:|---:|---:|---:|---:|']
    for m,v in stats.items():lines.append(f"| {m} | {v['correct']}/{v['wrong']} | {v['compact_correct']}/{v['correct']} ({v['compact_correct_fraction']:.3f}) | {v['compact_all_case_fraction']:.3f} | {v['mean_BDD_nonterminals']:.1f}/{v['max_BDD_nonterminals']} | {v['mean_signed_summary_coverage']:.3f}/{v['mean_positive_summary_coverage']:.3f} | {v['clean_prior_only']}/{v['signed_no_support']} |")
    lines += ['', 'Every discovery table enumerates all 32,768 masks; all 540 completed discovery ROBDDs match every row. OR, AND, XOR, blocker, mixed and constant fixtures pass. Input availability uses 15 disjoint 200-sample atoms and retains a 400-sample candidate only when both atoms survive. Full native paths are unchanged at all-visible masks. Hidden feature/RMS values are withheld, not reconstructed EEG. Collateral removal is measured separately.','', 'The 60 underlying audit examples are fixed by generation order (two per generated class per task/seed), reused across predictors: 180 predictor/example records are not 180 independent underlying draws. All 60 P0 clean predictions happened to be correct; E0/E1 wrong cases are retained and explained for their own predicted target, never retargeted to the true class. The greedy comparator is a local SIS-inspired backward search, not an official reproduction. Native/RMS/random summaries and greedy queries share verified tables; this avoids duplicated computation without making their standalone query cost equal to exhaustive WIC.','', 'Clean compactness means a discovered active clause can be displayed with at most eight retained seconds and eight signed literals. This is a bounded-search lower bound; the four displayed clauses are not all prime implicants and are subset-minimal, not globally minimum. Their coverage is a separate metric. Signed negative literals describe alternative internal states/possible repairs, not active reasons at the all-retained input. Prior-only responses claim no waveform support. Negative-nuisance overlap fields use base-view generator intervals, not a universal artifact annotation.','', '| Predictor | Nonmonotone cases | Negative / all signed clauses | Mean retained seconds | Mean neutral-state truth disagreement | Native 3-view universal / greedy clean universal |','|---|---:|---:|---:|---:|---:|']
    for m,v in stats.items():lines.append(f"| {m} | {v['nonmonotone_examples']}/60 | {v['negative_clause_count']}/{v['signed_clause_count']} | {v['retained_seconds_mean']:.3f} | {v['mean_neutral_truth_disagreement']:.6f} | {v['native_three_view_universal_fraction']:.3f}/{v['greedy_clean_universal_fraction']:.3f} |")
    lines += ['', '| Task | Seed | Predictor | Correct/wrong | Compact correct | Mean/max BDD nodes | Signed coverage |', '|---|---:|---|---:|---:|---:|---:|']
    all_groups=result['track_A1']['groups']+result['track_A2']['circuit_groups']
    for group in all_groups:
        rr=group['rows'];correct=[v for v in rr if v['correct']];nn=[v['nonterminal_nodes'] for r in rr for v in r['circuits']]
        lines.append(f"| {group['task']} | {group['seed']} | {group['predictor']} | {len(correct)}/{len(rr)-len(correct)} | {sum(v['compact_active_clean_found'] for v in correct)}/{len(correct)} | {np.mean(nn):.1f}/{max(nn)} | {np.mean([v['signed_summary']['coverage'] for v in rr]):.3f} |")
    lines += ['', 'Those last validity rates have different stated domains and are not a direct superiority score. Exact response fidelity is engineering; generated diagnostic alignment, readability, prediction quality and clinical validity are separate. Native mean pooling is never renormalized by retained count. Neutral .01 is a sensitivity control with RMS still zero, not proof of in-distribution intervention.','', '## Track A2: all clean matched heads','', 'All six P0 parents met independent clean development F1>=.80; zero ineligible cells. Four heads per parent received identical frozen encoder hashes, paired initialization, 150 batch16 clean TRAIN updates. All declared models have 3,593 parameters; E0/E1/E3 train 185 head parameters, E2 trains 95. Function classes differ despite matched declared capacity. No severe-corruption prerequisite suppressed this comparison.','', '| Task | Seed | Parent dev F1 | E0 clean/competitive | E1 clean/competitive | E2 clean/competitive | E3 clean/competitive | Utility gate |','|---|---:|---:|---|---|---|---|---|']
    for t in c['tasks']:
        for s in c['seeds']:
            parent=next(v for v in heads['eligibility'] if (v['task'],v['seed'])==(t,s));values=[]
            for m in c['head_methods']:
                row=next(v for v in heads['rows'] if (v['task'],v['seed'],v['method'])==(t,s,m));q=row['evaluation']['conditions'];values.append(f"{q['clean']['raw']['macro_f1']:.6f}/{q['competitive']['raw']['macro_f1']:.6f}")
            gate=next(v for v in heads['gate']['checks'] if (v['task'],v['seed'])==(t,s))
            lines.append(f"| {t} | {s} | {parent['development_macro_f1']:.6f} | "+' | '.join(values)+' | '+('PASS' if gate['pass'] else 'FAIL')+' |')
    lines += ['', '**Complete clean additive utility: FAIL (3/6 cells fail clean retention).** Single-witness E1 precision exceeds E3 RMS by at least .10 in all three cells, with higher event coverage. This does not establish superiority to E0: learned/conventional precision differences are much smaller, and E1 loses too much clean F1 in single/123, single/789 and multiple/456. IoU>=.25 is the historical interval-hit rule; 400-sample candidates can include irrelevant context and overlap. Five-output codes are synthetic, not clinically annotated stages.','', '## Track B: retained-evidence consistency','', '| Task | Seed | C0 clean/competitive | C1 clean/competitive | C2 clean/competitive | C2−C1 competitive | C2 full gate |','|---|---:|---|---|---|---:|---|']
    for check in b['gate']['checks']:
        t,s=check['task'],check['seed'];values=[]
        for m in ('C0','C1','C2'):
            row=next(v for v in b['rows'] if (v['task'],v['seed'],v['method'])==(t,s,m));q=row['evaluation']['conditions'];values.append(f"{q['clean']['raw']['macro_f1']:.6f}/{q['competitive']['raw']['macro_f1']:.6f}")
        lines.append(f"| {t} | {s} | "+' | '.join(values)+f" | {check['competitive_delta_C2_minus_C1']:+.6f} | "+('PASS' if check['pass'] else 'FAIL')+' |')
    inc=b['gate']['incremental'];ci=result['incremental_C2_minus_C1_seed_cluster_bootstrap95']
    lines += ['', f"**C2 full gate: {b['gate']['status']}; incremental benefit: {inc['status']}.** All six C2 clean F1 values meet .80 and retain C0 within .03; all competitive F1 values meet .70. Single/456 and multiple/123 fail paired diagnostic/nuisance behavior. Mean competitive delta versus C1 is {inc['mean_delta']:+.6f}; only {inc['positive_cells']}/6 cells improve, versus required mean>=.03 and >=4 positive cells. Three-seed cluster bootstrap95 interval for mean delta is [{ci[0]:+.6f}, {ci[1]:+.6f}], descriptive only. Both tasks stay together when resampling a seed; three seeds are not independent people.",'', 'C0/C1/C2 have identical architectures, initialization, minibatch ordering, clean/additive-nuisance pairs and update counts. C1 adds full-view JS; C2 adds two class-blind retained-mask queries, with fixed .5/.25 weights and 20% warmup. Masked subsets receive no original-label CE. This study tests paired generated interference, not TRADES reproduction or adversarial robustness. No capacity increase, seed search, loss sweep or post-hoc primary switching occurred. C2 has extra head query compute without demonstrated incremental benefit. No exact C-arm circuit superiority claim is made: exact circuit audits nominate P0/E0/E1, while C-arm null/removed conditions are reliability diagnostics.','', '## Independent-view stability','', '| Predictor | Full class stable | Fixed conditional | Substitute conditional | Fixed all cases | Substitute all cases | Gate |','|---|---:|---:|---:|---:|---:|---|']
    for m,v in result['stability']['decisions'].items():lines.append(f"| {m} | {v['stable_predictions']}/{v['examples']} | {v['fixed_active_conditional']:.3f} | {v['substitution_active_conditional']:.3f} | {v['fixed_all_case']:.3f} | {v['substitution_all_case']:.3f} | {v['status']} |")
    lines += ['', 'Unchanged clauses are tested against every mask completion in three independently sampled nuisance views, without refitting literals. The fixed/substitution quantifiers apply to the frozen displayed family. Full-input prediction stability is required separately; finding a favorable mask after the full prediction changes is not robust classification. The unrestricted full-input clause would be a trivial certificate whenever full predictions agree. Prior-only outputs and signed repair states do not count as active waveform explanations. All three predictors fail the .80 conditional fixed-clause criterion. Substitution can preserve some additional cases but does not rescue the full stability claim.','', '## Track C: operator-specific contracts','', '| Operator | Target removed mean | Nontarget max-error mean (uV or declared feature metric) | Historical compatibility accepted | Mean predicted-class margin effect |','|---|---:|---:|---:|---:|']
    opnames=['internal_availability','identity','exact_diagnostic_removal','exact_nuisance_removal','historical_bridge','historical_independent_baseline']
    for op in opnames:
        vs=[v for row in result['operators']['rows'] for v in row['records'] if v['operator']==op];target=[v['target_removal_fraction'] for v in vs if v['target_removal_fraction'] is not None];accept=[v['physical_compatibility']['accepted'] for v in vs if v['physical_compatibility'] is not None]
        lines.append('| '+op+' | '+(f'{np.mean(target):.6f}' if target else 'not a waveform erasure')+f" | {np.mean([v['nontarget_max_residual'] for v in vs]):.6f} | "+(f'{sum(accept)}/{len(accept)}' if accept else 'not applicable')+f" | {np.mean([v['margin_effect'] for v in vs]):.6f} |")
    lines += ['', f"Oracle component correctness: **{result['operators']['oracle_correctness']}**. Identity preserves the waveform but removes no target; exact diagnostic removal preserves every nontarget stored component even when total diagnostic spectrum changes. Diagnostic and nuisance controls use matching component semantics. Internal availability removes overlapping feature paths and can discard neighboring nontarget candidates; it does not generate a plausible waveform. The bridge is a linear waveform operator with residual target contribution and nontarget distortion measured separately. Independent replacement removes the original target but changes baseline context. Their predictive effects need not agree.",'', f"Both donor populations retain the historical .5 energy/.5 magnitude-L1/5uV boundary guards: {result['operators']['accepted_donors']}/{result['operators']['donor_attempts']} accepted. Accepted-donor conditional spread is null when fewer than two pass; no physiological validity or successful donor invariance is fabricated. Old donor failures remain unchanged. The partial-factor ordered-deletion condition remains descriptive/rule-identifiable and is never used as a label-preserving training pair.",'', '## Reliability, error strata and sanity controls','', 'All P0/E0–E3/C0–C2 rows retain clean, competitive, weak, artifact, eventless and diagnostic-removed NLL/Brier/ECE15, per-code recall/confusion, >=.9 incorrect confidence, calibrated and raw confidence, risk/coverage and paired-behavior intervals. Each predictor temperature is fitted only on its separate synthetic calibration stream. Full-view calibration does not imply calibration of all masked internal states. Constant-confidence correlation is correctly null. Incorrect original predictions are explained as model errors rather than relabeled targets.','', '| Arm | Mean competitive NLL / ECE | Incorrect confidence>=.9 | Removed accuracy / confidence |','|---|---:|---:|---:|']
    for m in ('C0','C1','C2'):
        group=[r for r in b['rows'] if r['method']==m];condition=[v['evaluation']['conditions'] for v in group]
        lines.append(f"| {m} | {np.mean([v['competitive']['raw']['nll'] for v in condition]):.6f}/{np.mean([v['competitive']['raw']['ece15'] for v in condition]):.6f} | {np.mean([v['competitive']['raw']['false_confidence_0_9'] for v in condition]):.6f} | {np.mean([v['removed']['raw']['accuracy'] for v in condition]):.6f}/{np.mean([v['removed']['raw']['confidence'] for v in condition]):.6f} |")
    lines += ['', 'Random-label controls keep model queries unchanged and report accuracy against permuted bookkeeping labels. Model-randomization controls query the same original target and measure response disagreement. They add no neural fit and do not retarget explanations to make them look correct. Sanity controls are in results.json. No population significance, clinical absence recognition or safety claim follows from calibration or these finite controls.','', '## Compute, integrity and limitations','', f"{res['neural_fits']} neural fits / {res['optimizer_updates']} updates; preserved first E0 counted once; no baseline reproduction. Cached-head training consumes {res['cached_head_training_examples']:,} feature examples and zero full-network training examples. Paired training consumes {res['paired_training_base_examples']:,} base examples and {res['paired_training_full_view_forwards']:,} encoded waveform views; C2 adds {res['paired_training_retained_query_head_forwards']:,} masked-head examples, not additional encoded waveforms. Scalar temperature fits ({res['scalar_calibration_fits']}) are separate from neural fits.",'', f"Completed scientific availability queries: {res['completed_scientific_availability_queries']:,}; reporting recovery repeats {res['reporting_recovery_duplicate_availability_queries']:,}. Native ranking and full-view calls are separately scoped and not called full-network mask forwards. Feature cache is 1,064 measured bytes/example/view; the full Boolean mask grid is 491,520 bytes. Exact margin/probability tables and full ROBDD archives stay private. Peak scientific process RSS {res['peak_scientific_process_rss_KiB']:,}KiB. Total internal scientific wall including failed reporting attempt: **{res['scientific_wall_seconds_including_failed_attempt']:.6f}s**, below 3,600s. Stage times: {res['per_track_wall_seconds']}. Python imports/setup, reporting and CI are excluded and not kernel-time measurements. Isolated first-E0 training time/loss were lost before failed export and remain null.",'', f"CPU WP13: **{cpu['passed']} passed / {cpu['skipped']} CUDA skips / {cpu['failures']} failures / {cpu['errors']} errors**, {cpu['seconds']:.3f}s. All 42 new checkpoint/creation/seal hashes and private circuit archive digests verify. CUDA **{cuda['status']}**: another compute process occupied the device; zero new CUDA model operations. No historical GPU success is substituted. Final publication verification/CI may add privacy tests; the exact final counts and final commit/Actions URL are recorded in the PR and final response to avoid a self-referential commit hash.",'', 'Main limits: one synthetic generator family with intentional local spectra; three optimization seeds; stratified small explanation audit; a neutral-state-dependent internal operator; bounded incomplete clause enumeration; calibration learned on full views; no external explainer implementation benchmark; and no clinical data or event annotations. Results are not a novelty certification. See the focused primary-source matrix in PHASE_IX_METHOD_AND_NOVELTY_V1.md; SIS, formal reasons, ROBDDs, MIL and consistency objectives are established. The supported contribution is this tested combination and its scoped findings, not a new Boolean theorem or first EEG explanation method.','', '## Reproduction and next decision','', 'The default CLI verifies metadata and never trains. Explicit stages are sequential and refuse completed/stopped experiment reuse unless the pinned reporting-only continuation applies. Replaying elsewhere needs the authenticated private P0 artifacts or a separately identified authorized reproduction. Existing output paths must be preserved.','', '```bash', "export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''", 'CPU_PYTHON=mist_transfer_runs/mist_cove/phase4_v1/ci_cpu_environment/bin/python', '"$CPU_PYTHON" scripts/run_mist_cove_phase9.py verify', '"$CPU_PYTHON" -m pytest tests -ra', '# Archived executed scientific stages: circuits, heads, robustness,', '# stability, operators, reliability. Completed directories refuse repetition.', '```', '', 'Next priorities: assess human usefulness of compact operator-specific explanations and finite-view failure cases; do not spend more query-consistency compute on the same failed claim without a new hypothesis; and review a narrow source-model dependency pilot separately. A synthetic robustness failure is not a universal veto on every justified source question, but this execution grants no real-data access. No automatic Phase X follows.']
    if 'final_cpu' in result['correctness']:
        final=result['correctness']['final_cpu'];lines += ['',f"Final publication regression: **{final['passed']} passed / {final['skipped']} CUDA skips / {final['failures']} failures / {final['errors']} errors**, {final['seconds']:.3f}s. This adds publication/privacy checks after the WP13 suite; it does not replace its receipt."]
    return '\n'.join(lines)+'\n'


def publish_report():
    progress=json.loads((PUBLIC/'progress.json').read_text())
    if progress['work_packages'][12]['status']!='COMPLETE':raise ValueError('WP13 must complete first')
    update(14,'IN_PROGRESS');started=time.perf_counter();receipts=verified_receipts();result=build_results(receipts)
    publish_json(PUBLIC/'results.json',result);figures=render_figures(receipts)
    report=report_markdown(result);publish_bytes(ROOT/'docs/mist_cove/PHASE_IX_RESULTS_V1.md',report.encode())
    resource=result['resources'];stats=result['circuit_statistics']
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>MIST-WIC Phase IX</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17232b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3d8;padding:8px;text-align:left}figure{margin:30px 0}img{max-width:100%}.fail{color:#9e2020}</style><h1>MIST-WIC Phase IX</h1>', '<p>Exact finite-model circuits; synthetic only. No clinical validity or real EEG access.</p>', '<h2>Scientific decisions</h2><p class="fail">Clean additive utility: FAIL. C2 complete robustness: FAIL. Incremental query consistency: FAIL. Independent-view fixed-witness stability: FAIL.</p>', '<table><tr><th>Predictor</th><th>Correct/wrong audit</th><th>Compact correct</th><th>Signed coverage</th><th>Stable class</th><th>Fixed/substitute all cases</th></tr>']
    for m,v in stats.items():
        s=result['stability']['decisions'][m];parts.append(f"<tr><td>{m}</td><td>{v['correct']}/{v['wrong']}</td><td>{v['compact_correct']}/{v['correct']}</td><td>{v['mean_signed_summary_coverage']:.3f}</td><td>{s['prediction_stability']:.3f}</td><td>{s['fixed_all_case']:.3f}/{s['substitution_all_case']:.3f}</td></tr>")
    parts += ['</table>',f"<p>{resource['neural_fits']} neural fits; {resource['optimizer_updates']} updates; {resource['scientific_wall_seconds_including_failed_attempt']:.3f}s scientific CPU wall. CUDA: {html.escape(result['correctness']['cuda']['status'])}.</p>",'<p><a href="results.json">All task/seed scores, calibration, operator and integrity records (JSON)</a> · <a href="progress.json">WP01–WP16 ledger</a></p>']
    for name in figures:parts.append(f'<figure><img src="figures/{name}" alt="{html.escape(name)}"></figure>')
    parts += ['<p>Next: review operator-specific explanation usefulness, preserve negative training results, and separately authorize any narrowly scoped source pilot. No automatic Phase X.</p></html>']
    publish_bytes(PUBLIC/'dashboard.html',''.join(parts).encode())
    manifest={'version':'phase9-publication-v1','files':{},'figure_sha256':figures,'principal_figures':len(figures),'scientific_inputs':result['private_receipt_sha256'],'real_data_accessed':False,'complete_per_mask_tables_public':False}
    for path in [PUBLIC/'results.json',PUBLIC/'dashboard.html',ROOT/'docs/mist_cove/PHASE_IX_RESULTS_V1.md']+[PUBLIC/'figures'/name for name in figures]:manifest['files'][str(path.relative_to(ROOT))]=file_sha(path)
    publish_json(PUBLIC/'publication_manifest.json',manifest)
    update(14,'COMPLETE',artifacts=['docs/mist_cove/PHASE_IX_RESULTS_V1.md','reports/mist_cove/phase9_v1/results.json','reports/mist_cove/phase9_v1/dashboard.html','reports/mist_cove/phase9_v1/figures'],tests=['all private receipt/checkpoint/circuit hashes verified','public schema excludes states and per-mask arrays','six principal figures'],findings=['Track-specific scientific failures retained independently','Exhaustive finite fidelity and bounded clean compactness supported'])
    print(json.dumps({'status':'COMPLETE','principal_figures':len(figures),'report_generation_seconds':time.perf_counter()-started,'scientific_wall_seconds':resource['scientific_wall_seconds_including_failed_attempt']}))


def verify_publication(*,check_git=True):
    manifest=json.loads((PUBLIC/'publication_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if file_sha(ROOT/name)!=expected:raise ValueError('publication mismatch: '+name)
    if len(manifest['figure_sha256'])>6:raise ValueError('principal figure ceiling exceeded')
    for name,expected in manifest['figure_sha256'].items():
        if file_sha(PUBLIC/'figures'/name)!=expected:raise ValueError('figure mismatch')
    for p in PUBLIC.glob('*.json'):privacy_check(json.loads(p.read_text()))
    if check_git:
        import subprocess
        frozen();plan_proof()
        changes=subprocess.check_output(['git','diff','--name-status','b50e41efb3b09a9cb7fa4142366424725fc603e5','--'],cwd=ROOT,text=True).splitlines()
        if any(not v.startswith('A\t') for v in changes):raise ValueError('historical tracked file changed')
    return {'status':'PASS','verified_files':len(manifest['files']),'principal_figures':len(manifest['figure_sha256']),'real_data_opens':0,'private_state_or_table_exports':0}

"""Audit and publish aggregate synthetic metrics only; never opens EEG/weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mist_cove.phase2.provenance import implementation_snapshot,digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(source,output):
    source=Path(source);output=Path(output);r=json.loads(source.read_text())
    assert r['status']=='CONFIRMATORY_SYNTHETIC_COMPLETE'
    assert r['implementation']==implementation_snapshot()
    assert r['config_sha256']==digest(r['config'])
    for p,h in r['preregistration']['files'].items():assert sha(p)==h,p
    rows=r['rows'];c=r['config']
    expected={(t,s,m,'end_to_end') for t in c['tasks'] for s in c['seeds'] for m in c['methods']}
    expected|={(t,s,m,'fixed_encoder') for t in c['tasks'] for s in c['seeds'] for m in c['frozen_encoder_heads']}
    assert len(rows)==54 and {(v['task'],v['seed'],v['method'],v['regime']) for v in rows}==expected
    for row in rows:
        steps=300 if row['regime']=='end_to_end' else 150
        assert row['training']['optimizer_updates']==steps and row['training']['training_example_exposures']==steps*16
        assert set(row['conditions'])==set(c['conditions'])
        assert max(v['max_additive_residual'] for v in row['conditions'].values())<=1e-5
        assert max(v['max_reconciliation_residual'] for v in row['audits'].values())<=1e-5
        if row['regime']=='fixed_encoder':
            parent=next(p for p in rows if (p['task'],p['seed'],p['method'],p['regime'])==(row['task'],row['seed'],'M1','end_to_end'))
            assert parent['encoder_digest']==row['encoder_digest']
    summary={}
    for task in c['tasks']:
        summary[task]={}
        for regime,methods in [('end_to_end',c['methods']),('fixed_encoder',c['frozen_encoder_heads'])]:
            summary[task][regime]={}
            for method in methods:
                group=[v for v in rows if (v['task'],v['regime'],v['method'])==(task,regime,method)]
                conditions={}
                for condition in c['conditions']:
                    met={}
                    for key in ['macro_f1','accuracy','nll','brier','ece15','confidence','balanced_accuracy','kappa']:
                        vals=[v['conditions'][condition][key] for v in group]
                        met[key]={'mean':float(np.mean(vals)),'seed_sd':float(np.std(vals,ddof=1))}
                    met['per_stage_f1_mean']=np.mean([v['conditions'][condition]['per_stage_f1'] for v in group],axis=0).tolist()
                    met['per_stage_recall_mean']=np.mean([v['conditions'][condition]['per_stage_recall'] for v in group],axis=0).tolist()
                    met['confusion_sum_over_generator_seeds']=np.sum([v['conditions'][condition]['confusion_matrix'] for v in group],axis=0).tolist()
                    met['top1_localization_mean']=float(np.mean([v['conditions'][condition]['localization']['top1_hit'] for v in group]))
                    conditions[condition]=met
                summary[task][regime][method]={'conditions':conditions,'parameters':group[0]['training']['parameters'],
                    'trainable_parameters':group[0]['training']['trainable_parameters'],
                    'total_updates':sum(v['training']['optimizer_updates'] for v in group),
                    'training_seconds':sum(v['training']['wall_seconds'] for v in group),
                    'run_wall_seconds':sum(v['run_wall_seconds'] for v in group)}
    output.mkdir(parents=True,exist_ok=False)
    with (output/'per_run_v2.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    metadata={k:v for k,v in r.items() if k!='rows'}
    metadata.update(original_private_result_sha256=sha(source),per_run_aggregate_file='per_run_v2.jsonl',
                    per_run_aggregate_sha256=sha(output/'per_run_v2.jsonl'),n_runs=len(rows),
                    all_shared_encoders_verified_identical=True,total_optimizer_updates=sum(v['training']['optimizer_updates'] for v in rows),
                    total_example_exposures=sum(v['training']['training_example_exposures'] for v in rows))
    (output/'synthetic_receipt_v2.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (output/'aggregate_v2.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (output/'per_run_metrics.csv').open('x',newline='') as stream:
        fields=['task','seed','regime','method','parameters','trainable','updates','training_seconds','run_wall_seconds','clean_f1','competitive_f1','competitive_top1_hit','erased_confidence','clean_donor_spread']
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(fields)
        for v in rows:
            tr=v['training'];writer.writerow([v['task'],v['seed'],v['regime'],v['method'],tr['parameters'],tr['trainable_parameters'],tr['optimizer_updates'],tr['wall_seconds'],v['run_wall_seconds'],v['conditions']['clean']['macro_f1'],v['conditions']['competitive']['macro_f1'],v['conditions']['competitive']['localization']['top1_hit'],v['conditions']['erased']['confidence'],v['audits']['clean']['donor_spread_max_mean']])
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ti,task in enumerate(c['tasks']):
        for ci,condition in enumerate(['clean','competitive']):
            ax=axes[ti,ci];values=summary[task]['end_to_end'];methods=c['methods']
            means=[values[m]['conditions'][condition]['macro_f1']['mean'] for m in methods]
            sd=[values[m]['conditions'][condition]['macro_f1']['seed_sd'] for m in methods]
            ax.bar(range(len(methods)),means,yerr=sd,capsize=3,color=['#888888','#2b6c9b','#d78040','#888888','#c15d61','#55945c'])
            ax.set(xticks=range(len(methods)),xticklabels=methods,ylim=(0,1),ylabel='Fixed-five synthetic Macro-F1',title=task+' / '+condition)
            ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Generated phase codes; mean and SD over three seeds. No clinical EEG.')
    fig.savefig(output/'synthetic_comparison.png',dpi=150);plt.close(fig)
    return metadata,summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(a.source,a.output)

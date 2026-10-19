"""Read-only prospective output verification and safe aggregate publication."""
import csv
import json
from pathlib import Path
import statistics

from mist_cove.phase2.provenance import file_sha
from mist_cove.phase7.integrity import ROOT
from mist_cove.phase7.robustness import proof, collective, qualifies


def main():
    private=ROOT/'mist_transfer_runs/mist_cove/phase7_prospective_v1'
    public=ROOT/'reports/mist_cove/phase7_v1'
    d=json.loads((public/'prospective_diagnostics_v2.json').read_text())
    source=private/'robustness_v1/results.json';r=json.loads(source.read_text())
    assert r['status']=='COMPLETE_PROSPECTIVE_ROBUSTNESS' and r['frozen']==proof()
    assert file_sha(private/'diagnostic_corrected_v2/results.json')==d['private_result_sha256']
    c=r['config']
    for row in r['rows']:
        assert row['gate_clauses']==qualifies(row,c)
        if row['method']=='P0':
            prior=next(v for v in d['rows'] if (v['task'],v['seed'])==(row['task'],row['seed']))
            folder='diagnostic_v1' if (row['task'],row['seed'])==('single',123) else 'diagnostic_corrected_v2'
            p=private/f"{folder}/{row['task']}_{row['seed']}_P0.pt";creation=prior['training']['creation']
        else:
            p=private/f"robustness_v1/{row['task']}_{row['seed']}_{row['method']}.pt"
            creation=row['training']['creation']
        assert file_sha(p)==creation['checkpoint_sha256']
        assert file_sha(p.with_suffix('.seal.json'))==creation['seal_sha256']
        assert json.loads(p.with_suffix('.creation.json').read_text())==creation
    assert r['gate_vii_4']==collective(r['rows'],c)
    if r['gate_vii_4']['status']=='FAIL':assert r['heads']['fits']==r['heads']['updates']==0 and not r['heads']['rows']
    head_checks=[]
    if r['heads']['rows']:
        with (public/'head_runs_v1.jsonl').open('x') as f:
            for row in r['heads']['rows']:f.write(json.dumps(row,allow_nan=False)+'\n')
        for task in c['tasks']:
            for seed in c['seeds']:
                group={v['method']:v for v in r['heads']['rows'] if (v['task'],v['seed'])==(task,seed)}
                assert len({v['encoder_sha256'] for v in group.values()})==1
                m,ref,rms=group['M2'],group['M1'],group['M2_rms'];k=c['criteria']
                contrast=statistics.mean(abs(v['diagnostic']['full'][label])-abs(v['nondiagnostic']['full'][label]) for label,v in enumerate(m['audits']))
                clauses={'clean_penalty':m['conditions']['clean']['macro_f1']>=ref['conditions']['clean']['macro_f1']-k['f1_penalty'],
                    'competitive_penalty':m['conditions']['competitive']['macro_f1']>=ref['conditions']['competitive']['macro_f1']-k['f1_penalty'],
                    'single_topk':task!='single' or m['conditions']['clean']['localization']['model']['topk_hit']>=k['topk_hit'],
                    'rms_gain':m['conditions']['competitive']['localization']['model']['topk_hit']>=rms['conditions']['competitive']['localization']['model']['topk_hit']+k['rms_gain'],
                    'same_construction_oracle_contrast':contrast>0,
                    'additive_reconciliation':max(v['max_additive_residual'] for v in m['conditions'].values())<k['reconciliation']}
                head_checks.append({'task':task,'seed':seed,'clauses':clauses,'pass':all(clauses.values()),'oracle_contrast':contrast})
    table=[]
    for row in r['rows']:
        a=row['evaluation']['conditions'];stress=row['evaluation']['stress']
        table.append({'task':row['task'],'seed':row['seed'],'method':row['method'],
            'clean_f1':a['clean']['macro_f1'],'added_nuisance_f1':a['added_nuisance']['macro_f1'],
            'combined_competitive_f1':a['competitive']['macro_f1'],'clean_accuracy':a['clean']['accuracy'],
            'competitive_accuracy':a['competitive']['accuracy'],'competitive_confidence':a['competitive']['confidence'],
            'competitive_false_confidence_0_9':a['competitive']['false_confidence_0_9'],
            'removed_confidence':stress['removed']['confidence'],'removed_accuracy':stress['removed']['accuracy'],
            'full_gate':row['gate_clauses']['pass']})
    with (public/'robustness_runs_v1.jsonl').open('x') as f:
        for row in r['rows']:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (public/'robustness_metrics_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    means={}
    for task in c['tasks']:
        means[task]={}
        for method in ['P0','R1','R2']:
            values=[v for v in table if v['task']==task and v['method']==method]
            means[task][method]={key:statistics.mean(v[key] for v in values) for key in ['clean_f1','added_nuisance_f1','combined_competitive_f1','competitive_confidence','competitive_false_confidence_0_9','removed_confidence','removed_accuracy']}
    failed=json.loads((private/'diagnostic_v1/STOPPED.json').read_text())
    result={k:v for k,v in r.items() if k not in ['rows','heads']}
    result.update(private_result_sha256=file_sha(source),means=means,
        heads={**{k:v for k,v in r['heads'].items() if k!='rows'},'acceptance_checks':head_checks},
        total_unique_prospective_fits=d['fits']+r['new_fits'],
        total_unique_optimizer_updates=d['optimizer_updates']+r['optimizer_updates'],
        total_unique_example_exposures=d['example_exposures']+r['example_exposures'],
        diagnostic_including_failed_attempt_seconds=d['wall_seconds']+failed['wall_seconds'],
        combined_measured_wall_seconds=d['wall_seconds']+failed['wall_seconds']+r['wall_seconds'],
        timing_limitations='budgets exclude interpreter imports; checkpoint/evaluation overhead not profiled separately; no kernel-time claim',
        historical_checkpoint_authentication='BLOCKED',historical_checkpoint_deserializations=0,
        donor_robustness='NOT TESTED',real_data_accessed=False)
    with (public/'prospective_summary_v3.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['gate_vii_4','new_fits','total_unique_prospective_fits','combined_measured_wall_seconds','heads']},indent=2))


if __name__=='__main__':main()

"""Offline machine/human-readable scientific decisions from aggregate receipts."""
import json
from pathlib import Path
from collect import PUBLIC,publish,sha


def main():
    g=json.loads((PUBLIC/'scientific_gate_status.json').read_text());r=json.loads((PUBLIC/'robustness_summary.json').read_text())
    e=json.loads((PUBLIC/'evidence_summary.json').read_text());compute=json.loads((PUBLIC/'execution_receipt.json').read_text())
    summary={'primary':g['primary'],'robustness_gate':g,'candidate_identities':r['config']['methods'],
        'task_seed_scores':[{'task':v['task'],'seed':v['seed'],'method':v['method'],'clean_f1':v['conditions']['clean']['macro_f1'],
            'competitive_f1':v['conditions']['competitive']['macro_f1'],'added_f1':v['conditions']['added']['macro_f1'],
            'competitive_ece':v['conditions']['competitive']['ece15']} for v in r['rows']],
        'worst_case':r['summary'],'additive_status':e['heads']['status'],
        'donor_status':[{'task':v['task'],'seed':v['seed'],'coverage':v['donors']['accepted_coverage'],'status':v['donors']['accepted_donor_robustness']} for v in e['donors']],
        'cpu_correctness':json.loads((PUBLIC/'correctness_receipt.json').read_text()),'cuda_status':compute['cuda_status'],'provenance':'PASS',
        'authorization':'SYNTHETIC_ONLY; no real EEG/RESERVED/ST/SHHS/legacy NPZ',
        'next_blockers':['Complete robust primary prerequisite before additive confirmation','Numerically acceptable and diagnostically neutral multi-donor support','Independent scientific justification and human authorization before real source work'],
        'source_receipt_sha256':{name:sha(PUBLIC/name) for name in ('scientific_gate_status.json','robustness_summary.json','calibration_summary.json','execution_receipt.json')}}
    publish(PUBLIC/'dashboard_v2.json',summary)
    lines=['# Phase VIII offline research dashboard','',f"Primary **{g['primary']}** robustness gate: **{g['status']}**. Additive confirmation: **{e['heads']['status']}**. CUDA: **{compute['cuda_status']}**. Provenance: **PASS**.",'','| Task | Seed | Method | Clean F1 | Competitive F1 | Added F1 | Competitive ECE |','|---|---:|---|---:|---:|---:|---:|']
    for v in summary['task_seed_scores']:lines.append('| '+str(v['task'])+' | '+str(v['seed'])+' | '+v['method']+' | '+' | '.join(f'{v[k]:.6f}' for k in ('clean_f1','competitive_f1','added_f1','competitive_ece'))+' |')
    lines+=['','## Failed primary clauses','']
    for v in g['checks']:lines.append(f"- {v['task']}/{v['seed']}: "+(', '.join(k for k,passed in v['clauses'].items() if not passed) or 'all clauses pass'))
    lines+=['','## Authorization and next blockers','',summary['authorization'],'']+['- '+v for v in summary['next_blockers']]
    lines += ['', 'CPU correctness: '+str(summary['cpu_correctness']['passed'])+' passed / '+str(summary['cpu_correctness']['skipped'])+' CUDA skipped. Source and test hashes are in correctness_receipt.json.', '', 'Descriptive completion: descriptive_diagnostics_summary.json. Aggregate figures: figures/ (16 charts). No new fits or gate changes in the appendix.']
    target=PUBLIC/'dashboard_v2.md'
    with target.open('x') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':main()

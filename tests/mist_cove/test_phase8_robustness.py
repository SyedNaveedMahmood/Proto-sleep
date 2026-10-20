from mist_cove.phase8.evaluation import gate

def rows():
    result=[]
    for task in ('single','multiple'):
        for seed in (123,456,789):
            for method in ('P0','B1'):
                result.append({'task':task,'seed':seed,'method':method,
                    'conditions':{k:{'macro_f1':.9,'collapsed':False} for k in ('clean','competitive','added')},
                    'pairs':{k:{'appropriate_diagnostic_change':.9,'nuisance_agreement':.9} for k in ('clean','competitive')}})
    return result

C={'tasks':['single','multiple'],'seeds':[123,456,789],'primary':'B1'}

def test_collective_gate_every_seed_task_and_no_rounding():
    r=rows();assert gate(r,C)['status']=='PASS'
    r[-1]['conditions']['competitive']['macro_f1']=.69999999
    assert gate(r,C)['status']=='FAIL' and gate(r[:-1],C)['status']=='FAIL'

def test_clean_penalty_and_pairs_gate_even_when_macro_f1_passes():
    r=rows();r[-1]['conditions']['clean']['macro_f1']=.86;assert gate(r,C)['status']=='FAIL'
    r=rows();r[-1]['pairs']['competitive']['nuisance_agreement']=.799;assert gate(r,C)['status']=='FAIL'

def test_failure_taxonomy_and_independent_donor_pipeline():
    import json,torch
    from mist_cove.phase8.provenance import CONFIG
    from mist_cove.phase8.robustness import fail_analysis
    from mist_cove.phase8.interventions import donor_audit
    from mist_cove.phase4.model import make_model
    from mist_cove.phase6.generator import generate
    from mist_cove.phase5.runtime import Budget
    c=json.loads(CONFIG.read_text());m=make_model('M1');d=generate(10,11128,'single','competitive')
    with torch.no_grad():removed=m(d.base)
    result=fail_analysis(m,d,removed)
    assert result['n']==10 and 0<=result['localized_but_wrong']<=10
    donors=donor_audit(m,'single',11128,c,Budget(60))
    assert sum(v['attempted'] for v in donors['population'].values())==250
    assert 0<=donors['accepted_coverage']<=1
    if not donors['multiple_accepted_examples']:assert donors['mean_effect_spread'] is None

"""Aggregate finite-circuit evidence without publishing source coordinates."""
import json
import numpy as np
from mist_cove.phase11b.evaluation import distribution,clustered_mean
from .provenance import PUBLIC,PRIVATE,write_new


def read_rows(path):
    with path.open() as stream:return [json.loads(line) for line in stream]


def compact_active(row):
    return any(c['negative_bits']==0 and 0<c['retained_seconds']<=8 and c['signed_literals']<=8 for c in row['summary']['signed']['clauses'])


def group(rows):
    count=len(rows)
    if not count:return {'cases':0}
    ss=[r['summary'] for r in rows];subjects=[r['coordinate']['subject'] for r in rows]
    positive=[s['positive_minimum_mask'] for s in ss]
    result={'cases':count,'people':len(set(subjects)),'correct':sum(r['correct'] for r in rows),
        'exact_tables':sum(s['exact_all_rows'] for s in ss),'positive_support_exists':sum(v is not None for v in positive),
        'exact_nonempty_compact_positive':sum(v is not None and 0<v.bit_count()<=4 for v in positive),
        'displayed_nonempty_compact_active':sum(compact_active(r) for r in rows),
        'prior_response':sum(s['prior_response'] for s in ss),'universal_prior':sum(s['universal_prior'] for s in ss),
        'positive_support_seconds':distribution([v.bit_count()*2 for v in positive if v is not None]),
        'BDD_nodes':distribution([s['BDD_nonterminal_nodes'] for s in ss]),
        'displayed_clause_state_coverage':distribution([s['signed']['coverage'] for s in ss]),
        'nonmonotone_cases':sum(s['nonmonotone_edges']['response_lost_when_adding_atom']>0 for s in ss),
        'signed_withheld_literal_cases':sum(any(c['negative_bits'] for c in s['signed']['clauses']) for s in ss),
        'global_stable_local_changed':sum(r['branch_predictions']['global_only']==r['prediction'] and r['branch_predictions']['local_only']!=r['prediction'] for r in rows),
        'local_stable_global_changed':sum(r['branch_predictions']['local_only']==r['prediction'] and r['branch_predictions']['global_only']!=r['prediction'] for r in rows),
        'participant_compact_fraction':clustered_mean([compact_active(r) for r in rows],subjects),
        'views':{},'physical':{},'neutral':{}}
    for view in rows[0]['views']:
        vv=[r['views'][view] for r in rows];eligible=[v for v in vv if v['fixed_clause_valid'] is not None]
        result['views'][view]={'cases':count,'prediction_stable':sum(v['prediction_stable'] for v in vv),
            'fixed_clause_eligible':len(eligible),'fixed_clause_valid':sum(v['fixed_clause_valid'] for v in eligible),
            'active_substitution':sum(v['active_substitution'] for v in vv),
            'any_original_active_valid':sum(v['any_original_active_clause_valid'] for v in vv),
            'prediction_stable_and_fixed_invalid':sum(v['prediction_stable'] and v['fixed_clause_valid'] is False for v in vv),
            'prediction_stability_clustered':clustered_mean([v['prediction_stable'] for v in vv],subjects)}
    for kind in ('bridge','background'):
        rr=[p for r in rows for p in r['replacements'] if p['kind']==kind];attempted=[p for p in rr if p['status']=='ATTEMPTED']
        eligible=[p for p in attempted if p['historical_compatibility']]
        # Count comparisons using each case's original exact withheld response.
        comparisons=[]
        for r in rows:
            comparisons.extend([e for e in r['triad']['edges'] if e['type']=='OPERATOR_COMPARISON' and e['operator']==kind])
        result['physical'][kind]={'requests':len(rr),'attempted':len(attempted),'no_eligible_background':sum(p['status']=='NO_ELIGIBLE_BACKGROUND' for p in rr),
            'nontarget_exact':sum(p['nontarget_exact'] for p in attempted),'historically_compatible':len(eligible),
            'compatible_target_response_lost':sum(not p['target_retained'] for p in eligible),
            'operator_disagreements':sum(e['disagrees'] for e in comparisons),
            'unrestricted_prediction_changes':sum(p['prediction']!=r['prediction'] for r in rows for p in r['replacements'] if p['kind']==kind and p['status']=='ATTEMPTED'),
            'clinical_target_removal':'NOT_TESTED','label_preservation':'UNVERIFIED'}
    nn=[r['neutral'] for r in rows if 'neutral' in r]
    if nn:result['neutral']={'cases':len(nn),'any_table_change':sum(v['table_disagreement_fraction']>0 for v in nn),
        'fraction_states_changed':distribution([v['table_disagreement_fraction'] for v in nn]),
        'positive_minimum_changed':sum(v['positive_minimum_mask_changed'] for v in nn),'prior_changed':sum(v['prior_changed'] for v in nn)}
    return result


def summarize():
    execution=json.loads((PUBLIC/'mechanistic_execution.json').read_text())
    if execution['status']!='COMPLETE':raise ValueError('complete fixed denominator required')
    original=read_rows(PRIVATE/'original_primary_mechanistic.jsonl');models=[]
    for seed in (123,456,789):
        for size in ('S','L'):
            for route in ('R0','R1'):
                for step in (300,1000,3000):
                    rr=read_rows(PRIVATE/f'{size}_{route}_{seed}_{step}_mechanistic.jsonl')
                    if len(rr)!=25:raise ValueError('mechanistic stratum missing')
                    models.append({'capacity':size,'routing':route,'seed':seed,'step':step,'all':group(rr),
                        'correct':group([r for r in rr if r['correct']]),'wrong':group([r for r in rr if not r['correct']])})
    write_new(PUBLIC/'explanation_validity.json',{'human_annotation':'NOT_TESTED','physiological_alignment':'NOT_TESTED',
        'review_infrastructure':'PASS_GENERATED_ONLY','primary_original':group(original),
        'original_correct':group([r for r in original if r['correct']]),'original_wrong':group([r for r in original if not r['correct']]),
        'new_states':models,'secondary':'30enriched annotation-only cases; not pooled',
        'limitations':['Signed withheld literals describe counterfactual internal states','No independent event labels or clinical target deletion','25mechanistic cases from5development people; same cases repeated across36states',
            'Original150 balanced early-epoch examples from historically fitted TRAIN people','Displayed compactness is distinct from complete graph accuracy']})
    print('Original150 and900new state-case pairs aggregated; clinical alignment NOT TESTED')

"""Aggregate-only publication from private fixed audit rows."""
import json
import numpy as np
from .campaign import PUBLIC,PRIVATE,write_new
from .evaluation import distribution,clustered_mean,public_metrics
from .triad import VIEW_NAMES,PROBE_NAMES

def compact(summary,active=True):
    return any(c['retained_seconds']<=8 and c['signed_literals']<=8 and c['positive_bits']!=0
               and (not active or c['negative_bits']==0) for c in summary['signed']['clauses'])

def group_summary(rows):
    subjects=[r['coordinate']['subject'] for r in rows]
    y=np.array([r['coordinate']['stage'] for r in rows]);z=np.array([r['native']['logits'] for r in rows]);s=[r['summary'] for r in rows]
    rates={'exact_all_rows':[v['exact_all_rows'] for v in s],
           'canonical_empty_target_response':[v['prior_response'] for v in s],
           'universally_constant_target':[v['universal_prior'] for v in s],
           'compact_nonempty_active_summary':[compact(v) for v in s],
           'compact_nonempty_any_signed_summary':[compact(v,False) for v in s],
           'has_signed_interfering_literal':[any(c['negative_bits'] for c in v['signed']['clauses']) for v in s],
           'whole_epoch_positive_minimum':[v['positive_minimum_mask']==32767 for v in s],
           'incomplete_readable_summary':[v['signed']['coverage']<1 for v in s],
           'local_only_prediction_stable':[r['feature_ablations']['local_only']['prediction']==r['native']['prediction'] for r in rows],
           'global_only_prediction_stable':[r['feature_ablations']['global_only']['prediction']==r['native']['prediction'] for r in rows]}
    result={'n':len(rows),'classification':public_metrics(y,z,subjects),'rates':{k:{'count':int(sum(v)),'denominator':len(v),'participant_interval':clustered_mean(v,subjects)} for k,v in rates.items()},
            'BDD_nodes':distribution([v['BDD_nonterminal_nodes'] for v in s]),
            'signed_true_mask_coverage':distribution([v['signed']['coverage'] for v in s]),
            'positive_only_true_mask_coverage':distribution([v['positive_only']['coverage'] for v in s]),
            'canonical_minimum_seconds':distribution([2*v['canonical_minimum_mask'].bit_count() for v in s if v['canonical_minimum_mask'] is not None]),
            'universal_positive_minimum_seconds':distribution([2*v['positive_minimum_mask'].bit_count() for v in s if v['positive_minimum_mask'] is not None]),
            'full_identity_max_abs':max(r['full_identity_max_abs'] for r in rows)}
    for correct in (True,False):
        group=[r for r in rows if r['correct']==correct]
        result['correct' if correct else 'incorrect']={'n':len(group),'compact_active':sum(compact(r['summary']) for r in group),
                      'canonical_minimum_seconds':distribution([2*r['summary']['canonical_minimum_mask'].bit_count() for r in group if r['summary']['canonical_minimum_mask'] is not None]),
                      'BDD_nodes':distribution([r['summary']['BDD_nonterminal_nodes'] for r in group])}
    differences=[];units=[]
    for subject in sorted(set(subjects)):
        correct=[compact(r['summary']) for r in rows if r['coordinate']['subject']==subject and r['correct']]
        incorrect=[compact(r['summary']) for r in rows if r['coordinate']['subject']==subject and not r['correct']]
        if correct and incorrect:differences.append(np.mean(correct)-np.mean(incorrect));units.append(subject)
    result['compact_correct_minus_incorrect']=clustered_mean(differences,units)
    return result

def run():
    rows=[json.loads(line) for line in (PRIVATE/'audit_rows.jsonl').read_text().splitlines()];primary=[r for r in rows if r['primary']]
    if len(rows)!=750 or len(primary)!=150:raise ValueError('preserve incomplete denominator before summary')
    result={'primary':group_summary(primary),'extended_inclusive':group_summary(rows),'extension_is_independent':False}
    write_new(PUBLIC/'circuit_metrics.json',result)
    subjects=[r['coordinate']['subject'] for r in primary];y=np.array([r['coordinate']['stage'] for r in primary])
    stability={}
    for view in VIEW_NAMES:
        values=[r['triad']['stability'][view] for r in primary]
        stability[view]={k:{'count':int(sum(v[k] is True for v in values)),'eligible':sum(v[k] is not None for v in values)} for k in ('prediction_stable','fixed_clause_valid','any_original_clause_valid','any_original_active_clause_valid','active_substitution')}
        stability[view]['fixed_valid_clustered']=clustered_mean([v['fixed_clause_valid'] is True for v in values],subjects)
    probes={}
    for name in PROBE_NAMES:
        z=np.array([r['probes'][name]['logits'] for r in primary]);p=[r['probes'][name] for r in primary]
        probes[name]={'metrics':public_metrics(y,z,subjects),
                      'prediction_stable':int(sum(v['prediction']==r['native']['prediction'] for v,r in zip(p,primary))),
                      'target_margin_change':clustered_mean([v['target_margin']-r['native']['target_margin'] for v,r in zip(p,primary)],subjects),
                      'accuracy_change':clustered_mean([(v['prediction']==r['coordinate']['stage'])-int(r['correct']) for v,r in zip(p,primary)],subjects),
                      'energy_distortion':distribution([v['distortion']['relative_energy_distortion'] for v in p]),
                      'spectral_distortion':distribution([v['distortion']['spectral_magnitude_L1'] for v in p])}
    operators={}
    for kind in ('bridge','background'):
        attempts=[v for r in primary for v in r['replacements'] if v['kind']==kind];eligible=[v for v in attempts if v['status']=='ATTEMPTED'];valid=[v for v in eligible if v['historical_compatibility']]
        operators[kind]={'requests':len(attempts),'no_eligible_background':len(attempts)-len(eligible),'attempted':len(eligible),'compatible':len(valid),
                         'compatible_target_losses':sum(not v['target_retained'] for v in valid),'rejected_target_losses':sum(not v['target_retained'] for v in eligible if not v['historical_compatibility']),
                         'all_nontarget_exact':all(v['nontarget_exact'] for v in eligible),'energy_distortion':distribution([v['relative_energy_distortion'] for v in eligible]),
                         'spectral_distortion':distribution([v['spectral_magnitude_L1'] for v in eligible]),'boundary_jump_uV':distribution([v['boundary_jump_uV'] for v in eligible]),
                         'donor_conditional_status':'TESTED' if valid else 'NOT_TESTED_NO_COMPATIBLE_INTERVENTIONS'}
    comps={}
    for name in primary[0]['comparators']:
        ranking=[r['comparators'][name]['ranking'] for r in primary];random=[r['comparators'][name]['random_matched_size'] for r in primary]
        comps[name]={'ranking_canonical_success':sum(x['canonical_response'] for x in ranking),'ranking_universal_valid':sum(x['universal_positive_valid'] for x in ranking),
                     'random_canonical_success':sum(x['canonical_response'] for x in random),'random_universal_valid':sum(x['universal_positive_valid'] for x in random),
                     'retained_seconds':distribution([x['retained_seconds'] for x in ranking]),'standalone_canonical_head_queries':sum(x['canonical_queries'] for x in ranking),
                     'additional_gradient_head_forwards':150,'access':'original feature gradients and fixed pooled-window projection; shared exhaustive truth table'}
    comps['greedy']={'canonical_success':sum(r['summary']['greedy']['supported'] for r in primary),
                     'universal_valid':sum(r['summary']['greedy'].get('universal_positive_valid',False) for r in primary),
                     'standalone_head_queries':sum(r['summary']['greedy']['local_queries'] for r in primary),
                     'retained_seconds':distribution([r['summary']['greedy']['retained_seconds'] for r in primary if r['summary']['greedy']['supported']])}
    comps['external_timeseries_implementation']={'status':'NOT_TESTED','reason':'No externally implemented baseline validated against this internal operator and bounded query/access contract; no superiority claim'}
    triad={'n':150,'stability':stability,'probes':probes,'operators':operators,'comparators':comps,
           'neutral_table_disagreement_fraction':distribution([r['triad']['neutral_table_disagreement_fraction'] for r in primary]),
           'neutral_positive_minimum_changed':sum(r['summary']['positive_minimum_mask']!=r['neutral_summary']['positive_minimum_mask'] for r in primary),
           'neutral_canonical_minimum_changed':sum(r['summary']['canonical_minimum_mask']!=r['neutral_summary']['canonical_minimum_mask'] for r in primary),
           'compatible_operator_comparisons':sum(r['triad']['compatible_comparisons'] for r in primary),
           'operator_disagreements':sum(r['triad']['operator_disagreements'] for r in primary),
           'examples_with_operator_disagreement':sum(r['triad']['operator_disagreements']>0 for r in primary),
           'clinical_event_localization':'NOT_TESTED_NO_INDEPENDENT_EVENT_LABELS'}
    write_new(PUBLIC/'triad_metrics.json',triad)
    examples={}
    predicates={'wrong_valid_compact':lambda r:not r['correct'] and compact(r['summary']),
                'neutral_sensitive':lambda r:r['triad']['neutral_table_disagreement_fraction']>0,
                'operator_disagreement':lambda r:r['triad']['operator_disagreements']>0,
                'correct_unstable':lambda r:r['correct'] and any(v['fixed_clause_valid'] is False for v in r['triad']['stability'].values()),
                'global_only_stable_local_changed':lambda r:r['feature_ablations']['global_only']['prediction']==r['native']['prediction'] and r['feature_ablations']['local_only']['prediction']!=r['native']['prediction']}
    counts={}
    for name,predicate in predicates.items():
        selected=[r for r in primary if predicate(r)];counts[name]=len(selected);examples[name]=selected[0] if selected else None
    write_new(PRIVATE/'deterministic_examples.json',examples);write_new(PUBLIC/'failure_mechanisms.json',{'n':150,'overlapping_counts':counts,'selection':'first canonical example per category, private only','unresolved':['Clinical ambiguity versus annotation noise','True physiological events versus nuisance','Causal global shortcut versus valid stage information']})
    print(json.dumps({'primary_accuracy':result['primary']['classification']['accuracy'],'primary_compact_active':result['primary']['rates']['compact_nonempty_active_summary']['count'],
                      'primary_whole_epoch_positive':result['primary']['rates']['whole_epoch_positive_minimum']['count'],
                      'primary_global_only_stable':result['primary']['rates']['global_only_prediction_stable']['count'],
                      'probe_accuracy':{k:v['metrics']['accuracy'] for k,v in probes.items()},'stability':stability,'operators':operators,'mechanisms':counts}))

if __name__=='__main__':run()

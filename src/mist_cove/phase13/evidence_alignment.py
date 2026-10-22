"""Locked-label alignment entrypoint, with no fabricated fallback labels."""
from .annotation_metrics import require_locked_expert_labels,overlap_metrics,controls,mask_intervals


def evaluate_locked(exports,support_mask,events,seed=123):
    require_locked_expert_labels(exports)
    comparison={'exact_positive':support_mask,**{k:v for k,v in controls(support_mask,seed).items() if k.endswith('_mask')}}
    return {method:{str(t):overlap_metrics(mask_intervals(mask),events,t) for t in (0,.1,.25)}
            for method,mask in comparison.items()}


def candidate_regions(wave,support_mask,greedy_mask,gradient_ranking,seed=123):
    import numpy as np
    x=np.asarray(wave).reshape(15,200);k=int(support_mask).bit_count()
    if sorted(gradient_ranking)!=list(range(15)):raise ValueError('complete physical ranking required')
    rms=np.std(x,axis=1);power=abs(np.fft.rfft(x-x.mean(1,keepdims=True),axis=1))**2;freq=np.fft.rfftfreq(200,.01)
    sigma=power[:,(freq>=11)&(freq<=16)].sum(1)/power.sum(1).clip(1e-12)
    def rank_mask(rank):return sum(1<<int(i) for i in rank[:k])
    return {'exact_positive':int(support_mask),'greedy_own_budget':int(greedy_mask),
            'RMS_matched_budget':rank_mask(np.argsort(-rms,kind='stable')),
            'sigma_fraction_matched_budget':rank_mask(np.argsort(-sigma,kind='stable')),
            'gradient_matched_budget':rank_mask(gradient_ranking),**controls(support_mask,seed),
            'sigma_is_clinical_event_label':False,'greedy_budget_matches':int(greedy_mask).bit_count()==k}


def align_locked_panel(exports,model_cases,case_subjects,primary_ids):
    """Analyze genuine locked exports locally; never derive events from stages.

    Coordinator supplies only primary opaque IDs and their private grouping.
    This function returns detailed private results for later aggregate release.
    """
    import numpy as np
    from .annotation_schema import CATEGORIES
    from .annotation_metrics import review_quality,interval_mask
    from mist_cove.phase11b.evaluation import clustered_mean
    require_locked_expert_labels(exports)
    primary_ids=list(primary_ids)
    if not primary_ids or set(primary_ids)-set(model_cases) or set(primary_ids)-set(case_subjects):
        raise ValueError('frozen primary mapping incomplete')
    result={'reviewers':[],'clinical_certification':False,'category_analysis':'exploratory multiple comparisons',
            'quality':review_quality(exports[0],exports[1],primary_ids)}
    for export in exports:
        rows=[];paired_differences=[];units=[]
        for case in primary_ids:
            model=model_cases[case];candidates=model.get('alignment_candidates')
            if candidates is None:continue
            assessment=export['assessments'].get(case)
            if assessment is None:continue
            for event in assessment['assessments']:
                if event['state'] not in ('present','absent'):continue
                events=event['intervals'];measures={}
                for method,mask in candidates.items():
                    if method.endswith('_mask') or method in ('exact_positive','greedy_own_budget','RMS_matched_budget','sigma_fraction_matched_budget','gradient_matched_budget'):
                        measures[method]={str(t):overlap_metrics(mask_intervals(mask),events,t) for t in (0,.1,.25)}
                if candidates['exact_positive'] and events:
                    a=measures['exact_positive']['0']['precision'];b=measures['length_matched_mask']['0']['precision']
                    paired_differences.append(a-b);units.append(case_subjects[case])
                rows.append({'case_id':case,'category':event['category'],'state':event['state'],'confidence':event['confidence'],
                    'correct':model['correct'],'methods':measures,
                    'greedy_budget_matches':candidates['greedy_budget_matches'],
                    'counterfactual_signed_clauses':model['summary']['signed']['clauses'],
                    'global_branch_predictions':model['branch_predictions']})
        result['reviewers'].append({'reviewer':export['reviewer'],'rows':rows,
            'precision_minus_length_control':clustered_mean(paired_differences,units),
            'primary_completion':len(set(primary_ids)&set(export['assessments']))/len(primary_ids)})
    # Paired-event intersections are an explicitly separate conservative analysis.
    intersections=[]
    for case in primary_ids:
        aa={v['category']:v for v in exports[0]['assessments'].get(case,{}).get('assessments',[])}
        bb={v['category']:v for v in exports[1]['assessments'].get(case,{}).get('assessments',[])}
        for category in set(aa)&set(bb):
            if aa[category]['state']=='present' and bb[category]['state']=='present':
                overlap=interval_mask(aa[category]['intervals'])&interval_mask(bb[category]['intervals'])
                intersections.append({'case_id':case,'category':category,'intersection_seconds':float(overlap.sum()/100)})
    result['paired_event_intersections']=intersections
    result['automatic_adjudication']=False
    return result

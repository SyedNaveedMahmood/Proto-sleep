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

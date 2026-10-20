"""Universally verified signed subcubes; bounded summaries are incomplete."""
import numpy as np


def matching_indices(n,positive,negative):
    limit=(1<<n)-1
    if positive&negative or positive<0 or negative<0 or (positive|negative)>limit:
        raise ValueError('disjoint, in-range signed literals required')
    ids=np.arange(1<<n);return (ids[(ids&positive)==positive] if negative==0 else ids[((ids&positive)==positive)&((ids&negative)==0)])


def verify_clause(truth_table,positive,negative):
    table=np.asarray(truth_table)
    if table.dtype!=bool or table.ndim not in (1,2):raise ValueError('Boolean finite-view truth required')
    size=table.shape[-1]
    if not size or size&(size-1):raise ValueError('complete truth table required')
    indices=matching_indices(size.bit_length()-1,positive,negative)
    return bool(table[...,indices].all())


def minimize_clause(truth,positive,negative):
    if not verify_clause(truth,positive,negative):raise ValueError('invalid initial clause')
    n=np.asarray(truth).shape[-1].bit_length()-1
    # Removal enlarges completions; a once-invalid removal stays invalid.
    for j in range(n):
        bit=1<<j
        if positive&bit and verify_clause(truth,positive^bit,negative):positive^=bit
        elif negative&bit and verify_clause(truth,positive,negative^bit):negative^=bit
    for j in range(n):
        bit=1<<j
        if positive&bit and verify_clause(truth,positive^bit,negative):raise ValueError('positive literal not subset-minimal')
        if negative&bit and verify_clause(truth,positive,negative^bit):raise ValueError('negative literal not subset-minimal')
    return positive,negative


def extract_readable_clauses(truth_table,*,max_display=4,positive_only=False,margins=None):
    table=np.asarray(truth_table);truth=table.all(0) if table.ndim==2 else table
    n=len(truth).bit_length()-1;limit=(1<<n)-1
    if truth.dtype!=bool:raise ValueError('Boolean truth required')
    prior=bool(truth.all());covered=np.zeros(len(truth),dtype=bool);clauses=[]
    if not truth.any():return {'clauses':[],'coverage':0.,'status':'NO_SUPPORT','prior_only':False,'complete_prime_enumeration':False}
    # Exact full-state reason first; later anchors favor short retained support.
    anchors=[limit]+sorted(np.flatnonzero(truth).tolist(),key=lambda i:(int(i).bit_count(),i))[:128]
    for anchor in anchors:
        if len(clauses)>=max_display:break
        if not truth[anchor] or covered[anchor]:continue
        positive=anchor;negative=0 if positive_only else limit^anchor
        if not verify_clause(table,positive,negative):continue
        p,q=minimize_clause(table,positive,negative);indices=matching_indices(n,p,q);covered[indices]=True
        clauses.append({'positive':[j for j in range(n) if p&(1<<j)],'negative':[j for j in range(n) if q&(1<<j)],
            'positive_bits':p,'negative_bits':q,'retained_seconds':p.bit_count()*2,'signed_literals':(p|q).bit_count(),
            'minimum_margin':float(np.asarray(margins)[...,indices].min()) if margins is not None else None,
            'subset_minimal':True,'minimum_cardinality':False,'active_at_original_full_mask':q==0,
            'counterfactual_state_only':q!=0,'waveform_support_claim':not prior})
    coverage=float((covered&truth).sum()/truth.sum())
    return {'clauses':clauses,'coverage':coverage,'status':'PRIOR_ONLY' if prior else ('PARTIAL_SUMMARY' if coverage<1 else 'COVERED'),
        'prior_only':prior,'complete_prime_enumeration':False,'anchor_policy':'full then first128 true by popcount/index; at most4 clauses'}


def greedy_sufficient_mask(truth):
    """SIS-inspired local reimplementation: canonical-state checks only."""
    truth=np.asarray(truth,dtype=bool);mask=len(truth)-1;n=mask.bit_length();queries=0
    if not truth[mask]:return {'mask':None,'supported':False,'local_queries':0}
    changed=True
    while changed:
        changed=False
        for j in range(n):
            if mask&(1<<j):
                queries+=1
                if truth[mask^(1<<j)]:mask^=1<<j;changed=True
    return {'mask':mask,'supported':True,'local_queries':queries,'retained_seconds':mask.bit_count()*2,
        'universal_positive_valid':verify_clause(truth,mask,0),'global_minimum':False}

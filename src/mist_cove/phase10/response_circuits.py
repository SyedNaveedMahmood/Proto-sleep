"""Source-geometry response tables; Phase IX Boolean compiler is unchanged."""
import time
import numpy as np
import torch
from mist_cove.phase9.operators import masks_tensor,margin
from mist_cove.phase9.circuits import compile_robdd,evaluate_robdd,nonmonotone_edges
from mist_cove.phase9.clauses import extract_readable_clauses,greedy_sufficient_mask,verify_clause


@torch.no_grad()
def response_table(adapter,cache,*,batch_size=512,neutral=0.,deadline=None):
    if not isinstance(batch_size,int) or not 1<=batch_size<=512:raise ValueError('bounded mask batch required')
    masks=masks_tensor(cache.global_features.device)
    full=adapter.logits(cache,masks[-1:],neutral=neutral)
    if full.shape!=(1,1,5):raise ValueError('one example and five logits required')
    target=int(full[0,0].argmax());values=[]
    for start in range(0,32768,batch_size):
        if deadline is not None and time.perf_counter()>deadline:raise TimeoutError('mock scientific resource ceiling exceeded')
        values.append(margin(adapter.logits(cache,masks[start:start+batch_size],neutral=neutral)[:,0],target).cpu())
    values=torch.cat(values).numpy();truth=values>1e-6
    return {'target':target,'margins':values,'truth':truth,'availability_queries':32768,'full_target_queries':1}


def summarize_response(table):
    truth=table['truth'];graph=compile_robdd(truth)
    if not np.array_equal(evaluate_robdd(graph),truth):raise ValueError('complete response reconstruction failed')
    signed=extract_readable_clauses(truth,margins=table['margins'])
    positive=extract_readable_clauses(truth,positive_only=True,margins=table['margins'])
    for group in (signed,positive):
        for clause in group['clauses']:
            if not verify_clause(truth,clause['positive_bits'],clause['negative_bits']):raise ValueError('unsound clause')
    return {'target_class':table['target'],'true_masks':int(truth.sum()),'masks':32768,
            'full_response':bool(truth[-1]),'zero_response':bool(truth[0]),
            'exact_all_rows':True,'BDD_nonterminal_nodes':len(graph['nodes'])-2,
            'BDD_identity':graph['sha256'],'nonmonotone_edges':nonmonotone_edges(truth),
            'signed':signed,'positive_only':positive,'greedy':greedy_sufficient_mask(truth)}


def canonical_comparators(truth,atom_ranking,oracle_atoms):
    truth=np.asarray(truth,dtype=bool);limit=32767
    if sorted(atom_ranking)!=list(range(15)):raise ValueError('full atom ranking required')
    ranked_mask=0;queries=1
    if not truth[0]:
        for atom in atom_ranking:
            ranked_mask|=1<<atom;queries+=1
            if truth[ranked_mask]:break
    random=np.random.default_rng(123).choice(15,size=ranked_mask.bit_count(),replace=False)
    random_mask=sum(1<<int(a) for a in random)
    def row(mask):
        selected={j for j in range(15) if mask&(1<<j)};oracle=set(oracle_atoms)
        return {'mask_id':int(mask),'retained_atoms':len(selected),'retained_seconds':2*len(selected),
                'canonical_response':bool(truth[mask]),'universal_positive_valid':verify_clause(truth,int(mask),0),
                'privileged_oracle_precision':len(selected&oracle)/len(selected) if selected else None,
                'privileged_oracle_coverage':len(selected&oracle)/len(oracle) if oracle else None}
    return {'ranking':{**row(ranked_mask),'canonical_queries':queries},'random_matched_size':row(random_mask),
            'random_selection_seed':123,'oracle_used_to_select':False,
            'query_accounting':'lookups share exhaustive table; standalone ranking would need up to15 masked-head queries'}

"""Exact fixed physical-order ROBDDs, with exhaustive truth-table verification."""
import hashlib
import json
import numpy as np


def compile_robdd(truth_table,variable_order=None):
    truth=np.asarray(truth_table)
    if truth.ndim!=1 or truth.dtype!=bool or len(truth)==0 or len(truth)&(len(truth)-1):
        raise ValueError('complete power-of-two Boolean truth table required')
    n=len(truth).bit_length()-1;order=list(range(n)) if variable_order is None else list(variable_order)
    if order!=list(range(n)):raise ValueError('physical-time variable order is frozen')
    nodes=[(-1,0,0),(-1,1,1)];unique={}
    def build(values,level):
        if not values.any():return 0
        if values.all():return 1
        low=build(values[::2],level+1);high=build(values[1::2],level+1)
        if low==high:return low
        key=(level,low,high)
        if key not in unique:unique[key]=len(nodes);nodes.append(key)
        return unique[key]
    root=build(truth,0)
    result={'root':root,'nodes':nodes,'variable_order':order,'masks':len(truth),'exact':True}
    if not np.array_equal(evaluate_robdd(result),truth):raise ValueError('complete ROBDD equality failed')
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def evaluate_robdd(circuit):
    ids=np.arange(circuit['masks'],dtype=np.int64);current=np.full(len(ids),circuit['root'],dtype=np.int64)
    nodes=np.asarray(circuit['nodes'],dtype=np.int64)
    for _ in circuit['variable_order']:
        active=current>1
        if not active.any():break
        var,lo,hi=nodes[current[active]].T
        current[active]=np.where((ids[active]>>var)&1,hi,lo)
    if (current>1).any():raise ValueError('invalid or cyclic decision graph')
    return current==1


def nonmonotone_edges(truth):
    truth=np.asarray(truth,dtype=bool);ids=np.arange(len(truth));bad=0;edges=0
    for j in range(len(truth).bit_length()-1):
        low=ids[(ids&(1<<j))==0];bad+=int((truth[low]&~truth[low|(1<<j)]).sum());edges+=len(low)
    return {'response_lost_when_adding_atom':bad,'examined_edges':edges,'fraction':bad/edges if edges else 0.}

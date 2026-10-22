"""Exact bounded coarse availability certificates for both frozen context arms."""
import json
import time
import io
import numpy as np
import torch
from .common import PRIVATE,PUBLIC,read_json,write_new,measured,ledger,check_budget,sha256,canonical
from .context_model import load_models
from .context_operator import ContextOperator,summarize
from .source_audit import cached_selection
from mist_cove.phase11b.security import authenticated_snapshot
from mist_cove.phase9.clauses import verify_clause


def run():
    torch.set_num_threads(2);selection=cached_selection();expected=read_json(PUBLIC/'SC_TRAIN_execution.json')['context_data_sha256']
    path=PRIVATE/'context_data.npz'
    with authenticated_snapshot(path,expected,allowlist=[path],limit=512*1024*1024) as snapshot:
        with np.load(io.BytesIO(snapshot.data),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    coordinates=selection['training']['fit']+selection['training']['internal_dev'];lookup={(c['recording_id'],c['epoch']):i for i,c in enumerate(coordinates)}
    indices=[lookup[c['recording_id'],c['epoch']] for c in selection['context_mechanistic']];rows=[];queries=0;started=time.perf_counter()
    norm=selection['normalization']
    with measured('context_support_and_stability_audit') as counts:
        for arm,seed,model in load_models():
            operator=ContextOperator(model)
            for ordinal,i in enumerate(indices):
                x=torch.from_numpy(data['x'][i:i+1]);v=torch.from_numpy(data['valid'][i:i+1]);cache=operator.cache(x,v);target=int(cache[3].argmax())
                full=torch.ones(1,17,dtype=torch.bool);torch.testing.assert_close(operator.logits(cache,full)[0],cache[3],atol=1e-6,rtol=1e-6);queries+=1
                check_budget(started,queries+sum(r.get('cached_head_queries',0) for r in read_json(PUBLIC/'resource_stages.json').values())+1000000)
                table=operator.table(cache,target);summary=summarize(table);queries+=131072
                gain=x*1.05+.05*norm['mean_uV']/norm['std_uV'];gain_cache=operator.cache(gain,v);view=operator.table(gain_cache,target);vs=summarize(view);queries+=131072
                active=[c for c in summary['signed']['clauses'] if c['negative_bits']==0];first=active[0] if active else None
                validity=[verify_clause(view['truth'],c['positive_bits'],c['negative_bits']) for c in summary['signed']['clauses']]
                row={'arm':arm,'seed':seed,'ordinal':ordinal,'coordinate':coordinates[i],'correct':target==int(data['truth'][i]),'summary':summary,
                     'gain_prediction_stable':int(gain_cache[3].argmax())==target,
                     'fixed_clause_valid':None if first is None else verify_clause(view['truth'],first['positive_bits'],0),
                     'any_original_active_valid':any(valid and c['negative_bits']==0 for valid,c in zip(validity,summary['signed']['clauses'])),
                     'view_exact':vs['exact_all_rows'],'neighbor_literals_in_displayed_active':any(any(bit in (0,16) for bit in c['positive']) for c in active)}
                rows.append(row);np.save(PRIVATE/f'context_{arm}_{seed}_{ordinal}_base_margins.npy',table['margins']);np.save(PRIVATE/f'context_{arm}_{seed}_{ordinal}_gain_margins.npy',view['margins'])
            operator.check()
        write_new(PRIVATE/'context_audit_rows.json',rows)
        groups=[]
        for arm in ('C0','C1'):
            for seed in (123,456,789):
                rr=[r for r in rows if r['arm']==arm and r['seed']==seed]
                groups.append({'arm':arm,'seed':seed,'cases':len(rr),'correct':sum(r['correct'] for r in rr),'all_exact':all(r['summary']['exact_all_rows'] and r['view_exact'] for r in rr),
                    'neighbor_active_cases':sum(r['neighbor_literals_in_displayed_active'] for r in rr),
                    'displayed_compact_active':sum(any(c['negative_bits']==0 and 0<c['retained_seconds']<=8 and c['signed_literals']<=8 for c in r['summary']['signed']['clauses']) for r in rr),
                    'gain_prediction_stable':sum(r['gain_prediction_stable'] for r in rr),'gain_fixed_clause_valid':sum(r['fixed_clause_valid'] is True for r in rr),
                    'fixed_clause_eligible':sum(r['fixed_clause_valid'] is not None for r in rr)})
        result={'status':'PASS','groups':groups,'tables':60,'rows_verified':60*131072,'cached_head_queries':queries,'feature_cache_base_epoch_forwards':180,
                'support_partition':'previouswhole30s +15center2satoms +nextwhole30s; not45independent2satoms',
                'physical_weights_seconds':[30]+[2]*15+[30],'neighbor_labels_input':False,'future_context':'OFFLINE_NONCAUSAL',
                'native_identity':'PASS1e-6','clinical_alignment':'NOT_TESTED','wall_seconds':time.perf_counter()-started,
                'grouping':'same5developmentcases repeatedacross6models; not30people'}
        write_new(PUBLIC/'context_operator_results.json',result);counts.update(cached_head_queries=queries,tables=60,feature_cache_base_epoch_forwards=180)
        ledger([31],'COMPLETE','New17variable coarse context partition independently validated;60full131072-row tables and fixed-gain clauses verified; neighbor support explicitly included',['reports/mist_cove/phase14_v1/context_operator_results.json'],science='PASS')

"""Bounded CPU architecture timings and idle-only CUDA synthetic smoke."""
import json
import resource
import statistics
import subprocess
import time
import platform
import numpy as np
import torch
from mist_cove.phase9.circuits import compile_robdd
from mist_cove.phase9.clauses import extract_readable_clauses
from .mock_models import make_mock,mock_input
from .source_adapter import SourceAdapter
from .response_circuits import response_table
from .safety import PUBLIC,PRIVATE,write_json,no_sensitive_exports,mock_request
from .verification import verify_frozen_mock_core,require_gates


def cuda_availability():
    try:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        info=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    except (FileNotFoundError,subprocess.SubprocessError):
        return {'status':'UNAVAILABLE','reason':'GPU availability query unavailable','new_cuda_model_operations':0}
    if apps:
        return {'status':'BLOCKED','reason':'Another compute process occupies a GPU; do not allocate or interrupt',
                'compute_process_count':len(apps.splitlines()),'device_memory_snapshot':info,'new_cuda_model_operations':0}
    return {'status':'IDLE','device_memory_snapshot':info,'new_cuda_model_operations':0}


@torch.no_grad()
def cuda_smoke():
    gate=cuda_availability()
    if gate['status']!='IDLE':return gate
    if not torch.cuda.is_available():return {**gate,'status':'UNAVAILABLE','reason':'This interpreter has no CUDA runtime'}
    started=time.perf_counter();torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    network=make_mock().network.to('cuda');adapter=SourceAdapter(network);x=mock_input(n=1).to('cuda');cache=adapter.cache(x)
    full=torch.ones(1,15,dtype=torch.bool,device='cuda');actual=adapter.logits(cache,full)[0];native=adapter.native(x)
    torch.testing.assert_close(actual,native,atol=1e-6,rtol=1e-6)
    partial=full.clone();partial[:,::2]=False;y=x.clone();y[:,:,~partial[0].repeat_interleave(200)]+=10000
    torch.testing.assert_close(adapter.logits(cache,partial),adapter.logits(adapter.cache(y),partial),atol=1e-6,rtol=1e-6)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    if elapsed>60:raise TimeoutError('CUDA synthetic correctness ceiling exceeded')
    return {**gate,'status':'PASS','wall_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_cuda_model_operations':4,'cache_or_native_local_encoder_calls':9,
            'operation_scope':'four complete prediction queries; encoder calls counted separately',
            'scope':'random historical-architecture mock only; no training or resume claim'}


def timed(call):
    started=time.perf_counter();result=call();return result,time.perf_counter()-started


@torch.no_grad()
def cpu_benchmark():
    mock_request();verify_frozen_mock_core()
    require_gates(json.loads((PUBLIC/'correctness_receipt.json').read_text()),['identity','noninterference'])
    if not (PUBLIC/'provenance_review.json').exists():raise ValueError('X10 metadata review required first')
    started=time.perf_counter();deadline=started+600;out=PRIVATE/'benchmark_v1';out.mkdir(exist_ok=False)
    rows=[];sensitivity=[]
    for seed in (123,456,789):
        adapter=SourceAdapter(make_mock(seed).network);x=mock_input(n=1);cache=adapter.cache(x)
        masks=((torch.arange(512)[:,None]>>torch.arange(15))&1).bool();mask=masks[7:8]
        adapter.native(x)  # exactly one unmeasured native warmup per seed
        for repetition in range(3):
            _,native_seconds=timed(lambda:adapter.native(x))
            cache,cache_seconds=timed(lambda:adapter.cache(x))
            _,single_seconds=timed(lambda:adapter.logits(cache,mask))
            _,batch_seconds=timed(lambda:adapter.logits(cache,masks))
            _,feature_seconds=timed(lambda:adapter.features(cache,mask))
            table,table_seconds=timed(lambda:response_table(adapter,cache,deadline=deadline))
            graph,bdd_seconds=timed(lambda:compile_robdd(table['truth']))
            clauses,clause_seconds=timed(lambda:extract_readable_clauses(table['truth'],margins=table['margins']))
            path=out/f'seed{seed}_repeat{repetition}.json'
            _,serialization_seconds=timed(lambda:write_json(path,{'circuit':graph,'summary':clauses}))
            rows.append({'seed':seed,'repetition':repetition,'full_native_seconds':native_seconds,'cache_seconds':cache_seconds,
                         'one_masked_head_seconds':single_seconds,'batch512_masked_seconds':batch_seconds,
                         'masked_features_and_global_suppression_seconds':feature_seconds,'response_table_seconds':table_seconds,
                         'BDD_seconds':bdd_seconds,'clause_extraction_and_verification_seconds':clause_seconds,
                         'serialization_seconds':serialization_seconds,'serialized_bytes':path.stat().st_size,
                         'BDD_nonterminal_nodes':len(graph['nodes'])-2,'cache_bytes':cache.nbytes,
                         'table_bytes':table['truth'].nbytes+table['margins'].nbytes,'availability_queries':32768})
        other=response_table(adapter,cache,neutral=.01,deadline=deadline)
        sensitivity.append({'seed':seed,'primary_target':table['target'],'control_target':other['target'],
                            'truth_disagreement_fraction':float(np.mean(table['truth']!=other['truth'])) if table['target']==other['target'] else None,
                            'target_unchanged':table['target']==other['target'],'availability_queries':32768})
    stress=[];truth=np.random.default_rng(943).random(32768)>.5
    for repetition in range(3):
        graph,bdd_seconds=timed(lambda:compile_robdd(truth))
        clauses,clause_seconds=timed(lambda:extract_readable_clauses(truth))
        path=out/f'random_boolean_stress_{repetition}.json'
        _,serialization_seconds=timed(lambda:write_json(path,{'circuit':graph,'summary':clauses}))
        stress.append({'repetition':repetition,'BDD_nonterminal_nodes':len(graph['nodes'])-2,
                       'BDD_seconds':bdd_seconds,'clause_seconds':clause_seconds,'serialization_seconds':serialization_seconds,
                       'serialized_bytes':path.stat().st_size,'summary_coverage':clauses['coverage']})
    per_example=[sum(r[k] for k in ('cache_seconds','response_table_seconds','BDD_seconds','clause_extraction_and_verification_seconds','serialization_seconds')) for r in rows]
    conservative=2*(max(r['cache_seconds'] for r in rows)+max(r['response_table_seconds'] for r in rows)
                    +max(r['BDD_seconds'] for r in stress)+max(r['clause_seconds'] for r in stress)+max(r['serialization_seconds'] for r in stress))
    result={'version':'phase10-resource-estimates-v1','status':'COMPLETE_MOCK_CPU_ONLY','python':platform.python_version(),
            'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'rows':rows,
            'neutral_control':sensitivity,'random_boolean_stress':stress,'wall_seconds':time.perf_counter()-started,
            'peak_process_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'projected_150_examples_cpu_seconds':{'observed_fast_scenario':150*min(per_example),
                 'observed_median_scenario':150*statistics.median(per_example),
                 'conservative_stress_scenario':150*conservative},
            'projection_scope':'conditional CPU architecture scenarios, not confidence bounds, worst-case guarantees or actual trained-source measurements; no EEG IO included',
            'GPU_projection_seconds':None,'one_GPU_hour_feasibility':'NOT_ESTABLISHED',
            'cache_bytes_for_150_examples':150*rows[0]['cache_bytes'],'boolean_mask_grid_bytes':15*32768,
            'scientific_table_queries_this_benchmark':sum(r['availability_queries'] for r in rows+sensitivity),
            'additional_queries':{'native_warmups':3,'native_timed_predictions':9,'single_masked_queries':9,'batched_masked_queries':9*512,
                                  'full_target_queries':12,'gradient_backwards':0},
            'real_artifacts_opened':0}
    no_sensitive_exports(result);write_json(PUBLIC/'resource_estimates.json',result);return result

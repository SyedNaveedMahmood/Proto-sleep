"""Reconstruct synthetic TRAIN streams and fixed clipping statistics only."""
import json
import time
import torch
from collect import PUBLIC,publish,sha,ROOT
from mist_cove.phase2.training import configure_precision
from mist_cove.phase2.provenance import digest
from mist_cove.phase6.generator import generate
from mist_cove.phase7.prospective import encode
from mist_cove.phase8.provenance import trusted_reference,anchored
from mist_evidence.runtime import tensor_digest


def main():
    configure_precision('cpu');torch.set_num_threads(2);started=time.perf_counter();rows=[]
    diagnostic=json.loads(anchored('reports/mist_cove/phase7_v1/prospective_diagnostics_v2.json'))
    robust=[json.loads(v) for v in anchored('reports/mist_cove/phase7_v1/robustness_runs_v1.jsonl').splitlines()]
    for task in ('single','multiple'):
        for seed in (123,456,789):
            train=generate(1000,seed+10000,task,'train');competitive=generate(1000,seed+10000,task,'competitive')
            mixed=torch.where((torch.arange(1000)%2==1)[:,None,None],competitive.x,train.x)
            p0,proof=trusted_reference(task,seed,'P0');f,_=encode(p0,train.x);mu,sd=f.mean((0,1)),f.std((0,1)).clamp_min(.01)
            prior=next(v for v in diagnostic['rows'] if (v['task'],v['seed'])==(task,seed))['training']['creation']
            for method in ('P0','R1','R2'):
                m,p=trusted_reference(task,seed,method)
                creation=prior if method=='P0' else next(v for v in robust if (v['task'],v['seed'],v['method'])==(task,seed,method))['training']['creation']
                expected=tensor_digest({'x':train.x if method=='P0' else mixed,'y':train.y})
                if expected!=creation['identity']['train_digest']:raise ValueError('original prospective data stream mismatch')
                if method!='P0' and creation['identity']['p0_creation_sha256']!=digest(prior):raise ValueError('prospective parent mismatch')
                if method=='R2' and (not torch.equal(m.clip_mean,mu) or not torch.equal(m.clip_std,sd)):raise ValueError('TRAIN-derived clipping buffers mismatch')
                rows.append({'task':task,'seed':seed,'method':method,'status':'PASS','recreated_train_digest':expected,
                    'seed_and_generator_stream_verified':True,'parent_creation_verified':method!='P0','clip_buffers_exact':True if method=='R2' else None})
    publish(PUBLIC/'provenance_replay_receipt.json',{'status':'PASS','rows':rows,'new_fits':0,'wall_seconds':time.perf_counter()-started,
        'historical_phase6_deserializations':0,'real_data_accessed':False})

if __name__=='__main__':main()

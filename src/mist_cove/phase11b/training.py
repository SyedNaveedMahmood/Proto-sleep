"""Exactly nine prospective CPU fits; no sweep, early stopping, or VAL fitting."""
import json
import time
import hashlib
import resource
import numpy as np
import torch
import torch.nn.functional as F
from .campaign import PUBLIC,PRIVATE,setup,write_new
from .source import select_epochs
from .calibration import verified_partition
from .historical import canonical
from .security import sha256
from .development import ReliabilityLedger,local_hf,preservation_loss
from .evaluation import public_metrics,distribution,clustered_mean

def prepare():
    start=time.perf_counter();allocation=verified_partition();_,access=setup();records=access.records('train')
    fit_records=[r for r in records if r['subject_id'] in allocation['fit']];dev_records=[r for r in records if r['subject_id'] in allocation['internal_dev']]
    fit,missing_fit=select_epochs(fit_records,100);dev,missing_dev=select_epochs(dev_records,40)
    coordinates={'fit':fit,'internal_dev':dev,'missing_fit':missing_fit,'missing_dev':missing_dev}
    write_new(PRIVATE/'new_model_selection.json',coordinates)
    data=np.load(PRIVATE/'train_predictions.npz',allow_pickle=False);selected=np.isin(data['subjects'],allocation['fit']);q=data['quality'][selected]
    mean=float(q[:,2].mean());variance=float(np.mean(q[:,1]**2+q[:,2]**2)-mean**2);std=float(np.sqrt(variance))
    norm={'mean_uV':mean,'std_uV':std,'fit_physical_epochs':int(selected.sum()),'fit_participants':10,'fit_role':'train_fit','method':'all physical fit-group moments, including unscored epochs; no labels used'}
    write_new(PRIVATE/'new_model_normalization.json',norm)
    samples=fit+dev;wave=np.empty((len(samples),1,3000),np.float32)
    for record in records:
        indices=[i for i,c in enumerate(samples) if c['recording_id']==record['recording_id']]
        if not indices:continue
        with access.reader(record) as reader:
            for start_i in range(0,len(indices),256):
                group=indices[start_i:start_i+256];_,w=access.read(record,reader,[samples[i]['epoch'] for i in group]);wave[group]=w
    normalized=(wave-mean)/std;labels=np.array([c['stage'] for c in samples]);subjects=np.array([c['subject'] for c in samples])
    np.savez_compressed(PRIVATE/'new_model_data.npz',x=normalized,y=labels,subjects=subjects,n_fit=len(fit))
    write_new(PUBLIC/'new_model_data_receipt.json',{'selection_sha256':sha256(canonical(coordinates)),'normalization_sha256':sha256(canonical(norm)),
        'fit_epochs':len(fit),'internal_dev_epochs':len(dev),'fit_normalization_physical_epochs':int(selected.sum()),'normalization_excludes_internal_dev':True,
        'missing_fit_strata_epochs':sum(v['missing'] for v in missing_fit),'missing_dev_strata_epochs':sum(v['missing'] for v in missing_dev),
        'source_access':access.counts,'wall_seconds':time.perf_counter()-start,'selection_policy':'first100/40 scored epochs per stage per fit/internal-dev participant; stage-balanced, not prevalence sample'})

def training_stream(seed,n,updates=300,batch_size=16):
    generator=torch.Generator().manual_seed(seed+10000)
    for _ in range(updates):
        yield (torch.randint(n,(batch_size,),generator=generator),torch.randint(15,(batch_size,),generator=generator),
               2*torch.pi*torch.rand(batch_size,generator=generator))

def evaluation_views(x):
    n=len(x);atoms=torch.arange(n)%15;rms=x.std(-1,unbiased=False).clamp_min(1e-8);t=torch.arange(3000,dtype=x.dtype)/100
    return {'clean':x,'local_hf':local_hf(x,atoms),'strong_hf':x+.5*rms[:,:,None]*torch.sin(2*torch.pi*35*t)[None,None],
            'mild_hf':x+.05*rms[:,:,None]*torch.sin(2*torch.pi*32*t)[None,None]}

@torch.no_grad()
def evaluate(model,x,y,subjects):
    model.eval();views=evaluation_views(x);results={};private={};clean_parts=None
    for name,values in views.items():
        batches=[];parts=[];routing=[];global_logits=[]
        for start in range(0,len(x),64):
            output=model(values[start:start+64]);batches.append(output['logits']);parts.append(output['contributions']);routing.append(output['reliability']);global_logits.append(output['global'])
        z=torch.cat(batches);c=torch.cat(parts);r=torch.cat(routing);g=torch.cat(global_logits)
        # Exact arithmetic accounting at original FP32 evaluation precision.
        torch.testing.assert_close(z,model.bias+g+c.sum(1),atol=0,rtol=0)
        if name=='clean':clean_parts=c
        target=z.argmax(1);runner=z.clone();runner[torch.arange(len(z)),target]=-torch.inf;other=runner.argmax(1)
        signed=c[torch.arange(len(c))[:,None],torch.arange(15)[None],target[:,None]]-c[torch.arange(len(c))[:,None],torch.arange(15)[None],other[:,None]]
        changed_region=torch.arange(len(x))%15
        results[name]={'metrics':public_metrics(y,z.numpy(),subjects),'mean_reliability':float(r.mean()),
                      'changed_region_reliability':float(r[torch.arange(len(x)),changed_region].mean()),
                      'mean_absolute_local_logit_contribution':float(c.abs().sum(1).mean()),
                      'mean_absolute_global_logit':float(g.abs().mean()),'nonzero_signed_local_atoms_mean':float((signed.abs()>1e-6).sum(1).float().mean()),
                      'negative_signed_atom_fraction':float((signed< -1e-6).float().mean()),
                      'exact_additive_accounting':True,'arithmetic_not_waveform_causality':True,
                      'mean_absolute_contribution_change':float((c-clean_parts).abs().mean())}
        private[name]={'logits':z.numpy(),'contributions':c.numpy(),'reliability':r.numpy()}
    return results,private

def run():
    start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config=json.loads((PUBLIC/'model_training_preregistration.json').read_text())
    for name,digest in config['implementation_sha256'].items():
        from .historical import ROOT
        if sha256((ROOT/name).read_bytes())!=digest:raise ValueError('frozen training implementation changed')
    if (PRIVATE/'training_started.json').exists():raise ValueError('nine-fit campaign already attempted; no silent restart')
    data=np.load(PRIVATE/'new_model_data.npz',allow_pickle=False);n=int(data['n_fit']);x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y']).long();subjects=data['subjects']
    if n>5000 or len(x)-n>1000:raise ValueError('frozen sample bound exceeded')
    write_new(PRIVATE/'training_started.json',{'fits_max':9,'updates_per_fit':300,'preregistration_sha256':sha256(canonical(config))})
    results=[];updates=0;forwards=0;data_streams={}
    for seed in (123,456,789):
        for arm in ('M0','M1','M2'):
            if time.perf_counter()-start>1800:raise TimeoutError('prospective lower CPU training ceiling')
            fit_start=time.perf_counter();torch.manual_seed(seed);model=ReliabilityLedger(arm!='M0');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            losses=[];stream_hash=hashlib.sha256();model.train()
            for indices,atoms,phase in training_stream(seed,n):
                if time.perf_counter()-start>1800:raise TimeoutError('training ceiling')
                for v in (indices,atoms,phase):stream_hash.update(v.numpy().tobytes())
                batch=x[indices];labels=y[indices];modified=local_hf(batch,atoms,phase);a=model(batch);b=model(modified)
                classification=.5*(F.cross_entropy(a['logits'],labels)+F.cross_entropy(b['logits'],labels))
                regularizer=preservation_loss(a,b,atoms) if arm=='M2' else classification.new_zeros(())
                loss=classification+regularizer
                if not torch.isfinite(loss):raise ValueError('nonfinite scientific fit; no retry')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                updates+=1;forwards+=32;losses.append([float(classification.detach()),float(regularizer.detach())])
            digest=stream_hash.hexdigest()
            if seed in data_streams and digest!=data_streams[seed]:raise ValueError('matched exposure stream mismatch')
            data_streams[seed]=digest
            fit_seconds=time.perf_counter()-fit_start;eval_start=time.perf_counter();metrics,private=evaluate(model,x[n:],y[n:].numpy(),subjects[n:]);forwards+=4*(len(x)-n)
            identity=arm+'_'+str(seed)
            torch.save({'state_dict':model.state_dict(),'arm':arm,'seed':seed,'updates':300},PRIVATE/(identity+'.pt'))
            np.savez_compressed(PRIVATE/(identity+'_evaluation.npz'),**{view+'_'+key:value for view,values in private.items() for key,value in values.items()})
            result={'arm':arm,'seed':seed,'updates':300,'clean_exposures':4800,'paired_nuisance_exposures':4800,
                    'stream_sha256':digest,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'train_classification_loss_first20_mean':float(np.mean(losses[:20],axis=0)[0]),'train_classification_loss_last20_mean':float(np.mean(losses[-20:],axis=0)[0]),
                    'regularizer_last20_mean':float(np.mean(losses[-20:],axis=0)[1]),'fit_seconds':fit_seconds,'evaluation_seconds':time.perf_counter()-eval_start,'evaluation':metrics}
            results.append(result);write_new(PUBLIC/(identity+'_metrics.json'),result)
            print(json.dumps({'fit':identity,'clean_macro_f1':metrics['clean']['metrics']['macro_f1'],'local_hf_macro_f1':metrics['local_hf']['metrics']['macro_f1'],'fit_seconds':fit_seconds}),flush=True)
    paired=[]
    for seed in (123,456,789):
        a=next(r for r in results if r['arm']=='M1' and r['seed']==seed);b=next(r for r in results if r['arm']=='M2' and r['seed']==seed)
        deltas={v:b['evaluation'][v]['metrics']['macro_f1']-a['evaluation'][v]['metrics']['macro_f1'] for v in ('clean','local_hf','strong_hf','mild_hf')}
        paired.append({'seed':seed,'M2_minus_M1':deltas,'primary_gate':deltas['local_hf']>=.02 and deltas['clean']>=-.02})
    write_new(PUBLIC/'model_comparison.json',{'status':'COMPLETE','fits':9,'optimizer_updates':updates,'full_example_forwards':forwards,
        'paired':paired,'all_seed_primary_gate':all(r['primary_gate'] for r in paired),'mean_delta':{v:float(np.mean([r['M2_minus_M1'][v] for r in paired])) for v in ('clean','local_hf','strong_hf','mild_hf')},
        'wall_seconds':time.perf_counter()-start,'GPU_seconds':0,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'final_all15_fit':'NOT_RUN_NO_SLOT_RESERVED','no_M3':True,'holdout_selection_exposure':'TRAIN-only mechanism discovery includes all fifteen historical TRAIN people; prospective training holdout is not independent scientific replication'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()

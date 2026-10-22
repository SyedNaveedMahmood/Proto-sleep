"""Equal-parameter offline context comparison; no neighbor labels as inputs."""
import json
import time
import resource
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mist_cove.phase13.model_adequacy import WidthLedger
from mist_cove.phase13.optimization import sampler,training_payload
from mist_cove.phase13.provenance import isolated_rng
from mist_cove.phase13.seals import save_sealed,load_sealed
from mist_cove.phase11b.historical import named_state_digest
from mist_cove.phase11b.evaluation import public_metrics
from mist_cove.phase11b.development import local_hf
from mist_cove.phase11b.security import authenticated_snapshot
from .common import ROOT,PRIVATE,PUBLIC,read_json,write_new,measured,ledger,check_budget,sha256,canonical


class ContextLedger(nn.Module):
    def __init__(self,arm='C0'):
        super().__init__()
        if arm not in ('C0','C1'):raise ValueError('frozen context arm required')
        self.arm=arm;self.base=WidthLedger('L',False)
        self.register_buffer('context_weights',torch.tensor([0.,1.,0.] if arm=='C0' else [.25,.5,.25]))
    def forward(self,x,valid):
        if x.ndim!=4 or x.shape[1:]!=(3,1,3000) or valid.shape!=x.shape[:2] or valid.dtype!=torch.bool or not valid[:,1].all():raise ValueError('three physical epochs with mandatory center/validity mask')
        weights=valid*self.context_weights[None];weights=weights/weights.sum(1,keepdim=True)
        out=self.base(x.reshape(-1,1,3000));c=out['contributions'].reshape(-1,3,15,5)*weights[:,:,None,None]
        g=out['global'].reshape(-1,3,5)*weights[:,:,None]
        return {'logits':self.base.bias+g.sum(1)+c.sum((1,2)),'contributions':c,'global':g,'weights':weights,
                'evidence':out['evidence'].reshape(-1,3,15,5),'global_features':out['global_features'].reshape(-1,3,7)}


def draw_update(model,optimizer,x,y,valid,generator):
    i=torch.randint(len(x),(16,),generator=generator);atoms=torch.randint(15,(16,),generator=generator);phase=2*torch.pi*torch.rand(16,generator=generator)
    a=model(x[i],valid[i]);changed=x[i].clone();changed[:,1]=local_hf(changed[:,1],atoms,phase)
    b=model(changed,valid[i]);loss=.5*(F.cross_entropy(a['logits'],y[i])+F.cross_entropy(b['logits'],y[i]));optimizer.zero_grad(set_to_none=True);loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise FloatingPointError('nonfinite/missing context gradient')
    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite state')
    return {'loss':float(loss.detach()),'gradient_norm':float(gradient),'draw_sha256':sha256(b''.join(v.numpy().tobytes() for v in (i,atoms,phase)))}


@torch.no_grad()
def evaluate(model,x,valid,y,people,transition):
    model.eval();z=[];global_size=[];local_size=[]
    for start in range(0,len(x),64):
        out=model(x[start:start+64],valid[start:start+64]);logits=out['logits']
        torch.testing.assert_close(logits,model.base.bias+out['global'].sum(1)+out['contributions'].sum((1,2)),atol=0,rtol=0)
        z.append(logits.numpy());global_size.extend(out['global'].sum(1).abs().mean(1).tolist());local_size.extend(out['contributions'].sum((1,2)).abs().mean(1).tolist())
    z=np.concatenate(z);result={'all':public_metrics(y,z,people),'transition':public_metrics(y[transition],z[transition],people[transition]),
         'nontransition':public_metrics(y[~transition],z[~transition],people[~transition]),'parameters':sum(p.numel() for p in model.parameters()),
         'active_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'mean_abs_global':float(np.mean(global_size)),
         'mean_abs_local':float(np.mean(local_size)),'additive_identity':'BITWISE'}
    return result,z


def context_identity(config,arm,seed):return {'phase':'phase14-context-v1','config_sha256':sha256(canonical(config)),'arm':arm,'seed':seed}


def run():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if (PRIVATE/'context_started.json').exists():raise ValueError('context campaign already attempted')
    prereg=read_json(PUBLIC/'context_preregistration.json');config=read_json(ROOT/'configs/mist_cove_phase14_v1.json')
    for name,digest in prereg['source_sha256'].items():
        if sha256((ROOT/name).read_bytes())!=digest:raise ValueError('context preregistered source changed')
    expected=read_json(PUBLIC/'SC_TRAIN_execution.json')['context_data_sha256']
    import io
    with authenticated_snapshot(PRIVATE/'context_data.npz',expected,allowlist=[PRIVATE/'context_data.npz'],limit=512*1024*1024) as snapshot:
        with np.load(io.BytesIO(snapshot.data),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    x=torch.from_numpy(data['x']);v=torch.from_numpy(data['valid']);y=torch.from_numpy(data['truth']).long();n=int(data['n_fit']);people=data['subjects'];transition=[]
    from .source_audit import cached_selection
    selection=cached_selection();lookup={r['record']['recording']:r['labels'] for r in read_json(PRIVATE/'geometry.json')['rows']}
    for c in selection['training']['fit']+selection['training']['internal_dev']:
        labels=lookup[c['recording_id']];i=c['epoch'];transition.append(i>0 and labels[i-1]>=0 and labels[i-1]!=labels[i])
    transition=np.array(transition);rows=[];updates=0;streams={};started=time.perf_counter()
    write_new(PRIVATE/'context_started.json',{'fits':6,'updates':18000,'preregistration_sha256':sha256(canonical(prereg))})
    with measured('context_confirmatory_training') as counts:
        for seed in (123,456,789):
            for arm in ('C0','C1'):
                with isolated_rng(seed):
                    model=ContextLedger(arm);optimizer=torch.optim.Adam(model.parameters(),lr=.001);generator=sampler(seed);history=[];start=time.perf_counter()
                    for step in range(3000):
                        if step%25==0:check_budget(started)
                        model.train();history.append(draw_update(model,optimizer,x[:n],y[:n],v[:n],generator));updates+=1
                    identity=context_identity(config,arm,seed);path=PRIVATE/f'context_{arm}_{seed}.pt'
                    payload=training_payload(model,optimizer,generator,3000,history,identity);seal=save_sealed(path,payload,{**identity,'step':3000})
                    evaluation,z=evaluate(model,x[n:],v[n:],data['truth'][n:],people[n:],transition[n:])
                    np.savez_compressed(PRIVATE/f'context_{arm}_{seed}_dev_predictions.npz',logits=z,truth=data['truth'][n:],subjects=people[n:],transition=transition[n:])
                    stream=sha256(''.join(row['draw_sha256'] for row in history).encode())
                    if seed in streams and streams[seed]!=stream:raise ValueError('context exposure streams differ')
                    streams[seed]=stream
                    row={'arm':arm,'seed':seed,'step':3000,'training_CE_last100':float(np.mean([r['loss'] for r in history[-100:]])),
                         'gradient_norm_last100':float(np.mean([r['gradient_norm'] for r in history[-100:]])),
                         'evaluation':evaluation,'checkpoint_byte_sha256':seal['byte_sha256'],'named_state_digest':seal['state_digest'],'stream_sha256':stream,
                         'optimizer_updates':3000,'clean_center_exposures':48000,'paired_center_exposures':48000,'base_waveform_forwards_training':288000,
                         'fit_wall_seconds':time.perf_counter()-start}
                    rows.append(row);write_new(PUBLIC/f'context_{arm}_{seed}.json',row)
                    print(json.dumps({'context':arm,'seed':seed,'F1':evaluation['all']['mean_subject_macro_f1'],'N1':evaluation['all']['per_stage_recall'][1]}),flush=True)
        by={(r['arm'],r['seed']):r for r in rows};gates={}
        for seed in (123,456,789):
            a=by['C1',seed]['evaluation'];b=by['C0',seed]['evaluation']
            delta=a['all']['mean_subject_macro_f1']-b['all']['mean_subject_macro_f1'];n1=a['all']['per_stage_recall'][1]-b['all']['per_stage_recall'][1]
            td=a['transition']['accuracy']-b['transition']['accuracy'];stage=np.array(a['all']['per_stage_f1'])-b['all']['per_stage_f1']
            gates[str(seed)]={'F1_delta':delta,'N1_recall_delta':n1,'transition_accuracy_delta':td,'worst_stage_F1_delta':float(stage.min()),
                             'pass':bool(delta>=.02 and n1>=.05 and td>=.03 and stage.min()>=-.10)}
        result={'status':'COMPLETE','fits':6,'updates':updates,'rows':rows,'gates':gates,'primary_gate':'PASS' if all(r['pass'] for r in gates.values()) else 'FAIL',
                'matched_streams':'PASS','wall_seconds':time.perf_counter()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_seconds':0,
                'ST_data_or_labels_used':False,'ST_candidate_policy':'all6fixedfinalstates asprespecifiedcomparators; no STpromotion following SCfailure'}
        write_new(PUBLIC/'context_results.json',result);counts.update(fits=6,optimizer_updates=updates,base_waveform_forwards_training=1728000,peak_RSS_KiB=result['peak_RSS_KiB'])
        ledger([28,29,30],'COMPLETE','All6equal-parameter fixed context trajectories complete; all seeds and exposure streams retained; SCpromotion gate evaluated',['reports/mist_cove/phase14_v1/context_results.json'],science=result['primary_gate'])


_MODELS=None

def load_models():
    global _MODELS
    if _MODELS is None:
        config=read_json(ROOT/'configs/mist_cove_phase14_v1.json');result=read_json(PUBLIC/'context_results.json');_MODELS=[]
        for row in result['rows']:
            arm=row['arm'];seed=row['seed'];path=PRIVATE/f'context_{arm}_{seed}.pt';seal=read_json(path.with_suffix('.seal.json'))
            if seal['byte_sha256']!=row['checkpoint_byte_sha256']:raise ValueError('context byte anchor differs')
            payload,_=load_sealed(path,{**context_identity(config,arm,seed),'step':3000})
            with torch.random.fork_rng(devices=[]):m=ContextLedger(arm).eval()
            m.load_state_dict(payload['state_dict']);
            if named_state_digest(m.state_dict())!=row['named_state_digest']:raise ValueError('context named state differs')
            _MODELS.append((arm,seed,m))
    return _MODELS


@torch.no_grad()
def st_predict(record,wave,normalization,outdir,n):
    indices=np.arange(n)[:,None]+np.array([-1,0,1])[None];valid=(indices>=0)&(indices<n);indices=indices.clip(0,n-1)
    inputs=(wave[indices]-normalization['mean_uV'])/normalization['std_uV'];inputs[~valid]=0
    for arm,seed,model in load_models():
        outputs=[]
        for start in range(0,n,64):outputs.append(model(torch.from_numpy(inputs[start:start+64].astype('float32')),torch.from_numpy(valid[start:start+64]))['logits'].numpy())
        np.save(outdir/(record['recording']+f'_{arm}_{seed}_logits.npy'),np.concatenate(outputs))

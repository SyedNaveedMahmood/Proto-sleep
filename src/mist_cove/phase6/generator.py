"""Complete physical waveforms composed from independent, auditable streams."""
import copy
from dataclasses import dataclass
import numpy as np
import torch

COMPLETE=(2.,5.,8.,11.,14.)
FACTOR=(3.5,6.5,9.5,12.5,15.5)
CONDITIONS={'train','clean','competitive','removed','nuisance_swap','position_shift','partial','eventless','redundant','complementary','repeated','correlated','adjacent','overlap','weak','artifact'}


def rng(seed,stream,index=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(stream),int(index)]))


def burst(hz,phase,rms):
    t=np.arange(200,dtype=np.float64)/100
    window=np.sin(np.pi*np.arange(200)/199)**2;window[[0,-1]]=0
    wave=window*np.sin(2*np.pi*hz*t+phase)
    wave*=rms/np.sqrt(np.mean(wave**2))
    return torch.from_numpy(wave.astype(np.float32))


def latent_draw(n,seed,task):
    if n<=0 or n%5 or task not in {'single','multiple'}: raise ValueError('balanced synthetic five-class task required')
    y=np.arange(n)%5;rng(seed,1).shuffle(y)
    rows=[]
    for i,label in enumerate(y):
        position=rng(seed,2,i);diagnostic=rng(seed,3,i);background=rng(seed,4,i)
        nuisance=rng(seed,6,i);artifact=rng(seed,7,i)
        starts=(position.choice([100,700,1300,1900,2500],2,replace=False)+position.integers(-80,81,2)).tolist()
        rows.append({'label':int(label),'task':task,'seed':int(seed),'index':i,
            'mode':'redundant' if rng(seed,8,i).random()<.5 else 'complementary',
            'diagnostic_onsets':starts,'diagnostic_rms':diagnostic.uniform(8,12,2).tolist(),
            'diagnostic_phase':diagnostic.uniform(0,2*np.pi,2).tolist(),
            'background_amplitude':float(background.uniform(0,2)),'background_phase':float(background.uniform(0,2*np.pi)),
            'noise_stream':[int(seed),5,i],
            'nuisance_onsets':nuisance.integers(20,2781,4).tolist(),
            'nuisance_codes':nuisance.integers(0,3,4).tolist(),'nuisance_phase':nuisance.uniform(0,2*np.pi,4).tolist(),
            'nuisance_unit_amplitude':nuisance.random(4).tolist(),
            'artifact_onset':int(artifact.integers(20,2961)),'artifact_sign':int(artifact.choice([-1,1])),
            'artifact_unit_amplitude':float(artifact.random()),'generation_order':i})
    return rows


@dataclass
class Draw:
    x:torch.Tensor
    y:torch.Tensor
    base:torch.Tensor
    background:torch.Tensor
    noise:torch.Tensor
    nuisance:torch.Tensor
    artifact:torch.Tensor
    events:torch.Tensor
    intervals:list
    nuisance_intervals:list
    latents:list
    task:str
    condition:str

    @property
    def diagnostic(self):return self.events.sum(1)


def compose(base,events,keep=None):
    if keep is not None:events=events*keep[:,:,None,None]
    return base+events.sum(1)


def generate(n,seed,task='single',condition='clean',*,latents=None,labels=None):
    if task not in {'single','multiple'} or condition not in CONDITIONS:raise ValueError('unknown synthetic task/condition')
    records=copy.deepcopy(latents) if latents is not None else latent_draw(n,seed,task)
    if len(records)!=n or any(r['task']!=task or r['seed']!=seed for r in records):raise ValueError('latent identity mismatch')
    if labels is not None:
        labels=np.asarray(labels)
        if labels.shape!=(n,) or not np.isin(labels,range(5)).all():raise ValueError('invalid diagnostic labels')
        for row,label in zip(records,labels):row['label']=int(label)
    bg=torch.zeros(n,1,3000);noise=bg.clone();nu=bg.clone();art=bg.clone();ev=torch.zeros(n,2,1,3000)
    intervals=[];nuisance_intervals=[];t=np.arange(3000)/100
    for i,r in enumerate(records):
        u=r
        # Nuisance substitution uses independent per-index streams, not a reused
        # five-example bank; the exact diagnostic record is retained.
        if condition=='nuisance_swap':
            u=copy.deepcopy(r);other_seed=r['seed']+1;j=r['index']
            b=rng(other_seed,4,j);v=rng(other_seed,6,j);a=rng(other_seed,7,j)
            u.update(seed=other_seed,noise_stream=[other_seed,5,j],background_amplitude=float(b.uniform(0,2)),background_phase=float(b.uniform(0,2*np.pi)),
                nuisance_onsets=v.integers(20,2781,4).tolist(),nuisance_codes=v.integers(0,3,4).tolist(),nuisance_phase=v.uniform(0,2*np.pi,4).tolist(),nuisance_unit_amplitude=v.random(4).tolist(),
                artifact_onset=int(a.integers(20,2961)),artifact_sign=int(a.choice([-1,1])),artifact_unit_amplitude=float(a.random()))
        r['rendered_baseline']={key:copy.deepcopy(u[key]) for key in ['background_amplitude','background_phase','noise_stream','nuisance_onsets','nuisance_codes','nuisance_phase','nuisance_unit_amplitude','artifact_onset','artifact_sign','artifact_unit_amplitude']}
        bg[i,0]=torch.from_numpy((u['background_amplitude']*np.sin(2*np.pi*.7*t+u['background_phase'])).astype(np.float32))
        noise[i,0]=torch.from_numpy((rng(*u['noise_stream']).standard_normal(3000)*.5).astype(np.float32))
        # rng(seed,stream,index) receives exactly the recorded three stream keys.
        count=4 if condition in {'competitive','repeated','correlated'} else 2
        sites=[]
        for j in range(count):
            k=0 if condition=='correlated' else j
            scale=(36+12*u['nuisance_unit_amplitude'][k]) if condition=='competitive' else (12+12*u['nuisance_unit_amplitude'][k])
            wave=burst((24,30,36)[u['nuisance_codes'][k]],u['nuisance_phase'][k],scale)
            lo=u['nuisance_onsets'][j];nu[i,0,lo:lo+200]+=wave;sites.append((lo,lo+200))
        alo=u['artifact_onset'];peak=(80+20*u['artifact_unit_amplitude']) if condition in {'competitive','artifact'} else (8+8*u['artifact_unit_amplitude'])
        art[i,0,alo:alo+20]=u['artifact_sign']*peak
        mode=condition if condition in {'redundant','complementary'} else r['mode']
        codes=[r['label']] if task=='single' else ([r['label']]*2 if mode=='redundant' else [r['label'],(r['label']+1)%5])
        family=COMPLETE if task=='single' or mode=='redundant' else FACTOR
        truth=[]
        for j,code in enumerate(codes):
            lo=r['diagnostic_onsets'][j]
            if j==1 and condition in {'adjacent','overlap'}:lo=r['diagnostic_onsets'][0]+(200 if condition=='adjacent' else 100)
            if condition=='position_shift':lo=lo+200 if lo+400<=3000 else lo-200
            wave=burst(family[code],r['diagnostic_phase'][j],r['diagnostic_rms'][j])
            if condition=='weak':wave=wave*.2
            present=condition not in {'removed','eventless'}
            if condition=='partial':
                if task=='multiple' and j==0:present=False
                elif task=='single':wave=wave.clone();wave[100:]=0
            if present:ev[i,j,0,lo:lo+200]=wave;truth.append((lo,lo+200))
        intervals.append(truth);nuisance_intervals.append(sites)
    base=bg+noise+nu+art
    if condition=='eventless':bg.zero_();noise.zero_();nu.zero_();art.zero_();base.zero_();ev.zero_();nuisance_intervals=[[] for _ in records]
    return Draw(compose(base,ev),torch.tensor([r['label'] for r in records]),base,bg,noise,nu,art,ev,intervals,nuisance_intervals,records,task,condition)


def class_swap(draw):
    return generate(len(draw.y),draw.latents[0]['seed'],draw.task,draw.condition,latents=draw.latents,labels=(draw.y.numpy()+1)%5)


def conditions_for(task,conditions):
    return [v for v in conditions if task=='multiple' or v not in {'redundant','complementary','adjacent','overlap'}]

"""Hand-constructed dependency controls, explicitly not learned EEG models.

The detectors inspect selected actual 100-sample windows. Their locations and
matched templates use privileged generated knowledge; native model rankings do
not receive that oracle. The common availability/pooling operator is reused.
"""
import torch
from mist_evidence.model import ModelConfig
from .dependency_graph import support_map
from .source_adapter import SourceCache,tensor_digest
from .source_availability import masked_features

CONTROL_NAMES=('localized','distributed','global_amplitude','high_frequency_nuisance',
               'broadband_interference','redundant','complementary','disruptor',
               'eventless','whole_epoch','mixed_global_local','neutral_bias',
               'aggregation_sensitivity','incorrect_explainable')


class ControlAdapter:
    def __init__(self,name):
        if name not in CONTROL_NAMES:raise ValueError('unknown frozen control')
        self.name=name
        self.graph=support_map(ModelConfig(raw_control=True,radius=0,use_crf=False,embedding_dim=4))
        self.locations=(2,10,13) if name=='distributed' else (2,10,6)
        self.oracle_atoms=[2,10,13] if name=='distributed' else ([2,10] if name in ('redundant','complementary','disruptor') else [2])
        if name in ('global_amplitude','whole_epoch','mixed_global_local'):self.oracle_atoms=list(range(15))
        if name in ('neutral_bias','eventless'):self.oracle_atoms=[]
        self.truth_label=4 if name=='incorrect_explainable' else 0

    def waveform(self):
        x=torch.zeros(1,1,3000);t=torch.arange(100)/100
        selected=[0]
        if self.name in ('redundant','complementary','disruptor','distributed'):selected.append(1)
        if self.name in ('disruptor','distributed','high_frequency_nuisance'):selected.append(2)
        if self.name in ('eventless','neutral_bias'):selected=[]
        for j in selected:
            start=self.locations[j]*200;amplitude=8. if self.name=='high_frequency_nuisance' and j==2 else 1.
            x[0,0,start:start+100]=amplitude*torch.sin(2*torch.pi*(4,9,30)[j]*t)
        if self.name in ('global_amplitude','whole_epoch','mixed_global_local'):
            # Fixed whole-epoch background amplitude, no real fitted statistics.
            x+=torch.randn(x.shape,generator=torch.Generator().manual_seed(822))
        if self.name=='broadband_interference':
            x+=4*torch.randn(x.shape,generator=torch.Generator().manual_seed(823))
        return x

    @torch.no_grad()
    def cache(self,x):
        if tuple(x.shape)!=(1,1,3000) or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('one finite synthetic tensor required')
        local=[];t=torch.arange(100,device=x.device)/100
        for scale in self.graph['scales']:
            values=x.new_zeros(1,len(scale['windows']),4)
            if scale['samples']==100:
                for j,atom in enumerate(self.locations):
                    index=next(w['index'] for w in scale['windows'] if w['start']==atom*200)
                    wave=x[:,0,atom*200:atom*200+100]
                    values[:,index,j]=(wave*torch.sin(2*torch.pi*(4,9,30)[j]*t)).sum(-1).clamp_min(0)/50
            local.append(values)
        global_=x.new_zeros(1,17);global_[:,-1]=x[:,0].square().mean(-1).sqrt()
        return SourceCache(tuple(local),global_,'analytic_generated_control',self.graph['identity'],tensor_digest((x,)),tensor_digest((*local,global_)))

    def features(self,cache,masks,*,neutral=0.):
        return masked_features(cache,self.graph,masks,neutral)

    def score(self,features):
        a,b,c=features[...,4],features[...,5],features[...,6]
        g=features[...,-1]
        if self.name=='distributed':return a+b+c-2.5
        if self.name in ('global_amplitude','whole_epoch'):return g-.5
        if self.name=='mixed_global_local':return a+g-1.5
        if self.name=='neutral_bias':return torch.ones_like(a)
        if self.name=='redundant':return a+b-.5
        if self.name=='complementary':return a+b-1.5
        if self.name=='disruptor':return 2*a+2*b-2.5*c-.5
        if self.name=='aggregation_sensitivity':return features[...,0]-.02
        return a-.5

    def logits(self,cache,masks,*,neutral=0.):
        f=self.features(cache,masks,neutral=neutral).features;s=self.score(f)
        return torch.stack((s,-s,torch.full_like(s,-3),torch.full_like(s,-4),torch.full_like(s,-5)),-1)

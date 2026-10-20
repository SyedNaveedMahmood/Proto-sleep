"""Explanation interface to exact reviewed raw_context architecture; no loaders."""
from dataclasses import dataclass
import hashlib
import torch
from mist_evidence.model import EvidenceModel, LocalEncoder, WaveformBank, RawTemporal, epoch_summary
from .dependency_graph import support_map, digest, validate_config
from .source_availability import masked_features


def tensor_digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        x = tensor.detach().contiguous().cpu()
        h.update(str((str(x.dtype), tuple(x.shape))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def model_identity(model):
    return digest({'architecture': repr(model), 'config': model.cfg.dictionary(),
                   'state': tensor_digest(model.state_dict().values())})


def verify_architecture(model):
    """Reject altered normalization, topology or hooks before support claims.

    A temporary CPU skeleton uses generated ramp buffers only. It is never
    attached to the classifier, fitted or used for inference. fork_rng keeps
    model validation from changing the caller's CPU random stream.
    """
    cfg=model.cfg
    anchors=[torch.arange(length,dtype=torch.float32).repeat(cfg.prototypes_per_scale,1) for length in cfg.scales]
    with torch.random.fork_rng(devices=[]):
        reference=EvidenceModel(cfg,anchors,(0.,1.)).eval()
    actual=list(model.named_modules());expected=list(reference.named_modules())
    if len(actual)!=len(expected):raise ValueError('unreviewed architecture topology')
    for (name,item),(wanted,standard) in zip(actual,expected):
        if name!=wanted or type(item) is not type(standard) or repr(item)!=repr(standard):
            raise ValueError('unreviewed architecture or normalization: '+name)
        if item._forward_hooks or item._forward_pre_hooks or 'forward' in item.__dict__:
            raise ValueError('unreviewed forward hooks or overrides')
    if 'encode' in model.__dict__ or 'emissions' in model.__dict__ or model.feature_names!=reference.feature_names:
        raise ValueError('unreviewed encoding/head override or feature ordering')


@dataclass(frozen=True)
class SourceCache:
    local_embeddings: tuple
    global_features: torch.Tensor
    model_identity: str
    dependency_identity: str
    input_identity: str
    content_identity: str

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in (*self.local_embeddings, self.global_features))


class SourceAdapter:
    """No new parameters; accepts in-memory model, never a checkpoint or bank path.

    Phase X runners construct only make_mock instances. Future source access
    requires a separate reviewed runner; this class grants no artifact access.
    """
    def __init__(self, model):
        if type(model) is not EvidenceModel or model.training:
            raise ValueError('exact historical EvidenceModel in eval mode required')
        validate_config(model.cfg)
        if type(model.temporal) is not RawTemporal or model.crf is not None:
            raise ValueError('historical pointwise raw head required')
        if any(type(b) is not WaveformBank or type(b.encoder) is not LocalEncoder for b in model.banks):
            raise ValueError('unreviewed local encoder')
        if len(model.banks) != len(model.cfg.scales):
            raise ValueError('bank/config mismatch')
        for b, scale in zip(model.banks, model.cfg.scales):
            if tuple(b.waveforms.shape) != (model.cfg.prototypes_per_scale, scale):
                raise ValueError('actual unfold geometry differs from audited config')
        verify_architecture(model)
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('frozen FP32 operator required')
        self.model = model
        self.graph = support_map(model.cfg)
        self.identity = model_identity(model)
        self.dependency_identity = self.graph['identity']
        self._graph_digest = digest(self.graph)

    def check(self, cache=None):
        for module in self.model.modules():
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or 'forward' in module.__dict__:
                raise ValueError('unreviewed hooks or forward override added after construction')
        if 'encode' in self.model.__dict__ or 'emissions' in self.model.__dict__:
            raise ValueError('unreviewed encoding/head override added after construction')
        if self.model.training or model_identity(self.model) != self.identity or digest(self.graph) != self._graph_digest:
            raise ValueError('frozen model/dependency identity changed')
        if cache is not None:
            if not isinstance(cache, SourceCache) or cache.model_identity != self.identity or cache.dependency_identity != self.dependency_identity:
                raise ValueError('cache/model support identity mismatch')
            if tensor_digest((*cache.local_embeddings,cache.global_features)) != cache.content_identity:
                raise ValueError('cached representation tampered')

    @torch.no_grad()
    def cache(self, x):
        self.check()
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or tuple(x.shape[1:]) != (1,3000) or x.dtype != torch.float32 or len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError('finite FP32 [N,1,3000] in-memory normalized input required')
        if x.device != next(self.model.parameters()).device:
            raise ValueError('model and input devices must match')
        epochs = x[:,0].float()
        local = []
        for bank in self.model.banks:
            windows = epochs.unfold(-1, bank.waveforms.shape[-1], self.model.cfg.stride)
            n,p,length = windows.shape
            local.append(bank.encoder(windows.reshape(-1,length)).reshape(n,p,-1))
        global_ = []
        if self.model.cfg.use_summary:
            global_.append(epoch_summary(epochs))
            if self.model.cfg.use_amplitude:
                rms = (epochs-epochs.mean(-1,keepdim=True)).square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
                global_.append(((rms.log()-self.model.amplitude_stats[0])/self.model.amplitude_stats[1]).sigmoid())
        global_ = torch.cat(global_,-1) if global_ else epochs.new_empty(len(x),0)
        if any(not torch.isfinite(v).all() for v in (*local,global_)):
            raise ValueError('nonfinite cached features')
        return SourceCache(tuple(local),global_,self.identity,self.dependency_identity,tensor_digest((x,)),tensor_digest((*local,global_)))

    @torch.no_grad()
    def features(self, cache, atoms, *, neutral=0.):
        self.check(cache)
        return masked_features(cache, self.graph, atoms, neutral)

    @torch.no_grad()
    def logits(self, cache, atoms, *, neutral=0.):
        trace = self.features(cache, atoms, neutral=neutral)
        q,n,f = trace.features.shape
        output = self.model.emissions(trace.features.reshape(1,q*n,f))[0].reshape(q,n,5)
        if not torch.isfinite(output).all():
            raise ValueError('finite five-stage output required')
        return output

    @torch.no_grad()
    def native(self, x):
        self.check()
        return self.model.emissions(self.model.encode(x)[None])[0]

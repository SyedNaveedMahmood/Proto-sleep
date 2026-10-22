"""Width-only scaling of the Phase XII local/global additive model."""
import torch
from torch import nn
from mist_cove.phase11b.development import features


class WidthLedger(nn.Module):
    def __init__(self, capacity='S', routing=True):
        super().__init__()
        if capacity not in ('S', 'L') or type(routing) is not bool:
            raise ValueError('only frozen S/L and R0/R1 family allowed')
        self.capacity, self.routing = capacity, routing
        factor = 1 if capacity == 'S' else 4
        first, second = 8*factor, 16*factor
        self.encoder = nn.Sequential(nn.Conv1d(1, first, 9, stride=2), nn.ELU(),
                                     nn.Conv1d(first, second, 5, stride=2), nn.ELU(),
                                     nn.AdaptiveAvgPool1d(1))
        self.local = nn.Linear(second, 5, bias=False)
        self.global_head = nn.Sequential(nn.Linear(7, 16*factor), nn.ELU(),
                                         nn.Linear(16*factor, 5, bias=False))
        self.bias = nn.Parameter(torch.zeros(5))
        self.reliability = nn.Sequential(nn.Linear(4, 8*factor), nn.ELU(),
                                         nn.Linear(8*factor, 1), nn.Sigmoid())
        if not routing:
            self.reliability.requires_grad_(False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (1, 3000):
            raise ValueError('original [N,1,3000] geometry required')
        atoms, quality, global_features = features(x)
        embedding = self.encoder(atoms.reshape(-1, 1, 200)).flatten(1).reshape(-1, 15, self.local.in_features)
        evidence = 3*torch.tanh(self.local(embedding))
        reliability = self.reliability(quality) if self.routing else torch.ones_like(evidence[..., :1])
        contributions = reliability*evidence/15
        global_logits = self.global_head(global_features)
        logits = self.bias+global_logits+contributions.sum(1)
        return {'logits': logits, 'global': global_logits, 'contributions': contributions,
                'reliability': reliability, 'evidence': evidence,
                'global_features': global_features, 'embedding': embedding}

    def counts(self):
        active = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stored = sum(p.numel() for p in self.parameters())
        return {'trainable': active, 'frozen': stored-active, 'stored': stored,
                'local_convolution_receptive_field_samples': 17,
                'local_atom_samples': 200, 'global_features': 7,
                'temporal_context_epochs': 1}

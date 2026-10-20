"""Frozen conservative source feature operator. No file IO or learned modules."""
from dataclasses import dataclass
import torch

NEUTRALS = (0., .01)


def require_masks(atoms, device):
    if not isinstance(atoms, torch.Tensor) or atoms.dtype != torch.bool or atoms.ndim != 2 or atoms.shape[1] != 15 or len(atoms) == 0:
        raise ValueError('nonempty [queries,15] Boolean physical masks required')
    if atoms.device != device:
        raise ValueError('mask and cached feature device must match')


def window_visibility(atoms, windows):
    return torch.stack([atoms[:, w['atoms']].all(-1) for w in windows], -1)


def aggregate_embeddings(embed, visible, neutral=0.):
    if neutral not in NEUTRALS:
        raise ValueError('unregistered neutral representation')
    # [N,P,D] and [Q,P] -> [Q,N,D]. Fixed P, never count-normalized.
    q = visible[:, None, :, None]
    values = embed[None]
    mean = torch.where(q, values, neutral).mean(2)
    max_ = values.expand(len(visible), -1, -1, -1).masked_fill(~q, -torch.inf).amax(2)
    max_ = torch.where(visible.any(-1)[:, None, None], max_, neutral)
    return mean, max_


@dataclass(frozen=True)
class FeatureTrace:
    features: torch.Tensor
    window_visibility: tuple
    feature_availability: torch.Tensor
    valid_window_counts: torch.Tensor
    full_mask: torch.Tensor


def masked_features(cache, graph, atoms, neutral=0.):
    require_masks(atoms, cache.global_features.device)
    if neutral not in NEUTRALS:
        raise ValueError('unregistered neutral representation')
    values, visibility, flags = [], [], []
    for embed, scale in zip(cache.local_embeddings, graph['scales']):
        valid = window_visibility(atoms, scale['windows'])
        visibility.append(valid)
        values.extend(aggregate_embeddings(embed, valid, neutral))
        flags.append(valid.any(-1)[:, None].expand(-1, 2*scale['embedding_dim']))
    full = atoms.all(-1)
    if graph['global_features']:
        values.append(torch.where(full[:, None, None], cache.global_features[None], neutral))
        flags.append(full[:, None].expand(-1, len(graph['global_features'])))
    features = torch.cat(values, -1)
    if not torch.isfinite(features).all():
        raise ValueError('masked feature vector must be finite')
    return FeatureTrace(features, tuple(visibility), torch.cat(flags,-1),
                        torch.stack([v.sum(-1) for v in visibility], -1), full)

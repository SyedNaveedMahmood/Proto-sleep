import torch
from torch.nn import functional as F


def test_explicit_pool_matches_adaptive_means_and_gradients():
    from protosleep.evidence.model import DeterministicAdaptivePool1d
    for length in (13, 25, 50):
        x = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
        a = DeterministicAdaptivePool1d(4)(x)
        b = F.adaptive_avg_pool1d(x, 4)
        torch.testing.assert_close(a, b)
        ga, = torch.autograd.grad(a.square().sum(), x, retain_graph=True)
        gb, = torch.autograd.grad(b.square().sum(), x)
        torch.testing.assert_close(ga, gb)

"""Generated counterexamples for alignment, leakage and objective interpretation."""
import numpy as np
import pytest
import torch

from mist_sleep.augmentation import paired_view, stress_view
from mist_sleep.core import MISTSleep, mpi_terms
from mist_sleep.experiment import make_sampler, sample_batch
from mist_sleep.metrics import fixed_five


def test_fit_sampler_never_draws_development():
    stage = np.tile(np.arange(5, dtype=np.int8), 4)
    role = np.r_[np.zeros(10, dtype=np.int8), np.ones(10, dtype=np.int8)]
    pools, rng = make_sampler({"stage": stage, "role": role}, 123)
    chosen = sample_batch(pools, rng, 100)
    assert np.all(role[chosen] == 0)


def test_unaligned_shift_is_not_a_zero_mpi_pair():
    torch.manual_seed(4)
    model = MISTSleep().eval()
    x = torch.randn(2, 1, 3000)
    with torch.no_grad():
        a = model(x)
        shifted = model(torch.roll(x, 100, dims=-1))
    distance, kl = mpi_terms(a, shifted)
    assert float(distance) > 0 and float(kl) > 0


def test_permuted_prototype_identity_has_nonzero_kl():
    embedding = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[[.9, .1], [.8, .2]]])
    a = {"embedding": embedding, "q": q}
    changed = {"embedding": embedding, "q": q.flip(-1)}
    assert float(mpi_terms(a, a)[1]) == pytest.approx(0, abs=1e-6)
    assert float(mpi_terms(a, changed)[1]) > 0
    with pytest.raises(ValueError):
        mpi_terms(a, {"embedding": embedding, "q": q[:, :1]})


def test_constant_embedding_uniform_assignments_expose_loss_limitation():
    # A trivial representation can satisfy MPI; stage, variance and occupancy
    # checks are therefore separate scientific gates, not consequences of MPI.
    a = {"embedding": torch.zeros(4, 8), "q": torch.full((4, 3, 64), 1/64)}
    distance, kl = mpi_terms(a, a)
    assert float(distance) == 0 and abs(float(kl)) < 1e-7
    assert float(a["embedding"].var()) == 0


def test_invalid_stage_nonfinite_and_destructive_view_are_separate():
    with pytest.raises(ValueError):
        fixed_five(np.array([5]), np.zeros((1, 5)), np.array([0]))
    x = torch.zeros(1, 1, 3000)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        paired_view(x, torch.Generator())
    clean = torch.ones(1, 1, 3000)
    destructive = stress_view(clean, torch.Generator().manual_seed(5))
    assert float((destructive-clean).abs().mean()) > .1

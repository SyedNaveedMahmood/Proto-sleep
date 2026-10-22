import numpy as np
import pytest
import torch

from mist_sleep.augmentation import paired_view, stress_view
from mist_sleep.core import (Geometry, MISTSleep, MorphologyTokenizer,
                             SharedPrototypeBank, mpi_terms, prototype_diagnostics,
                             supcon_loss)
from mist_sleep.metrics import fixed_five
from mist_sleep.experiment import _checkpoint, restore_checkpoint, new_model, one_update


def test_token_geometry_and_locality():
    tokenizer = MorphologyTokenizer()
    impulse = torch.zeros(1, 1, 3000)
    impulse[0, 0, 1000] = 1
    windows = tokenizer(impulse)
    bounds = tokenizer.bounds_seconds()
    assert windows.shape == (1, 29, 200)
    assert bounds[0].tolist() == [0, 2]
    assert bounds[-1].tolist() == [28, 30]
    assert ((windows != 0).any(-1)).sum() == 2


def test_prototype_normalization_kmeans_and_diagnostics():
    bank = SharedPrototypeBank()
    torch.manual_seed(4)
    data = torch.randn(128, 64)
    bank.initialize_from_fit(data, 123)
    q, recon = bank(data[:4].reshape(1, 4, 64))
    assert q.shape == (1, 4, 64)
    assert torch.allclose(q.sum(-1), torch.ones(1, 4), atol=1e-6)
    assert torch.allclose(bank.vectors.norm(dim=-1), torch.ones(64), atol=1e-5)
    assert prototype_diagnostics(q, bank.vectors)["effective_prototypes"] > 1
    with pytest.raises(ValueError):
        bank.initialize_from_fit(data[:10], 123)


def test_mpi_independent_terms_gradients_and_identity():
    torch.manual_seed(6)
    model = MISTSleep()
    x = torch.randn(2, 1, 3000)
    a = model(x)
    zero = mpi_terms(a, a)
    assert abs(float(zero[0])) < 1e-6
    assert abs(float(zero[1])) < 1e-6
    b = model(x * .97 + .02)
    distance, kl = mpi_terms(a, b)
    (distance + kl).backward()
    assert float(model.bank.vectors.grad.abs().sum()) > 0
    assert float(model.encoder.branches[0][0].weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        mpi_terms(a, {**b, "q": b["q"][:, :-1]})


def test_supcon_rare_positive_and_no_positive():
    z = torch.randn(3, 8, requires_grad=True)
    assert float(supcon_loss(z, torch.arange(3))) == 0
    loss = supcon_loss(torch.cat((z, z)), torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert float(z.grad.abs().sum()) > 0


def test_bounded_augmentation_preserves_grid_and_determinism():
    x = torch.zeros(2, 1, 3000)
    a = paired_view(x, torch.Generator().manual_seed(8))
    b = paired_view(x, torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert a.shape == x.shape
    assert float(a.abs().max()) < .25
    assert stress_view(x, torch.Generator().manual_seed(8)).shape == x.shape
    with pytest.raises(ValueError):
        paired_view(x[:, :, :-1], torch.Generator())


def test_fixed_five_includes_missing_classes_and_participants():
    logits = np.zeros((4, 5))
    logits[:, 0] = 1
    result = fixed_five(np.array([0, 0, 1, 1]), logits, np.array([0, 0, 1, 1]))
    assert len(result["per_stage_recall"]) == 5
    assert result["per_stage_recall"][1:] == [0, 0, 0, 0]
    assert result["participant_count"] == 2
    assert sum(result["prediction_counts"]) == 4


def test_full_epoch_rejects_ambiguous_geometry():
    model = MISTSleep()
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 2999))


def test_checkpoint_exact_cpu_resume(tmp_path):
    torch.manual_seed(31)
    model = new_model("M0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    sample_rng = np.random.default_rng(31)
    aug_rng = torch.Generator().manual_seed(32)
    x = torch.randn(2, 1, 3000)
    y = torch.tensor([0, 1])
    one_update(model, optimizer, x, y, x*.98, "M0")
    path = tmp_path / "state.pt"
    _checkpoint(path, model, optimizer, sample_rng, aug_rng, 1, "M0", 31, "protocol")
    one_update(model, optimizer, x, y, x*.98, "M0")
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    other = new_model("M0")
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=.001)
    other_rng = np.random.default_rng(999)
    other_aug = torch.Generator().manual_seed(999)
    assert restore_checkpoint(path, other, other_optimizer, other_rng, other_aug,
                              "M0", 31, "protocol") == 1
    one_update(other, other_optimizer, x, y, x*.98, "M0")
    assert all(torch.equal(other.state_dict()[key], value) for key, value in expected.items())

import os
import numpy as np
import pytest
import torch

from mist_sleep.experiment import _checkpoint, new_model, one_update, restore_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bounded CUDA correctness needs an idle compatible GPU")
def test_generated_cuda_update_and_exact_resume(tmp_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    model = new_model("M2").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    sampler = np.random.default_rng(77)
    augment = torch.Generator().manual_seed(78)
    x = torch.randn(4, 1, 3000, device="cuda")
    y = torch.tensor([0, 1, 2, 4], device="cuda")
    first = one_update(model, optimizer, x, y, x*.98, "M2")
    assert np.isfinite(first["total_loss"]) and first["gradient_norm"] > 0
    path = tmp_path / "cuda_state.pt"
    _checkpoint(path, model, optimizer, sampler, augment, 1, "M2", 77, "seal")
    one_update(model, optimizer, x, y, x*.98, "M2")
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    other = new_model("M2").cuda()
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=.001)
    other_sampler = np.random.default_rng(3)
    other_augment = torch.Generator().manual_seed(3)
    assert restore_checkpoint(path, other, other_optimizer, other_sampler,
                              other_augment, "M2", 77, "seal") == 1
    one_update(other, other_optimizer, x, y, x*.98, "M2")
    assert all(torch.equal(other.state_dict()[key], value) for key, value in expected.items())

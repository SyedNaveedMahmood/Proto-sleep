from __future__ import annotations

import torch

from protosleep.morphmae_representation import (
    TARGET_NAMES,
    morphology_targets,
    patch_mask_view,
    summarize_mrcnn,
)


def test_morphology_targets_shape_finite_and_amplitude_robust():
    torch.manual_seed(7)
    t = torch.arange(3000, dtype=torch.float32) / 100.0
    base = (
        torch.sin(2 * torch.pi * 1.2 * t)
        + 0.4 * torch.sin(2 * torch.pi * 10.0 * t)
        + 0.2 * torch.sin(2 * torch.pi * 13.5 * t)
    )
    x = torch.stack([base, base + 0.02 * torch.randn_like(base)], dim=0).unsqueeze(1)

    a = morphology_targets(x)
    b = morphology_targets(3.0 * x)

    assert a.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(a).all()
    assert torch.allclose(a, b, atol=2e-4, rtol=2e-4)


def test_patch_mask_view_masks_exact_patch_count_without_changing_shape():
    torch.manual_seed(11)
    x = torch.ones(4, 1, 3000)
    view, mask = patch_mask_view(x, patch_size=25, mask_ratio=0.30)

    assert view.shape == x.shape
    assert mask.shape == (4, 120)
    assert torch.equal(mask.sum(dim=1), torch.full((4,), 36, dtype=torch.long))
    reshaped = view.reshape(4, 1, 120, 25)
    assert torch.all(reshaped[mask[:, None, :, None].expand_as(reshaped)] == 0)


def test_summarize_mrcnn_returns_mean_and_std_channels():
    torch.manual_seed(3)
    afr = torch.randn(5, 30, 80)
    summary = summarize_mrcnn(afr)
    assert summary.shape == (5, 60)
    assert torch.allclose(summary[:, :30], afr.mean(dim=-1))
    assert torch.allclose(summary[:, 30:], afr.std(dim=-1, unbiased=False))

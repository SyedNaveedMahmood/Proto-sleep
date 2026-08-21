import torch

from protosleep.losses import masked_spherical_barycenter_loss


def test_geometry_loss_accepts_half_logits_with_float32_geometry():
    """Regression test for CUDA-autocast-style Half logits + FP32 targets/vectors."""
    batch, time, k, dim = 2, 8, 48, 30

    logits = torch.randn(batch, time, k, dtype=torch.float16, requires_grad=True)

    target = torch.rand(batch, time, k, dtype=torch.float32)
    target = target / target.sum(dim=-1, keepdim=True)

    mask = torch.zeros(batch, time, dtype=torch.bool)
    mask[:, 2:6] = True
    valid = torch.ones(batch, time, dtype=torch.bool)

    prototype_vectors = torch.randn(k, dim, dtype=torch.float32)

    loss = masked_spherical_barycenter_loss(
        logits,
        target,
        mask,
        valid,
        prototype_vectors,
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad.float()).all()

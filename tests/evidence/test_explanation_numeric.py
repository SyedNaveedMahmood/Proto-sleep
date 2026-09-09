import torch

from mist_evidence.explain import (
    SCORE_RECONSTRUCTION_ATOL,
    score_reconstruction_close,
)


def test_score_reconstruction_accepts_small_fp32_reduction_residual():
    reference = torch.tensor([0.7, -0.4, 1.2, 0.0, -1.5], dtype=torch.float32)
    reconstructed = reference.clone()
    reconstructed[2] += 3.2e-5
    assert score_reconstruction_close(reconstructed, reference)


def test_score_reconstruction_rejects_material_mismatch():
    reference = torch.zeros(5, dtype=torch.float32)
    reconstructed = reference.clone()
    reconstructed[0] = 10 * SCORE_RECONSTRUCTION_ATOL
    assert not score_reconstruction_close(reconstructed, reference)


def test_score_reconstruction_rejects_nonfinite_and_shape_mismatch():
    reference = torch.zeros(5, dtype=torch.float32)
    assert not score_reconstruction_close(torch.zeros(4), reference)
    bad = reference.clone()
    bad[0] = float("nan")
    assert not score_reconstruction_close(bad, reference)

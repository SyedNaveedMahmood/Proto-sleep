"""Descriptive representation moments; no target fitting or normalization."""
import numpy as np


def summarize(embedding, usage):
    z, q = np.asarray(embedding), np.asarray(usage)
    if z.ndim != 2 or q.shape != (len(z), 64) or not len(z) or not np.isfinite(z).all() or not np.isfinite(q).all():
        raise ValueError("invalid representation geometry")
    if (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError("invalid prototype assignment distribution")
    mean_usage = q.mean(0)
    return {"embedding_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
            "embedding_feature_variance": float(z.var(0).mean()),
            "effective_prototype_usage": float(np.exp(-(mean_usage * np.log(mean_usage.clip(1e-12))).sum())),
            "dead_prototypes_below_001": int((mean_usage < .001).sum()),
            "epoch_mean_assignment_entropy": float(-(q * np.log(q.clip(1e-12))).sum(1).mean()),
            "source_distance": None, "source_distance_reason": "requires authenticated frozen source-fit reference moments"}

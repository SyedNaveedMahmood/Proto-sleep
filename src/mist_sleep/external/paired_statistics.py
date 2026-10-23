"""Paired recording estimand; seeds remain repeated training realizations."""
from __future__ import annotations

import numpy as np


def paired_effect(m0, m2, recording_ids, *, participant_groups=None, draws=10000, seed=20261023):
    a, b = np.asarray(m0, dtype=np.float64), np.asarray(m2, dtype=np.float64)
    ids = np.asarray(recording_ids)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(ids) != len(a) or len(a) < 2:
        raise ValueError("one row per recording, two registered seed columns required")
    if len(set(ids.tolist())) != len(ids) or not np.isfinite([a, b]).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("invalid paired recording units")
    if draws != 10000 or seed != 20261023:
        raise ValueError("bootstrap configuration differs from preregistration")
    groups = ids if participant_groups is None else np.asarray(participant_groups)
    if groups.shape != ids.shape or any(v is None or str(v) == "" for v in groups.tolist()):
        raise ValueError("verified complete participant mapping required")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent bootstrap clusters required")
    delta = b - a
    cluster_sums = np.stack([delta[groups == g].sum(0) for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        selected = rng.integers(0, len(unique), len(unique))
        # Preserve equal-recording weighting even with unequal-size participant clusters.
        values[i] = (cluster_sums[selected].sum(0) / counts[selected].sum()).mean()
    return {"recordings": len(a), "clusters": len(unique),
            "known_participants": len(unique) if participant_groups is not None else None,
            "bootstrap_unit": "participant" if participant_groups is not None else "recording",
            "m0_mean_by_seed": a.mean(0).tolist(), "m2_mean_by_seed": b.mean(0).tolist(),
            "per_seed_effect": delta.mean(0).tolist(), "effect": float(delta.mean()),
            "interval_95": np.quantile(values, [.025, .975]).tolist(), "bootstrap_draws": draws,
            "seed_direction_agreement": bool(np.sign(delta.mean(0)[0]) == np.sign(delta.mean(0)[1])),
            "between_seed_effect_sd": float(delta.mean(0).std(ddof=1)),
            "leave_one_record_out_effect_range": [float(v) for v in (
                ((delta.sum(0) - delta) / (len(a) - 1)).mean(1).min(),
                ((delta.sum(0) - delta) / (len(a) - 1)).mean(1).max())]}

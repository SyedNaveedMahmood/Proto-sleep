"""Explicit public field allowlist; detailed vectors and rows stay private."""
from .protocol import digest

PUBLIC_FIELDS = {"status", "decision", "flag", "reason", "recordings", "known_participants", "effect",
                 "interval_95", "per_seed_effect", "m0_mean_by_seed", "m2_mean_by_seed", "clusters",
                 "bootstrap_unit", "bootstrap_draws", "seed_direction_agreement", "between_seed_effect_sd",
                 "leave_one_record_out_effect_range"}


def public_aggregate(value):
    unknown = set(value) - PUBLIC_FIELDS
    if unknown:
        raise ValueError("nonaggregate fields rejected from publication")
    # Do not stringify arbitrary exception messages or file paths as public reasons.
    for key in ("reason", "status", "decision", "flag"):
        if isinstance(value.get(key), str) and ("/" in value[key] or "\\" in value[key]):
            raise ValueError("private path-like value rejected")
    digest(value)  # Reject nonfinite/non-JSON values too.
    return dict(value)

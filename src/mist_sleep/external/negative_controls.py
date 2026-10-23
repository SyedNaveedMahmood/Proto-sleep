"""Generated analytical falsification evidence; never clinical outcomes."""
import numpy as np
from .stage_metrics import stage_metrics
from .paired_statistics import paired_effect
from .protocol import scientific_decision


def generated_controls():
    y = np.tile(np.arange(5), 100)
    logits = np.eye(5)[y] * 10
    perfect = stage_metrics(y, logits)["macro_f1"]
    permuted = stage_metrics(y, np.roll(logits, 1, axis=1))["macro_f1"]
    shuffled = stage_metrics(np.random.default_rng(34).permutation(y), logits)["macro_f1"]
    a = np.ones((3, 2)) * .4
    paired = paired_effect(a, a + .03, ["generated_a", "generated_b", "generated_c"])
    blocked = scientific_decision(None, [], None, integrity_passed=False)["decision"]
    return {"kind": "GENERATED_ONLY", "perfect_fixed_five_f1": perfect,
            "stage_order_permutation_f1": permuted, "shuffled_label_f1": shuffled,
            "constant_paired_effect": paired["effect"], "constant_paired_CI": paired["interval_95"],
            "missing_integrity_decision": blocked,
            "clinical_inference": False,
            "pass": bool(perfect == 1 and permuted == 0 and .1 < shuffled < .3 and
                         np.allclose(paired["interval_95"], [.03, .03]) and blocked == "PRIMARY_EVALUATION_BLOCKED")}

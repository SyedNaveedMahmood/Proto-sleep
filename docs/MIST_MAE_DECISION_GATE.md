# MIST MAE decision gate

Date: 2026-08-23

## Recovered historical A3/A4 fold-0 result

The recovered WaveSleepNet experiment used the audited historical ProtoPNet architecture,
exact archived `OneFoldTrainer.protop_loss`, historical Adam settings, the documented
27->30 MorphMAE compatibility patch, a fold-specific leakage-safe MorphMAE-v2 checkpoint,
and validation Macro-F1 checkpoint selection. The designated test subject was not opened.

Frozen fold-0, three-supervised-seed result:

| seed | A3_legacy_recovered | A4_legacy_recovered | A4-A3 |
|---:|---:|---:|---:|
| 123 | 0.8163 | 0.8134 | -0.0029 |
| 456 | 0.8235 | 0.8147 | -0.0088 |
| 789 | 0.8082 | 0.8030 | -0.0052 |
| mean | 0.816019 | 0.810371 | -0.005648 |

A4 was below A3 for all three supervised seeds. A repeated execution with the same seeds
reproduced the same values; that is a reproducibility check, not an additional independent
experiment.

The earlier two-epoch development smoke had A4-A3 = +0.0083. It is not a scientific result
and demonstrates why transient early-epoch ordering must not be used to judge the mechanism.

## Decision

Do not expand recovered historical A3/A4 to additional folds and do not tune learning rates,
freezing schedules, prototype weights, or patience after observing this result. Combined with
the already completed five-fold current-prototype screen, there is no stable evidence that
MorphMAE initialization improves the prototype classifier.

This does not establish that MorphMAE itself is useless. The remaining causal question is
whether MorphMAE helps plain AttnSleep independently of the prototype pathway.

## Frozen next gate: A1 vs A2

Run a matched five-fold x three-supervised-seed validation-only screen:

- `A1`: AttnSleep with random MRCNN initialization.
- `A2`: byte-identical AttnSleep initialization outside MRCNN, with only MRCNN replaced by the
  fold-specific leakage-safe MorphMAE-v2 checkpoint.
- Use the current AttnSleep supervised training protocol and validation Macro-F1 selection.
- Do not open designated test-subject NPZ arrays and do not compute test metrics.

Predeclared screening rule, fixed before viewing A1/A2 results:

1. Mean fold-level `A2 - A1` must be positive.
2. `A2 > A1` must hold in at least 3 of 5 folds.

Interpretation:

- If the gate passes, MorphMAE has a reproducible AttnSleep benefit but interacts poorly with
  the prototype objective. The MAE contribution can remain, but the MAE+prototype mechanism
  should not be claimed.
- If the gate fails, stop treating MorphMAE as a main downstream staging contribution. Keep it
  as a negative/ablation result and pivot the main transfer story to the prototype pathway or
  another independently supported mechanism.

External transfer remains downstream of this decision gate.

# BHI 2026 one-page abstract: final experiment gate

## Evidence before this gate

The clean fold-specific MorphMAE-v2 initialization does **not** show stable downstream benefit under the current supervised AttnSleep protocol:

- matched A1 vs A2, 5 folds x 3 supervised seeds: mean fold-level `A2 - A1 = -0.005461`;
- positive folds: 2/5;
- therefore the predeclared MorphMAE gate failed.

The current spherical prototype pathway had previously shown a small positive A3-vs-A1 effect in 4/5 development folds, while both current and recovered-historical A4-vs-A3 tests failed to show stable MorphMAE/prototype synergy.

No further MorphMAE tuning, freezing schedule search, WCO recovery, or prototype-loss rescue search is allowed for the one-page abstract. Those would be post-hoc after multiple negative MAE gates.

## Final targeted experiment

Run a matched A1-vs-A3_current prototype comparison across all 20 rotating Sleep-EDF-20 folds and three supervised seeds.

The comparison is strengthened relative to the earlier five-fold screen:

- A1 and A3 have byte-identical initialized MRCNN state;
- A1 and A3 have byte-identical initialized TCE state;
- A1 and A3 have byte-identical initialized classifier state;
- the additional initialized state in A3 is the current prototype pathway;
- both members are trained with the same supervised seed and fresh loaders;
- only train and validation NPZ files are opened for each fold;
- that fold's designated test subject is not evaluated.

Primary endpoint: seed-average within fold first, then analyze fold-level `A3 - A1` validation Macro-F1.

### Frozen screening criterion

For a positive prototype claim in the abstract:

1. mean fold-level `A3 - A1 > 0`; and
2. `A3 > A1` in at least 12 of 20 folds.

A subject-bootstrap 95% interval and exact sign-test value are saved as supporting descriptive statistics, not as a replacement for the frozen criterion.

Per-stage F1, learned prototype beta, effective prototype count, and maximum off-diagonal prototype cosine are supporting analyses from the same trained models. They are not additional tuning experiments.

## Stopping rule

After this 20-fold matched prototype screen, training for the one-page abstract stops. The abstract will report the result as observed, whether the criterion passes or fails. Any external-transfer, MAE-recovery, WCO, or hyperparameter-search work belongs to a later full-paper phase.

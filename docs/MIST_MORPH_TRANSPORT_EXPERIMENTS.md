# MIST-Morph v2 experiment registry

This file is a preregistered development plan for the transport branch. It does not contain results unless explicitly marked later.

## T0 software verification

Required before real-data training:

- existing `tests/evidence` suite passes;
- transport source marginals reconstruct `1/P` within FP32 tolerance;
- transport gradients are finite;
- flat/high-cost windows can use background;
- anchor buffers remain immutable;
- additive score decomposition remains exact within the documented FP32 tolerance;
- synthetic CUDA smoke completes both `transport` and `evidence` and exports both explanation types.

Any failure stops the run.

## T1 real-PC integration smoke

Data: current Sleep-EDF-20 development fold 0. Because original epoch indices are missing, use `--epoch-only` unless preprocessing provenance is later repaired.

Seed: 123.

Budget: exactly 2 epochs, patience 2.

Variants:

1. `transport`
2. `evidence`
3. `summary_only`
4. `raw_context`

Purpose: execution, runtime and numerical behavior only. Do not use T1 to change epsilon, rho, background cost, background prior, scales, bank size, or learning rate.

Inspect:

- finite losses and gradients;
- validation predictions not collapsed to one class;
- transport source-marginal error;
- background mass is finite and not identically zero or one;
- explanation HTML contains actual source waveforms and nonempty transport assignments;
- runtime per epoch.

## T2 fixed local morphology comparison

Only after T1 passes. This is still development, not independent confirmation.

Initial plan: folds 0-4, supervised seeds 123/456/789, epoch-only, same training budget/early stopping for every variant. Compare `transport`, `evidence`, `summary_only`, and `raw_context`.

Average seeds within each fold/validation subject first. Report all folds and per-stage F1.

Primary v2-versus-v1 quantity: fold-level difference in validation Macro-F1, `transport - evidence`.

Secondary questions:

- `transport - summary_only`: do source-waveform correspondences add value?
- `transport - raw_context`: what accuracy is paid for the evidence constraint?
- assignment diagnostics: background mass, anchor usage entropy, effective anchor count, and concentration of mass by source subject.

No transport hyperparameter sweep is allowed before this fixed comparison.

## T3 transport mechanism ablations

Run only if T2 shows that transport is at least mechanically credible.

One-at-a-time registered ablations:

- no learned local distance (`neural_fraction_cap=0`);
- random rather than medoid anchors;
- transport without summary/amplitude evidence;
- background-disabled diagnostic only (not a candidate final model);
- v1 independent similarity versus v2 transport using identical banks and seeds.

A component remains only if it has either a reproducible predictive benefit or a clear independently measured explanation/robustness benefit. Simplicity wins ties.

## T4 morphology validity

No anchor receives a clinical event name from the training sleep-stage label. Before saying spindle, K-complex, slow wave, alpha burst, etc.:

- obtain independent event annotations or expert review;
- sample anchors/matches using a fixed policy that includes errors and artifacts;
- measure agreement across subjects;
- evaluate event interval precision/recall or an equivalent prespecified measure;
- retain negative examples and reviewer disagreements.

If this fails, retain neutral anchor IDs and call the model source-waveform evidence based, not clinically event interpretable.

## T5 temporal architecture

The current NPZ files lack verified original epoch indices. Do not test temporal context by assuming continuity.

Recover original retained epoch indices from preprocessing provenance first. Then compare, with the local morphology model fixed:

- radius 0, no CRF;
- additive context only;
- CRF only where technically meaningful;
- additive context + CRF.

Sequence components are dropped if they hurt minority-stage F1 or boundary/transition performance despite improving pooled accuracy.

## T6 external transfer / SOTA evaluation

Sleep-EDF-20 is development data after repeated adaptive use. Freeze code, training recipe, anchor construction, seeds, preprocessing, channel policy, and model-selection rule before external evaluation.

Use a genuinely external cohort such as SHHS or another appropriate dataset with a matched single-EEG protocol. Compare against reproduced, eligible strong baselines under the same modality and test subjects. Report subject-level uncertainty, seed variability separately, per-stage F1, Macro-F1, accuracy, kappa, calibration, boundary performance and runtime.

SOTA may be claimed only if the strongest eligible matched reference is actually exceeded under the frozen independent evaluation. Novelty and SOTA are separate claims.

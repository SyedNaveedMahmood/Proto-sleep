# MorphSpec-R1 held-out-fold confirmation

## Why this run exists

MorphSpec-R1 was selected on development folds 0-4 after its frozen MRCNN probe beat a matched frozen random MRCNN in all five development folds. The recipe is now frozen. No further changes to its label-free target set, mask ratio, learning rates, epoch count, or downstream probe protocol are allowed before this confirmation.

## Confirmation folds

Use rotating Sleep-EDF-20 folds 5-9. These fold roles were not used to select MorphSpec-R1. Each fold receives a newly trained leakage-safe MorphMAE-v2 checkpoint, followed by the frozen MorphSpec-R1 refinement using only that fold's SSL training subjects.

This is a held-out-fold confirmation within Sleep-EDF-20, not an external-dataset test and not a claim of independent unseen subjects across the entire project.

## Matched frozen probe

For every fold and supervised seed, construct three AttnSleep models with byte-identical initialized TCE and classifier state:

- `random_probe`: random MRCNN, frozen;
- `original_mae_probe`: fold-specific MorphMAE-v2 MRCNN, frozen;
- `morphspec_probe`: fold-specific MorphSpec-R1 MRCNN, frozen.

Only TCE/classifier parameters are trained. Frozen MRCNN BatchNorm statistics remain in evaluation mode. Train and validation files are opened; the designated test-subject NPZ files are not opened or evaluated.

## Primary endpoint and frozen gate

Average the three supervised seeds within each fold first. The primary effect is fold-level validation Macro-F1:

`morphspec_probe - random_probe`

The frozen confirmation gate passes only if:

1. mean held-out-fold `MorphSpec - random > 0`; and
2. the effect is positive in at least 60% of the held-out folds (3 of 5 for folds 5-9).

`MorphSpec - original MorphMAE` is a secondary mechanistic comparison. Exact sign-test values are descriptive because n=5 folds is small.

## Stopping rule

Do not tune MorphSpec-R1 on folds 5-9. If the frozen confirmation gate passes, the representation result is strong enough for the one-page BHI abstract, subject to careful wording that it is within-dataset held-out-fold validation. If it fails, report the failure and do not rescue it with a new recipe on these confirmation folds.

External-transfer claims still require a genuinely external dataset.

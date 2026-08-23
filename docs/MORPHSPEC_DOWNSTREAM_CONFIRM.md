# MorphSpec-R1 downstream confirmation gate

## Status before this gate

MorphSpec-R1 was designed on folds 0-4 and passed its frozen representation confirmation on folds 5-9:

- mean frozen-probe `MorphSpec - random = +0.077658` Macro-F1;
- positive in 5/5 representation-confirmation folds;
- mean frozen-probe `MorphSpec - original MorphMAE-v2 = +0.059716`;
- positive in 5/5 folds.

No test metric was computed and designated test-subject NPZ files were not opened.

## Why use new folds again?

Folds 5-9 have now been observed for the representation-level confirmation. The downstream-transfer question is therefore evaluated on folds 10-14 so its outcome is not selected on the same folds that established the frozen-probe result.

## Frozen downstream models

Within each fold and supervised seed:

1. `A1_standard`: random AttnSleep MRCNN, standard supervised AttnSleep training.
2. `original_stage`: leakage-safe original MorphMAE-v2 MRCNN, frozen for five epochs and then unfrozen at encoder LR `1e-4`; TCE/classifier LR remains `1e-3`.
3. `morphspec_stage`: MorphSpec-R1 MRCNN, exactly the same staged transfer recipe as `original_stage`.

All three models begin with byte-identical non-MRCNN initialization. `original_stage` and `morphspec_stage` therefore isolate the representation objective under the same downstream transfer protocol.

The staged recipe is `MAE_STAGE_1E4`, selected during the earlier transfer-recovery development experiment. No new transfer hyperparameter search is permitted in folds 10-14.

## Frozen decision criteria

Primary practical endpoint:

`morphspec_stage - A1_standard` validation Macro-F1, averaging supervised seeds within fold first.

Primary gate passes when:

- mean fold-level delta is positive; and
- MorphSpec is positive in at least 60% of folds (3/5 for the default run).

Secondary causal endpoint:

`morphspec_stage - original_stage` validation Macro-F1.

The same positive-mean and >=60%-positive-fold rule is applied. The full downstream gate passes only if both primary and secondary gates pass.

Bootstrap intervals and exact sign-test values are descriptive supporting statistics, not replacements for the frozen gate.

## Leakage / claim boundary

Only train and validation recordings are opened for each fold. The designated test subject is not evaluated. This experiment does not support an external-transfer claim.

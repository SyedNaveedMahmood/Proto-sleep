# MorphSpec-R1 frozen-transfer confirmation

## Why this experiment exists

MorphSpec-R1 passed the held-out frozen-representation confirmation on folds 5-9, but the
pre-frozen staged-unfreeze transfer recipe failed on new folds 10-14. Both original
MorphMAE-v2 and MorphSpec-R1 were below fully trainable A1 under staged unfreezing.

The mechanistic hypothesis is therefore narrower and directly motivated by observed data:
MorphSpec contains useful stage-relevant information when the MRCNN is frozen, and
unfreezing the encoder destroys or misaligns that useful representation. The final
Sleep-EDF-20 development experiment tests frozen encoder transfer rather than another
learning-rate or unfreezing search.

## Frozen protocol

Evaluation folds: **15,16,17,18,19**. These rotating validation folds were not used for
MorphSpec objective design (0-4), frozen representation confirmation (5-9), or the staged
transfer result (10-14).

Supervised seeds: **123,456,789**.

Within each fold/seed:

- `A1_standard`: random MRCNN; standard fully trainable AttnSleep.
- `random_frozen`: random MRCNN frozen/eval; train TCE + classifier only.
- `original_frozen`: fold-specific leakage-safe MorphMAE-v2 MRCNN frozen/eval; train TCE + classifier only.
- `morphspec_frozen`: fold-specific MorphSpec-R1 MRCNN frozen/eval; train TCE + classifier only.

All four models have matched non-MRCNN initialization within fold/seed. MorphSpec and
original checkpoints must match the fold training-subject metadata. Only train + validation
recordings are opened; designated test-subject files remain unopened.

## Endpoints frozen before results

Primary:

`MorphSpec-frozen - A1-standard` validation Macro-F1.

Secondary objective isolation:

`MorphSpec-frozen - original-MorphMAE-frozen` validation Macro-F1.

Representation control:

`MorphSpec-frozen - random-frozen` validation Macro-F1.

For each endpoint, the directional gate is:

- mean fold-level delta > 0, and
- positive in at least 3/5 folds.

The full frozen-transfer gate requires all three endpoints to pass. Bootstrap intervals and
an exact two-sided sign test are descriptive supporting summaries; with n=5 rotating folds,
formal significance is not the primary decision criterion.

## Interpretation policy

If the primary gate passes, MorphSpec supports a parameter-efficient downstream claim:
useful SSL features transfer when the encoder is retained rather than overwritten.

If MorphSpec beats the frozen controls but not fully trainable A1, the supported claim is
representation quality, not improved end-to-end staging accuracy.

If the representation control fails, do not continue tuning on Sleep-EDF-20. At that point
all 20 rotating folds have been observed in one of the sequential development gates.

No result from this experiment permits an external-dataset transfer claim or test-set claim.

# MorphMAE transfer recovery: theory-driven development gate

## Why this gate exists

The clean matched A1/A2 experiment on folds 0..4 found that directly replacing AttnSleep's random MRCNN with fold-specific MorphMAE-v2 and then applying the ordinary supervised optimizer did not improve validation Macro-F1 on average.

That result answers a narrow question: **naive end-to-end fine-tuning of the MorphMAE MRCNN is not beneficial under the current downstream recipe.** It does not yet distinguish between two different explanations:

1. MorphMAE did not learn stage-relevant features; or
2. MorphMAE learned useful features, but the downstream optimizer erases them too quickly.

The current supervised trainer applies one Adam learning rate (`1e-3`) and one weight decay (`1e-3`) to all parameters from the first update. MorphMAE-v2 itself was pretrained with learning rate `2e-4`. Thus the pretrained MRCNN is immediately updated at 5x its SSL learning rate, together with BatchNorm running statistics.

This gate separates representation quality from transfer retention before changing the SSL objective.

## Fixed development experiments

Development folds: 0,1,2,3,4.

Each fold uses the existing leakage-safe, fold-specific MorphMAE-v2 checkpoint and three supervised seeds (123,456,789). The designated test-subject NPZ files are not opened.

Four models are trained from matched initializations:

- `random_probe`: random MRCNN frozen; train only TCE + classifier.
- `mae_probe`: MorphMAE MRCNN frozen; train only TCE + classifier.
- `mae_stage_1e4`: freeze MorphMAE MRCNN for five epochs, then unfreeze it at `1e-4`; TCE + classifier remain at `1e-3`.
- `mae_stage_2e5`: freeze MorphMAE MRCNN for five epochs, then unfreeze it at `2e-5`; TCE + classifier remain at `1e-3`.

While frozen, the MRCNN is kept in evaluation mode so BatchNorm running statistics cannot drift. The optimizer is created only once, so Adam state for TCE/classifier is preserved across the unfreeze point.

The two encoder learning rates are a small theory-driven bracket rather than a general search:

- `1e-4` = 0.5x the MorphMAE-v2 pretraining LR and 0.1x the supervised head LR;
- `2e-5` = 0.1x the pretraining LR and 0.02x the supervised head LR.

## Interpretation

### Representation probe

Primary diagnostic: fold-level `mae_probe - random_probe`.

If the mean is positive and at least 3/5 development folds are positive, MorphMAE contains useful stage information when protected from end-to-end drift.

### Transfer recovery

Compare each staged recipe with the already-completed standard A1 result for the same fold/seed.

The exploratory transfer gate is:

- mean fold-level staged-A2 minus standard-A1 > 0; and
- positive in at least 3/5 development folds.

If a staged recipe passes, freeze that recipe and evaluate it on new folds after generating new fold-specific MorphMAE checkpoints. The development folds cannot be reused as confirmatory evidence for the tuned transfer recipe.

If the frozen representation probe is positive but neither staged recipe passes, the next intervention should preserve the pretrained representation more explicitly (for example a retention regularizer), not redesign SSL immediately.

If the frozen representation probe itself fails, the problem is upstream: MorphMAE-v2 is reconstructive but not sufficiently stage-informative, and the next experiment should redesign the self-supervised objective rather than continue downstream learning-rate tuning.

## What is not allowed in this gate

No WCO, no prototype rescue, no broad LR grid, no fold-specific hand tuning, no test evaluation, and no claim that a development-selected transfer recipe is confirmatory.

# MIST multifold replication

## Why this is the next experiment

Fold 0 produced a positive matched initialization effect for the current prototype model:

- A1 mean Macro-F1: 0.822220
- A3_current mean Macro-F1: 0.813167
- A4_current mean Macro-F1: 0.820646
- A4_current - A3_current: +0.007479
- A4_current - A1: -0.001574

This is evidence worth replicating, not a conclusion. The key mechanism question is whether the MorphMAE-initialized prototype model repeatedly improves over the same prototype model with a random MRCNN. One fold is not enough, and A4_current did not exceed A1 on fold 0.

The next frozen development screen is folds 0-4 with three matched supervised seeds per fold. The designated test subject remains untouched in every fold.

## One-command sequential workflow

The driver intentionally runs MorphMAE pretraining sequentially. It does not launch concurrent GPU jobs.

First inspect all missing folds without starting training:

```bash
python scripts/run_mist_multifold.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0,1,2,3,4 \
  --ssl-seed 1337 \
  --supervised-seeds 123,456,789 \
  --pretrain-output-dir mist_sleep_runs/morphmae_pretrain \
  --stability-output-dir mist_sleep_runs/multifold_stability \
  --prepare-only
```

Then run fold-specific MorphMAE pretraining only:

```bash
python scripts/run_mist_multifold.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0,1,2,3,4 \
  --ssl-seed 1337 \
  --supervised-seeds 123,456,789 \
  --pretrain-output-dir mist_sleep_runs/morphmae_pretrain \
  --stability-output-dir mist_sleep_runs/multifold_stability \
  --pretrain-only
```

A previously completed canonical checkpoint such as fold 0 is validated and skipped. A partial run containing `legacy_output/best_morphmae.pt` but no canonical `encoder.pt` is not silently reused; the driver stops for inspection.

After all five canonical checkpoints exist, run only the matched supervised audit:

```bash
python scripts/run_mist_multifold.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0,1,2,3,4 \
  --ssl-seed 1337 \
  --supervised-seeds 123,456,789 \
  --pretrain-output-dir mist_sleep_runs/morphmae_pretrain \
  --stability-output-dir mist_sleep_runs/multifold_stability \
  --stability-only
```

Alternatively, omit the mode flag to run missing MorphMAE pretraining and then the supervised stability audit in one sequential command.

## Safety checks

Before a fold-specific encoder is accepted, the driver verifies:

- strict AttnSleep MRCNN key/shape compatibility;
- checkpoint `fold` equals the requested fold;
- checkpoint `seed` equals the requested SSL seed;
- checkpoint `train_subjects` exactly equals that fold's 18 training subjects;
- neither validation nor designated test subject appears in SSL training metadata.

The supervised stability runner evaluates validation subjects only. It does not compute test metrics.

## How to interpret the five-fold screen

Treat folds/subjects, not the three supervised seeds, as the experimental units. Average the three seeds within each fold first, then inspect the fold-level `A4_minus_A3` values.

A positive mean across folds with the effect positive in a majority of folds would justify moving to a broader replication / external-transfer stage. A near-zero or inconsistent effect would mean the historical one-fold A4 mechanism has not stabilized under the clean implementation. This screen is not, by itself, a cross-dataset transfer result.

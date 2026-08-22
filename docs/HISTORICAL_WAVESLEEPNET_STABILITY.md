# Recovered historical WaveSleepNet A3/A4 stability protocol

This protocol is used only after both historical-recovery compatibility gates pass:

1. `smoke_legacy_wavesleepnet.py` proves the audited `ProtoPNet` can be instantiated with the documented MorphMAE `27 -> 30` channel bridge and that A3/A4 differ only in MRCNN initialization.
2. `smoke_legacy_wavesleepnet_objective.py` proves the exact archived `OneFoldTrainer.protop_loss` is finite and differentiable for both A3 and A4 on real Sleep-EDF training EEG.

The full runner is:

```bash
python scripts/run_legacy_mist_stability.py \
  --legacy-root /path/to/MorphMAE_Sleep_Codebase \
  --data-dir /path/to/Sleep-EDF-20-npz \
  --folds 0 \
  --seeds 123,456,789 \
  --mae-checkpoint /path/to/fold_00/ssl_seed_1337/encoder.pt \
  --output-dir mist_sleep_runs/legacy_fold0_stability
```

## What is historical

The recovered experiment uses the frozen archived WaveSleepNet `ProtoPNet`, the exact archived `OneFoldTrainer.protop_loss`, and the audited Sleep-EDF training hyperparameters (Adam, batch size 64, LR `5e-4`, weight decay `1e-4`, max epochs 5000, patience 50). Training remains FP32. No gradient clipping or AMP is introduced.

A3 and A4 are cloned from the same initialized template. A4 alone receives the fold-specific MorphMAE-pretrained MRCNN. The runner fails if the MorphMAE checkpoint's declared training subjects do not exactly match the fold training subjects.

## What is intentionally adapted

This is a recovered mechanism experiment, not a claim that the old end-to-end script is being replayed byte-for-byte. The archived WaveSleepNet loader/trainer also constructs a test loader and contains environment-specific dataset paths. The recovery runner therefore uses the current subject-disjoint fold definition and opens only train and validation NPZ files.

Checkpoint selection is by validation Macro-F1, matching the current MIST stability audit. This selection rule is recorded in every checkpoint/CSV/provenance file as an explicit protocol adaptation rather than being described as historical trainer behavior.

## Leakage barrier

For each fold, the runner resolves train/validation/test subject IDs from NPZ filenames before opening signal arrays. It then opens only train and validation files. Designated test-subject NPZ files are not opened and no test metrics are computed.

## Interpretation gate

The first run is fold 0 with three supervised seeds. The primary mechanism quantity is:

```text
A4_legacy_recovered - A3_legacy_recovered
```

A positive mean across the three supervised seeds is only a fold-0 screening signal. It does not establish external transfer, and it does not replace a multi-fold replication. If fold 0 is promising, extend the same frozen protocol to additional folds using the existing fold-specific MorphMAE checkpoints before any external-dataset claim.

Do not use `--max-epochs` or `--patience` for the frozen screen. Those flags exist only for development/runtime debugging and are explicitly marked as protocol overrides in the output.

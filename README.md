# Proto-sleep

Research code for prototype-guided single-channel EEG sleep staging, including the completed ProtoMAE development branch and a leakage-safe recovery of the original MIST/MorphMAE mechanism experiment.

> Research status: transition-aware macro masking is not a confirmed performance contribution. The current priority is the original MIST question: whether MorphMAE initialization improves prototype-guided representations when the comparison is repeated with clean subject-level splits.

## Repository layout

```text
Proto-sleep/
├── src/protosleep/
│   ├── config.py
│   ├── data.py
│   ├── attnsleep.py
│   ├── prototypes.py
│   ├── mist.py
│   ├── morphmae_bridge.py
│   ├── micro.py
│   ├── cache.py
│   ├── night.py
│   ├── masking.py
│   ├── macro.py
│   ├── losses.py
│   ├── train_macro.py
│   ├── evaluation.py
│   ├── selftest.py
│   └── runner.py
├── scripts/
│   ├── run_fold.py
│   ├── speed_audit.py
│   ├── inspect_morphmae_checkpoint.py
│   ├── audit_legacy_morphmae.py
│   ├── run_fold_morphmae_pretrain.py
│   └── run_mist_stability.py
├── tests/
│   ├── test_structural.py
│   ├── test_losses.py
│   ├── test_mist_checkpoint.py
│   └── test_morphmae_bridge.py
└── notebooks/
    └── README.md
```

## Install

```bash
git clone https://github.com/SyedNaveedMahmood/Proto-sleep.git
cd Proto-sleep
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

## Data convention

The current Sleep-EDF loader follows the AttnSleep notebook convention:

- one `*.npz` per recording;
- arrays `x` and `y`;
- `(N,3000,1)` is converted to `(N,1,3000)`;
- 30 s epochs at 100 Hz;
- subject ID is parsed from `filename[3:5]`;
- recording/night boundaries are preserved.

No EEG data, checkpoints, run directories, caches, local logs, or generated legacy-audit files should be committed. See `.gitignore`.

## Why the macro experiments train so fast

This is expected and is not mainly an AMP effect. AttnSleep/Proto-AttnSleep repeatedly process raw `[B,1,3000]` EEG epochs, whereas the macro models process cached whole-night `[T,48]` prototype trajectories. With roughly 35 train nights and macro batch size 2, there are only about 18 macro batches per epoch; 50 SSL epochs are roughly 900 optimizer steps through a ~111k-parameter model.

Use `--no-reuse` for timing runs so checkpoints/caches cannot make a rerun appear instantaneous.

## Historical MorphMAE audit result

The historical MorphMAE-v2 checkpoint is strictly compatible with the current AttnSleep MRCNN. The source audit recovered the actual v2 recipe rather than reconstructing it from memory:

- AttnSleep-compatible MRCNN + AFR encoder;
- patch size 25 samples, 120 patches;
- mixed mask spans of 2–24 patches;
- mask schedule 0.50 -> 0.65 -> 0.75;
- time/STFT/derivative/band weights 1.0 / 0.5 / 0.3 / 0.15;
- 100 epochs, batch 128, lr 2e-4, weight decay 1e-2;
- AMP disabled for the validated MorphMAE training because the STFT path must remain FP32.

The old `mae_edf78_v2` config used the full Sleep-EDF-78 NPZ root with `exclude_subject_ids: []`. Therefore that historical checkpoint is useful for compatibility/reconstruction provenance but is **not** a clean checkpoint for a current Sleep-EDF-20 fold comparison.

## Leakage-safe fold-specific MorphMAE-v2 pretraining

`run_fold_morphmae_pretrain.py` executes the verified historical MorphMAE implementation rather than rewriting its decoder/losses. The wrapper changes only runtime split/output fields.

For each fold it:

1. resolves the same AttnSleep 20-subject permutation used by the main repo by inspecting NPZ filenames only;
2. creates a symlink-only NPZ view containing the 18 training subjects;
3. physically excludes the validation and designated test recordings from the legacy trainer's data root;
4. loads and validates the historical `mae_npz_edf78_v2.yaml` signature;
5. copies the historical pretraining shell launcher and injects the generated fold YAML without modifying the legacy codebase;
6. runs historical MorphMAE-v2;
7. exports `encoder.pt`, an exact MRCNN state dict with top-level `train_subjects`, fold, seed, config hashes, legacy-source hash, and source-checkpoint SHA-256.

The legacy trainer may create its own SSL train/validation split, but it can only split the 18-subject train-only view. The current validation subject and designated test subject are unavailable to that process.

### Prepare-only check

```bash
python scripts/run_fold_morphmae_pretrain.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 \
  --seed 1337 \
  --output-dir mist_sleep_runs/morphmae_pretrain \
  --prepare-only
```

Inspect the printed train/validation/test subject IDs and generated config. No training is started in this mode.

### Train fold 0

```bash
python scripts/run_fold_morphmae_pretrain.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 \
  --seed 1337 \
  --output-dir mist_sleep_runs/morphmae_pretrain
```

Expected canonical checkpoint:

```text
mist_sleep_runs/morphmae_pretrain/fold_00/ssl_seed_1337/encoder.pt
```

Inspect it:

```bash
python scripts/inspect_morphmae_checkpoint.py \
  mist_sleep_runs/morphmae_pretrain/fold_00/ssl_seed_1337/encoder.pt
```

The inspector should show strict MRCNN compatibility and `train_subjects` matching the fold's 18 training subjects.

## Matched MIST stability audit

After the fold-specific SSL checkpoint exists, run the matched validation-only comparison:

```bash
python scripts/run_mist_stability.py \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0 \
  --seeds 123,456,789 \
  --mae-checkpoint mist_sleep_runs/morphmae_pretrain/fold_00/ssl_seed_1337/encoder.pt \
  --output-dir mist_sleep_runs/fold0_stability
```

`A3_current` and `A4_current` are initialized from one shared prototype-model template. Every non-MRCNN parameter is verified identical before training; `A4_current` differs only by the strict MorphMAE-pretrained MRCNN initialization. The script evaluates the validation subject only and refuses a checkpoint whose declared pretraining subjects do not match the fold train split.

For a stronger result, repeat with fold-specific MorphMAE checkpoints across multiple folds and, later, multiple SSL seeds. Do not use the old full-EDF78 checkpoint as confirmatory evidence.

## ProtoMAE development runner

```bash
python scripts/run_fold.py \
  --data-dir "/path/to/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0
```

Force fresh training:

```bash
python scripts/run_fold.py \
  --data-dir "/path/to/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 \
  --no-reuse
```

The normal runner builds train/validation feature caches only. Final test evaluation is opt-in and protected by a persistent `TEST_EVALUATED.lock`.

## Speed audit

```bash
python scripts/speed_audit.py
```

## Reproducibility rules

- Subject-level train/validation/test splits are disjoint.
- Recording boundaries are preserved.
- Validation controls development; final test evaluation is explicit and lock-protected.
- MIST checkpoint bridges use exact key/shape matching and `strict=True`.
- Fold-specific MorphMAE pretraining uses a physical train-only NPZ view.
- Generated configs, source/config hashes, split metadata, and checkpoint hashes are stored with each SSL run.
- The old full-EDF78 MorphMAE-v2 checkpoint is not a clean Sleep-EDF-20 fold checkpoint.

## Research status

The historical Phase-II MorphMAE-v2 reconstruction result remains the preferred reconstruction recipe. The current scientific question is whether its initialization benefit for prototype-guided staging replicates across matched seeds/folds. External transfer should be attempted only after that mechanism is stable enough to justify the larger experiment.

# Proto-sleep

Research code for **ProtoMAE-Sleep / prototype-guided Sleep-EDF experiments**, refactored from the development notebooks into an auditable Python package.

> Development status: the code is stable enough for controlled experiments, but the current transition-aware masking hypothesis is **not** a confirmed performance claim. The repository keeps the implementation reproducible while the research question is being refined.

## Why training can look "too fast"

This is expected for the macro experiments and is **not mainly an AMP effect**.

The pipeline has two very different compute regimes:

| Stage | Input seen during training | Approx. parameters | Typical work |
|---|---|---:|---|
| AttnSleep / Proto-AttnSleep | raw `[B,1,3000]` EEG epochs | ~523k / ~524k | hundreds of batches per epoch |
| macro classifier | cached whole-night `[T,48]` prototype trajectories | ~71k | ~18 batches/epoch on 35 nights |
| prototype MAE | cached whole-night `[T,48]` trajectories | ~111k | ~900 optimizer steps for 50 SSL epochs |

For the fold-0 setup, 35 training recordings with `MACRO_BATCH_SIZE=2` means only about **18 macro batches per epoch**. Fifty SSL epochs therefore mean roughly **900 optimizer steps**. A modern GPU can finish this in seconds.

CUDA AMP (`torch.autocast(..., float16)` + `GradScaler`) helps, especially for the raw EEG model, but the main reasons the macro stage is fast are:

1. the expensive CNN/AFR extraction has already been cached;
2. the macro models are tiny;
3. the dataset contains only a few dozen full-night sequences;
4. MAE encodes only visible positions;
5. `REUSE_EXISTING=True` can make reruns nearly instant by loading checkpoints/cache.

The runner prints `[checkpoint-hit]`, `[cache-hit]`, parameter counts, expected optimizer-step scale, and wall-clock timings. It also writes `tables/run_provenance.json`.

For a clean timing run:

```bash
python scripts/run_fold.py \
  --data-dir "/path/to/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 \
  --no-reuse
```

To compare AMP directly, use separate output directories so checkpoints cannot be reused accidentally:

```bash
python scripts/run_fold.py --data-dir "/path/to/data" --fold 0 --no-reuse --output-dir runs/amp
python scripts/run_fold.py --data-dir "/path/to/data" --fold 0 --no-reuse --no-amp --output-dir runs/fp32
```

## Repository layout

```text
Proto-sleep/
├── src/protosleep/
│   ├── config.py          # experiment constants / env-controlled runtime settings
│   ├── data.py            # Sleep-EDF NPZ loading and subject-level folds
│   ├── attnsleep.py       # AttnSleep reproduction
│   ├── prototypes.py      # spherical prototype bank + Proto-AttnSleep
│   ├── mist.py            # strict MAE->MRCNN bridge and matched A3/A4 builders
│   ├── micro.py           # epoch-level metrics/training
│   ├── cache.py           # recording-preserving feature cache
│   ├── night.py           # padded full-night loaders
│   ├── masking.py         # random / transition-aware span masks
│   ├── macro.py           # macro Transformer + masked autoencoder
│   ├── losses.py          # prototype distribution / geometry losses
│   ├── train_macro.py     # macro SSL and supervised fine-tuning
│   ├── evaluation.py      # pooled and transition-window evaluation
│   ├── selftest.py        # structural/optimizer-step checks
│   └── runner.py          # A/B/C/D/E experiment runner
├── scripts/
│   ├── run_fold.py
│   ├── speed_audit.py
│   ├── inspect_morphmae_checkpoint.py
│   └── run_mist_stability.py
├── tests/
│   ├── test_structural.py
│   ├── test_losses.py
│   └── test_mist_checkpoint.py
└── notebooks/
    └── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/SyedNaveedMahmood/Proto-sleep.git
cd Proto-sleep
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install the PyTorch build appropriate for your CUDA driver if the default package is not suitable.

## Data format

The loader intentionally follows the AttnSleep preprocessing convention used during development:

- one `*.npz` per recording;
- `x` and `y` arrays;
- `(N,3000,1)` is converted to `(N,1,3000)`;
- 30 s epochs at 100 Hz;
- Sleep-EDF subject ID is parsed from `filename[3:5]`;
- recording/night boundaries are preserved.

Set the dataset directory either with `--data-dir` or `SLEEP_EDF_NPZ_DIR`. No data or checkpoints are committed to the repository.

## Run

Structural self-tests only:

```bash
python scripts/run_fold.py --self-test-only
```

Fold-0 development run:

```bash
python scripts/run_fold.py --data-dir "/path/to/Preprocessed Sleep-EDF-20 dataset" --fold 0
```

Force fresh training:

```bash
python scripts/run_fold.py --data-dir "/path/to/Preprocessed Sleep-EDF-20 dataset" --fold 0 --no-reuse
```

Optional controls:

```bash
python scripts/run_fold.py --data-dir "/path/to/data" --run-f
python scripts/run_fold.py --data-dir "/path/to/data" --run-g
```

### Test-set safety

The normal runner builds/loads **train and validation caches only**. The designated test cache is not touched unless `--final-test` is explicitly supplied.

After final-test evaluation a persistent `TEST_EVALUATED.lock` is written. Re-evaluation is blocked unless `--allow-test-rerun` is also explicitly supplied.

## Experiments

- **A** — AttnSleep baseline.
- **B** — Proto-AttnSleep local prototype model.
- **C** — supervised full-night prototype Transformer, no SSL.
- **D** — random-span prototype masked autoencoder.
- **E** — transition-aware prototype masked autoencoder.
- **F** — optional continuous AFR-latent MAE control.
- **G** — optional prototype-geometry reconstruction control.

D and E use the same macro architecture and reconstruction objective; the intended controlled difference is the masking-center policy.

## Current development evidence

Do not treat a single fold as a paper result. Multi-fold development checks showed substantial subject-to-subject variance. In the matched 5-fold / 3-seed audit, full-label prototype SSL was close to neutral and transition-aware masking was not consistently better than random masking. Prototype-space MAE was more consistent than the continuous latent-MAE control, but this is still a development observation rather than a final claim.

This is why the repository emphasizes **reproducibility, cache/checkpoint provenance, subject-level splits, and explicit test locking** rather than presenting the current configuration as SOTA.

## Next experiment: MIST mechanism stability before transfer

The original MIST/MorphMAE project identified **prototype + MAE initialization without WCO** as the most promising mechanism. The next valid experiment is therefore to stabilize that interaction before attempting external transfer.

The current repository does **not** claim that its spherical `ProtoAttnSleep` is byte-for-byte identical to the historical WaveSleepNet-derived prototype implementation. For that reason the new audit calls the matched pair `A3_current` and `A4_current`:

- `A3_current`: current prototype model with randomly initialized AttnSleep MRCNN.
- `A4_current`: the exact same non-MRCNN initialization, but its MRCNN is replaced by a strictly compatible MAE-pretrained MRCNN checkpoint.

The A3/A4 builder uses one template plus `deepcopy`, verifies every non-MRCNN tensor is identical, performs an exact key/shape match for the pretrained MRCNN, loads with `strict=True`, and records SHA-256 provenance. It refuses to continue if the checkpoint manipulation is not actually active.

### 1. Find the historical MAE/MorphMAE checkpoint

```bash
find ~/Desktop -type f \
  \( -iname '*morph*mae*.pt' -o -iname '*morph*mae*.pth' \
     -o -iname '*mae*encoder*.pt' -o -iname '*mae*encoder*.pth' \) \
  -print
```

### 2. Inspect it before training

```bash
python scripts/inspect_morphmae_checkpoint.py /path/to/checkpoint.pt
```

A valid checkpoint prints `STRICT MRCNN COMPATIBILITY: PASS`, along with the detected state-dict container/prefix and file SHA-256.

Optionally convert it to a canonical MRCNN-only checkpoint:

```bash
python scripts/inspect_morphmae_checkpoint.py /path/to/checkpoint.pt \
  --write-canonical /path/to/morphmae_mrcnn_canonical.pt
```

### 3. Run a fold-0 / three-seed validation-only stability audit

```bash
python scripts/run_mist_stability.py \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0 \
  --seeds 123,456,789 \
  --mae-checkpoint /path/to/morphmae_mrcnn_canonical.pt \
  --output-dir mist_sleep_runs/fold0_stability
```

By default the runner requires the MAE checkpoint to declare `train_subjects`, `pretrain_subjects`, or `subjects` metadata matching the current fold's training subjects. This prevents accidentally using an SSL encoder pretrained on the validation/test subject.

Very old checkpoints may not contain split metadata. For an explicitly development-only pilot you may bypass only the *missing-metadata* check:

```bash
python scripts/run_mist_stability.py \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0 \
  --seeds 123,456,789 \
  --mae-checkpoint /path/to/checkpoint.pt \
  --allow-unverified-pretrain-split \
  --output-dir mist_sleep_runs/fold0_stability_unverified
```

Do **not** use an unverified checkpoint for a final scientific claim. If subject metadata exists and disagrees with the fold, the runner always aborts.

### 4. Multi-fold stability requires fold-specific MAE pretraining

A leakage-safe multi-fold A4 comparison requires a source-only MAE checkpoint for each fold. Once those exist:

```bash
python scripts/run_mist_stability.py \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0,1,2 \
  --seeds 123,456,789 \
  --mae-checkpoint-pattern "/path/to/mae/fold_{fold:02d}/encoder.pt" \
  --output-dir mist_sleep_runs/a1_a3_a4_stability
```

If each supervised seed also has its own SSL checkpoint, `{seed}` may be included in the pattern.

The stability script evaluates **validation subjects only**. Designated test subjects are never placed in an evaluation loader. It writes per-run provenance JSON, raw results, seed-averaged fold results, and paired A3/A4 effects.

## Speed audit

Without touching the dataset:

```bash
python scripts/speed_audit.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The structural tests cover AttnSleep shapes, prototype simplex behavior, masking, MAE forward paths, geometry loss, padding, actual optimizer parameter updates, AMP geometry dtype handling, and strict MAE-to-MRCNN checkpoint mapping.

## Reproducibility notes

- Random seeds are fixed per fold/run.
- Train/validation/test subjects are disjoint.
- Recording boundaries are preserved.
- Prototype transition statistics are fit from training nights only.
- Validation controls model development.
- Final test evaluation is opt-in and lock-protected.
- `--no-reuse` should be used for wall-clock benchmarking.
- The MIST stability audit records checkpoint SHA-256 and MRCNN state digests.
- Fold-specific MAE pretraining must remain subject-disjoint from validation/test subjects.

## Notebook status

The old one-cell and multi-cell notebooks were useful development artifacts, but they are not the canonical implementation anymore. The package under `src/protosleep/` is the source of truth.

## Citation / paper status

This is active research code. A formal citation will be added when the method and evaluation protocol are frozen.

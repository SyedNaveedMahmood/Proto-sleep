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
│   └── speed_audit.py
├── tests/
│   └── test_structural.py
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

The structural tests cover AttnSleep shapes, prototype simplex behavior, masking, MAE forward paths, geometry loss, padding, and actual optimizer parameter updates.

## Reproducibility notes

- Random seeds are fixed per fold.
- Train/validation/test subjects are disjoint.
- Recording boundaries are preserved.
- Prototype transition statistics are fit from training nights only.
- Validation controls model development.
- Final test evaluation is opt-in and lock-protected.
- `--no-reuse` should be used for wall-clock benchmarking.

## Notebook status

The old one-cell and multi-cell notebooks were useful development artifacts, but they are not the canonical implementation anymore. The package under `src/protosleep/` is the source of truth.

## Citation / paper status

This is active research code. A formal citation will be added when the method and evaluation protocol are frozen.

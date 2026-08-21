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
│   ├── config.py
│   ├── data.py
│   ├── attnsleep.py
│   ├── prototypes.py
│   ├── mist.py
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

## Data format

The loader intentionally follows the AttnSleep preprocessing convention used during development:

- one `*.npz` per recording;
- `x` and `y` arrays;
- `(N,3000,1)` is converted to `(N,1,3000)`;
- 30 s epochs at 100 Hz;
- Sleep-EDF subject ID is parsed from `filename[3:5]`;
- recording/night boundaries are preserved.

Set the dataset directory either with `--data-dir` or `SLEEP_EDF_NPZ_DIR`. No data or checkpoints are committed to the repository.

## Test-set safety

The normal runner builds/loads **train and validation caches only**. The designated test cache is not touched unless `--final-test` is explicitly supplied.

After final-test evaluation a persistent `TEST_EVALUATED.lock` is written. Re-evaluation is blocked unless `--allow-test-rerun` is also explicitly supplied.

## Current development evidence

Do not treat a single fold as a paper result. Multi-fold development checks showed substantial subject-to-subject variance. In the matched 5-fold / 3-seed audit, full-label prototype SSL was close to neutral and transition-aware masking was not consistently better than random masking. Prototype-space MAE was more consistent than the continuous latent-MAE control, but this is still a development observation rather than a final claim.

## Next experiment: recover the exact historical MorphMAE v2 recipe, then pretrain fold-specific encoders

The original MIST/MorphMAE project identified **prototype + MAE initialization without WCO** as the most promising mechanism. Historical `best_morphmae.pt` checkpoints have now been shown to be strictly compatible with the current AttnSleep MRCNN, but those old checkpoints do not declare pretraining subjects. They therefore cannot be used as leakage-safe fold checkpoints for a confirmatory A3/A4 comparison.

The repository deliberately does **not** guess the missing MorphMAE-v2 training details from the report. Before porting fold-specific pretraining, audit the historical source tree and checkpoint schema:

```bash
python scripts/audit_legacy_morphmae.py \
  "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --checkpoint "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase/outputs/mae_edf78_v2/best_morphmae.pt" \
  --report legacy_morphmae_audit.txt \
  --json legacy_morphmae_audit.json
```

The audit is read-only. It scans source/config files for the exact architecture, masking, loss, optimizer, epoch, batch-size, split and checkpointing implementation and summarizes the historical checkpoint without printing tensor contents. The generated audit reports are ignored by git.

Once the exact historical recipe is recovered, the next code change is a **fold-specific MorphMAE-v2 pretraining runner** that:

1. receives the current subject-level fold;
2. pretrains only on that fold's training subjects;
3. uses validation subjects for SSL checkpoint selection only if explicitly justified by the frozen protocol (otherwise a training-only SSL holdout will be created);
4. records `train_subjects`, fold, seed, code/config hash and checkpoint SHA-256;
5. exports a strict AttnSleep-compatible MRCNN state dict;
6. never reads the designated test subject during pretraining.

Only after those fold-specific checkpoints exist should `run_mist_stability.py` be used for the matched `A3_current` versus `A4_current` validation comparison.

The current repository does **not** claim that its spherical `ProtoAttnSleep` is byte-for-byte identical to the historical WaveSleepNet-derived prototype implementation. The current matched pair is therefore named `A3_current` / `A4_current`.

## MIST checkpoint inspection

```bash
python scripts/inspect_morphmae_checkpoint.py /path/to/checkpoint.pt
```

A compatible checkpoint prints `STRICT MRCNN COMPATIBILITY: PASS`, the detected state-dict container/prefix, tensor count and file SHA-256.

## MIST stability runner

After leakage-safe fold-specific MAE checkpoints exist:

```bash
python scripts/run_mist_stability.py \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --folds 0,1,2 \
  --seeds 123,456,789 \
  --mae-checkpoint-pattern "/path/to/mae/fold_{fold:02d}/encoder.pt" \
  --output-dir mist_sleep_runs/a1_a3_a4_stability
```

The runner evaluates validation subjects only, verifies fold metadata when available, records checkpoint provenance and refuses a mismatched pretraining split.

## Speed audit

```bash
python scripts/speed_audit.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Reproducibility notes

- Random seeds are fixed per fold/run.
- Train/validation/test subjects are disjoint.
- Recording boundaries are preserved.
- Prototype transition statistics are fit from training nights only.
- Validation controls model development.
- Final test evaluation is opt-in and lock-protected.
- `--no-reuse` should be used for wall-clock benchmarking.
- MIST checkpoint bridges use exact key/shape matching and `strict=True`.
- Fold-specific MAE pretraining must remain subject-disjoint from validation/test subjects.

## Notebook status

The old one-cell and multi-cell notebooks were useful development artifacts, but they are not the canonical implementation anymore. The package under `src/protosleep/` is the source of truth.

## Citation / paper status

This is active research code. A formal citation will be added when the method and evaluation protocol are frozen.

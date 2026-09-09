# Runbook: MIST-Evidence v1

## 1. Work separately from main

First-time setup, from the original repository:

```bash
cd ~/Desktop/bhi/proto-sleep/Proto-sleep
conda activate sleep_ic
git fetch origin
git worktree add ../Proto-sleep-evidence -b research/mist-evidence-v1 origin/research/mist-evidence-v1
cd ../Proto-sleep-evidence
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The new worktree keeps the original branch and files in place. If this worktree already exists, enter it and use `git pull --ff-only`; do not recreate it or reset it. The `PYTHONPATH` setting loads the isolated new package without redirecting the old environment's editable installation. The parent environment already includes torch, NumPy, scikit-learn, matplotlib, and pytest. No new package is required for the added code.

Do not pull code into a worktree while a run from that worktree is active. Fingerprints deliberately reject resume after implementation or configuration changes.

## 2. Run tests and a synthetic end-to-end smoke

```bash
mkdir -p mist_evidence_runs/checks
set -o pipefail
{
  python -m pytest tests/evidence -q &&
  python -u scripts/run_mist_evidence.py smoke \
    --output-dir "mist_evidence_runs/checks/synthetic_$(date +%Y%m%d_%H%M%S)" \
    --device cuda
} 2>&1 | tee "mist_evidence_runs/checks/check_$(date +%Y%m%d_%H%M%S).txt"
```

This trains a reduced model for two epochs on synthetic signals and exports waveform explanations. The synthetic labels and signals are software fixtures, not a biomedical benchmark. The CUDA test is skipped on CPU-only machines; use `--device cpu` to test without a GPU.

## 3. Check the real dataset before training

```bash
python -u scripts/run_mist_evidence.py preflight \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 --output-dir mist_evidence_runs/preflight_fold0 --device cuda \
  2>&1 | tee mist_evidence_runs/checks/data_preflight.txt
```

Temporal eligibility requires retained original epoch indices. If you see a missing-`epoch_indices` error, do not blindly claim adjacency. The safe first test is epoch-only:

```bash
python -u scripts/run_mist_evidence.py preflight \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 --output-dir mist_evidence_runs/preflight_epoch_only --device cuda \
  --epoch-only 2>&1 | tee mist_evidence_runs/checks/data_epoch_only.txt
```

`--epoch-only` makes every epoch independent and disables both temporal filtering and CRF in training. It does not invent original timestamps. This can run the morphology portion immediately, but it is not an evaluation of the full sequence architecture. Rebuild original epoch indices from preprocessing records before the full temporal study. Only use `--assume-contiguous` when that assertion has been independently checked; it is recorded as an assumption in the audit.

## 4. Two-epoch real-data integration test

For temporally verified data:

```bash
EPOCHS=2 PATIENCE=2 SEEDS=123 \
OUT=mist_evidence_runs/fold0_hardware_smoke \
bash scripts/run_mist_evidence_dev.sh
```

For the safe epoch-only alternative:

```bash
EPOCHS=2 PATIENCE=2 SEEDS=123 \
OUT=mist_evidence_runs/fold0_epoch_only_smoke \
bash scripts/run_mist_evidence_dev.sh --epoch-only
```

The shell script reruns tests, performs preflight, trains three conditions sequentially, and saves stdout, stderr, and timing to timestamped `.txt` files. It stops on errors. It does not overwrite earlier console logs. The pipeline does not infer success from the existence of a checkpoint file.

## 5. Fixed initial development comparison

After E0/E1 pass, and for temporally verified data:

```bash
SEEDS=123,456,789 OUT=mist_evidence_runs/fold0_development_v1 \
bash scripts/run_mist_evidence_dev.sh
```

To run another development fold, set `FOLD=1` and a distinct `OUT`. All EDF-20 folds are already development data. Do not describe these as independent confirmation.

Ablations are available through `VARIANTS`:

```bash
VARIANTS=evidence,observed_only,no_summary,no_amplitude,no_context,no_crf \
SEEDS=123,456,789 OUT=mist_evidence_runs/fold0_ablations_v1 \
bash scripts/run_mist_evidence_dev.sh
```

Do not launch this larger block until the initial comparison justifies it. `raw_context` is a new nonlinear control, not a reproduced SOTA model. Independent reference reproductions remain planned work.

## 6. Explicit manifest for another dataset

Use `--manifest path/to/manifest.json` instead of `--data-dir` when filenames do not use the EDF-20 convention. Paths may be relative to the manifest. Example structure:

```json
{
  "dataset": "source_cohort_name",
  "purpose": "development_only",
  "records": [
    {"path": "train/person_a.npz", "subject": "cohort:person_a", "split": "train"},
    {"path": "validation/person_b.npz", "subject": "cohort:person_b", "split": "val"},
    {"path": "reserved/person_c.npz", "subject": "cohort:person_c", "split": "test"}
  ]
}
```

This is a schema example, not a usable experiment with only three people. All training stages must be present. All recordings from a person belong to one split. Test paths are reserved but are never opened by this runner. The manifest does not establish that the underlying labels or channels are correct; that requires a preprocessing audit.

## 7. Output structure

```text
results/
  data_audit.json
  anchor_bank.pt
  anchor_provenance.json
  results.csv
  SUITE_COMPLETE.json
  seed_123/
    evidence/
      provenance.json
      history.csv
      last.pt
      best.pt
      COMPLETE.json
      validation_predictions.csv
      explanations/example_00/
        index.html
        epoch.png
        match_00.png ...
        explanation.json
        all_contributions.csv
```

Open `index.html` in a local browser. Source snippets are real training waveforms. Displayed clinical labels remain unvalidated. Complete score accounting is in CSV/JSON; top illustrated matches are not the whole explanation.

## 8. Resume and data safety

Rerun the identical command with `--resume` (the shell runner already adds it). Completed runs are re-evaluated against their saved completion marker. Interrupted runs resume from the last completed epoch, including optimizer/RNG state. A mid-epoch interruption repeats that epoch. Changing the architecture, seed, epoch budget, source data, code, or bank identity requires a new output directory.

`best.pt` is a best-so-far checkpoint, not proof of completion. Only `COMPLETE.json`/`SUITE_COMPLETE.json` indicate successful completion of the corresponding scope. A live PID alone does not prove progress; inspect `history.csv` and the flushed `.txt` log.

Generated explanations contain participant signal snippets. Keep `mist_evidence_runs/` private. Do not push these outputs to GitHub. The branch ignore rule covers the default output directory; custom output directories require equivalent protection.

## 9. Known scope limits

The new suite has been tested on CPU with synthetic/local fixtures, not on the user's GPU or actual EEG files. It intentionally does not claim complete medical validation, semantic event labels, external-transfer results, or SOTA. No old experiment code is imported into the new package.

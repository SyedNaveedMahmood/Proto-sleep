# Runbook

Use the separate branch, not main. Run a single GPU job at a time. Commands below use the existing `sleep_ic` environment and write terminal output to timestamped text files. They do not retrain MorphMAE.

## Environment and tests

```bash
cd ~/Desktop/bhi/proto-sleep/Proto-sleep
conda activate sleep_ic
git fetch origin
git switch research/mist-evidence-v1
python -m pip install -e '.[dev,evidence]'
mkdir -p mist_evidence_runs/logs
set -o pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q 2>&1 | tee "mist_evidence_runs/logs/pytest_$(date +%Y%m%d_%H%M%S).txt"
```

On a first checkout Git normally creates a local tracking branch for a unique remote branch. With conflicting local changes, stop and preserve them; do not reset or force a checkout. `main` is untouched by this branch.

## Synthetic main-PC check

```bash
RUN="mist_evidence_runs/smoke_cuda_$(date +%Y%m%d_%H%M%S)"
{ time python -u scripts/run_mist_evidence.py smoke --device cuda --output-dir "$RUN"; } \
  2>&1 | tee "${RUN}.txt"
```

The smoke exercises local training, validation, best/last checkpoints, sequence context, real-waveform report generation and raw perturbation checks on synthetic inputs. A deliberately corrupt locked-test file must never be read. Use `--device cpu` for a CPU check, not as a hidden substitute for a requested CUDA check. Synthetic scores have no scientific interpretation.

## Prepare the existing EDF-20 dataset

```bash
python -u scripts/run_mist_evidence.py make-split \
  --data-dir '/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset' \
  --fold 0 --output mist_evidence_runs/edf20_fold0.json \
  2>&1 | tee mist_evidence_runs/logs/split_fold0.txt

python -u scripts/run_mist_evidence.py inspect --manifest mist_evidence_runs/edf20_fold0.json \
  2>&1 | tee mist_evidence_runs/logs/data_inspection.txt
```

Split creation refuses overwriting an existing manifest. The supported NPZ arrays are x `[N,3000]`, `[N,1,3000]` or `[N,3000,1]`, y integers `0..4`, optionally `-1` for unknown, fs=100, and optional monotonically increasing `epoch_index`. No silent resampling, voltage calibration, wake trimming, staging remap, or multichannel selection occurs. Labels must correspond to Wake/N1/N2/N3/REM in that order.

## Two-epoch real-data integration check

```bash
{ time python -u scripts/run_mist_evidence.py fit \
  --manifest mist_evidence_runs/edf20_fold0.json \
  --config configs/mist_evidence/evidence_v1.json \
  --epochs 2 --patience 2 --device cuda \
  --output-dir mist_evidence_runs/edf20_runtime_evidence; } \
  2>&1 | tee "mist_evidence_runs/logs/runtime_$(date +%Y%m%d_%H%M%S).txt"

python -u scripts/run_mist_evidence.py explain \
  --manifest mist_evidence_runs/edf20_fold0.json \
  --checkpoint mist_evidence_runs/edf20_runtime_evidence/best.pt \
  --epoch 0 --perturb --device cuda \
  --output-dir mist_evidence_runs/edf20_runtime_report \
  2>&1 | tee mist_evidence_runs/logs/explanation.txt
```

Open `mist_evidence_runs/edf20_runtime_report/explanation.html` in a browser. It is self-contained. These are preliminary model waveforms, not validated event detectors. The two-epoch run uses its own output path and is not the full experiment.

## Controlled local experiment

After execution/outputs are verified:

```bash
{ time python -u scripts/run_mist_evidence.py suite \
  --manifest mist_evidence_runs/edf20_fold0.json \
  --config configs/mist_evidence/evidence_v1.json \
  --seeds 123,456,789 --arms descriptor,dense,evidence,attnsleep \
  --device cuda --resume --output-dir mist_evidence_runs/edf20_local_suite; } \
  2>&1 | tee "mist_evidence_runs/logs/local_suite_$(date +%Y%m%d_%H%M%S).txt"
```

This is 12 local trainings for one development split, not 12 independent held-out subjects. Repeat on predeclared development manifests; do not cherry-pick folds. Config/seed/code/data fingerprint changes require a new run folder.

## Optional context check

```bash
{ time python -u scripts/run_mist_evidence.py fit-context \
  --manifest mist_evidence_runs/edf20_fold0.json \
  --checkpoint mist_evidence_runs/edf20_local_suite/seed_123/evidence/best.pt \
  --device cuda --resume --output-dir mist_evidence_runs/edf20_context_seed123; } \
  2>&1 | tee "mist_evidence_runs/logs/context_$(date +%Y%m%d_%H%M%S).txt"
```

If `epoch_index` is missing, context training refuses to assume continuity. Recover the original epoch positions from preprocessing. Only after independently verifying that no epochs were dropped may the operator add `--assume-contiguous`. That assumption is recorded. Do not add it just to silence the check.

To obtain local plus context explanations, add the CRF `best.pt` as `--context-checkpoint` to `explain`. To fit identical contexts for all controls, use suite `--with-context` after continuity verification.

## Resume, monitoring and privacy

Repeat an interrupted fit/suite/context command with `--resume` and exactly the same arguments. It resumes from the last completed epoch, or verifies a COMPLETE.json and skips finished work. The presence of best.pt alone is not enough. Logs print flushed epoch progress; `python -u` also prevents pipe buffering.

Raw exemplar waveforms and record IDs are stored in local checkpoints/reports. Keep them under `mist_evidence_runs/`, whose .gitignore excludes generated files. Do not publish private-cohort examples or checkpoint banks without permission. Main-PC GPU compatibility and timing must be established by the smoke, not inferred from CPU tests.

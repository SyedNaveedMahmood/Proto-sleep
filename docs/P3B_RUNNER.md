# P3B matched full-data source baselines

The user's P3B authorization is limited to the existing 15 TRAIN/4 VAL people.
Use the frozen P3A split without regenerating it: the split generator reads
reserved identity prefixes, which are outside this phase's access scope.
P3B reads the existing frozen metadata/receipt and opens only the 37 development
descriptors and their sources. P1/P2 artifacts and previous reports remain
unchanged. No reserved SC, ST, SHHS, MCR or augmentation access is supported.

`configs/p3b_sc_matched_v1.json` implements the previously proposed and now
authorized four-model/three-seed policy. Each run uses maximum 60 epochs,
patience 12, AdamW lr 0.0003, weight decay 0.0001, core/encoding batches of 16,
gradient clipping 5 and deterministic FP32 (both AMP and TF32 disabled before
correctness checks or full scores). The same stopping policy and
participant-average fixed-five source-VAL Macro-F1 selection applies to every
model. Improvement must exceed 1e-8, preserving the earliest best on ties.
Record actual epochs/updates; identical maximum budgets do not imply identical
realized computation under early stopping. No hyperparameter tuning occurs.

SourceRecording is explicitly constructed with `smoke=False`. The loader
requires exactly 79,984 physical TRAIN/79,025 scored and 21,796 physical
VAL/21,789 scored epochs, preserving all original indices and masks. Supervised
cores cover every scored TRAIN epoch exactly once per epoch, without bridging
unknown positions/nights. Unscored physical TRAIN positions contribute to the
already fitted label-independent normalization, and remain in the full input
grid; they do not incur a forward pass or loss in epoch-only supervised cores.
VAL predicts all physical positions before scoring masks. Scaling is copied
from verified P3A all-TRAIN statistics, without fitting VAL. Seed 123's real
TRAIN medoid bank is reused; seeds 456/789 use the unchanged TRAIN-only medoid
recipe. All three banks are frozen and source-crop verified before full scores;
each seed shares one bank across its four controls. No clinical event labels.

Commit the final implementation before executing correctness checks or any real
training. Code/environment/config/source/hardware fingerprints are immutable
during active experiments. Required correctness receipt records successful
focused and full CPU suites plus all-four interrupted CUDA resume fixtures;
the fixtures require exact parameter, optimizer, CPU/CUDA RNG and validation
history agreement with uninterrupted execution. No CPU fallback is allowed.

```bash
set -euo pipefail
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/check_transfer_p3b.py \
  2>&1 | tee mist_transfer_runs/p3b/session_logs/correctness_console.txt
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/run_transfer_p3b.py --authorize-full-sc-train-val --mode gate-a \
  2>&1 | tee mist_transfer_runs/p3b/session_logs/gate_a_console.txt
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/run_transfer_p3b.py --authorize-full-sc-train-val --mode gate-b --resume \
  2>&1 | tee mist_transfer_runs/p3b/session_logs/gate_b_console.txt
```

Gate A performs one complete real TRAIN epoch and complete VAL pass for each
seed-123 model, then writes PAUSED.json and raises EpochPaused after sealing
last/best/optimizer/RNG/history. It never writes COMPLETE for those pauses. The
training configuration remains 60 epochs/patience 12, so Gate B resumes at epoch
two without replaying the benchmark or changing the run identity. The maximum
cost estimate is 180 times the sum of the four measured epoch durations, with
no early-stop discount. The prespecified operational ceiling is 24 GPU hours
for the complete 12-run maximum-budget workload. If exceeded, stop and report
the measured costs; never shrink subjects, epochs or variants silently.

Gate B requires Gate A PASS with validated output digests. It runs seeds
123/456/789 and the four models sequentially, skipping only sealed completed
runs. A worktree-wide GPU lock prevents parallel P3B writers, and launch rejects
other GPU compute processes without interrupting them. An additional per-output
lock protects the run directory. Request an operational epoch-boundary pause
by creating `mist_transfer_runs/p3b/v1/PAUSE_REQUESTED` or sending SIGUSR1 to
this runner. Remove the request before `--resume`; never edit configuration to
shorten an experiment. A paused run is not COMPLETE. Interrupted epochs replay
from the preceding sealed boundary. Corrupted or unsealed checkpoints block
resume before trusted checkpoint deserialization.

The reused trainer checks finite logits, loss, gradients, model and optimizer
state. Epoch reports include training/VAL wall times, supervised updates,
streamed CPU read/calibration/scaling time (a subset of wall time), CUDA peak
allocated/reserved bytes, checkpoint writes and process high-water RSS in KiB.
CPU RSS is the process lifetime high-water mark; GPU figures exclude driver
allocations. CUDA synchronizations from finite checks make timings conservative.
Per-run resource totals exclude setup and final best-checkpoint reevaluation;
record the enclosing console wall time separately. Completed-run markers seal
every checkpoint, epoch report and prediction file with digests. Root COMPLETE
exists only when all 12 VALIDATED_COMPLETE markers and their outputs validate.

One-class predictions and >=98% largest-class prediction fraction are recorded
as diagnostics, without changing the recipe or converting negative results to
software failures. Fixed-five metric, absent-class, Brier and 15-bin ECE
definitions remain identical to P3A. Subject variability is descriptive for only
four VAL people; seeds are repeated optimizations, not independent subjects.
Detailed predictions, waveform banks/checkpoints and logs stay local/ignored.
Publish aggregate metrics only. Stop after P3B; source validation establishes
reference models, not reserved performance or independent external transfer.


## Validated completion

All 12 frozen-v1 runs completed; see [the aggregate report](../reports/p3b/FINAL_COPY_PASTE.txt). Training implementation commit: 661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23. The later report-only commit intentionally changes Git HEAD. To use the runner to verify/resume these historical artifacts, retain the original frozen implementation identity and environment; a current report-commit HEAD is rejected rather than silently accepted. Standalone read-only output digest verification does not change training or model selection. The completed-run resume check was performed before creating the report commit and did not train or alter sealed outputs.

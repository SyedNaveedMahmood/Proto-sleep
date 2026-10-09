# P3A source-only SC baseline integration

Run from this research worktree using the existing `.venv` and `PYTHONPATH=src`.
The local canonical P2 manifest and reviewed completion must already exist.
No dataset download, NPZ reuse, ST or SHHS access is supported.

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_transfer_protocol.py --resume
PYTHONPATH=src .venv/bin/python scripts/run_transfer_experiment.py --dry-run
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/transfer/test_source_protocol.py tests/evidence -ra
PYTHONPATH=src .venv/bin/python scripts/run_transfer_experiment.py
PYTHONPATH=src .venv/bin/python scripts/run_transfer_experiment.py --resume
```

Use `set -euo pipefail` and `2>&1 | tee <new-log>.txt` to preserve output and exit
status. First-time freezing omits `--resume`; the existing frozen split must
never be overwritten. `experiments/splits/p3a_sc_source_v1.json` and its receipt
are exact metadata-only copies of the local freeze. They include public SC IDs,
input hashes and local descriptor locators, without annotations or waveforms.
Relocating verified data requires an explicitly reviewed locator migration;
never reroll subject roles or silently replace the fingerprint.

The CLI requires CUDA for the real integration, FP32, deterministic kernels,
the four fixed variants and one/two smoke epochs. Its four four-epoch windows at
20/40/60/80% of each TRAIN/VAL recording are selected from the physical grid
before labels. All development participants and nights remain included. This
bounded smoke is not an epoch over the complete TRAIN dataset. It establishes
execution, mask handling and matched optimizer budgets; it cannot establish
performance or select architectures/hyperparameters. Full-grid lazy views are
available in the adapter but the CLI rejects full P3B training configurations.

All four share calibrated EEG Fpz-Cz, native 100 Hz, 3,000 samples/epoch,
source-TRAIN global scaling, stage mapping, scored-only loss/metrics, roles,
AdamW budget and participant-average VAL selection. Temporal context is zero
and CRF is disabled in this frozen reference comparison. `raw_context` retains
the existing variant name; here it is the unrestricted local neural decision
control at radius zero. AttnSleep uses the existing repository implementation
and initialization; this is an integration of that implementation, not a claim
of independently reproducing published benchmark scores. Summary-only uses
the existing measured signal descriptors. Evidence uses existing observed
waveform matching and training-waveform medoids. No morphology event diagnosis
is implied. The optional `evidence_aug` adapter applies independent TRAIN-only
calibrated gain in [0.98, 1.02]; it passes synthetic checks but is outside the
four-way CUDA smoke. It needs its own CUDA integration before matched training.

TRAIN scaling reads every complete physical TRAIN epoch, including unscored
positions, without stage access. The 48 waveform medoids (16 per scale of
100/200/400 samples) come from bounded, stage-balanced TRAIN candidates only.
Their exact normalized crops, original indices, offsets, recording/source and
annotation hashes are verified before training. Amplitude descriptor statistics
use all physical TRAIN epochs. VAL contributes only checkpoint metrics.

The reused trainer saves atomic best/last checkpoints with optimizer and all
RNG states. Resume is exact at epoch boundaries; an interruption inside an
epoch replays that epoch from the preceding saved boundary. Synthetic tests
compare resumed parameters, optimizer state and RNG with uninterrupted runs.
Fingerprints bind source proofs, frozen roles, configuration, anchors,
normalization, implementation hashes, Git HEAD, environment and device. Resume
under changed inputs/code/configuration is rejected. Completed resume verifies
all output digests and TRAIN/VAL source proofs without training or rewriting
timings. To reproduce/resume an older run after a documentation commit, use its
recorded implementation commit in a separate worktree and verified locators.

Metrics use Wake/N1/N2/N3/REM = 0..4, with fixed-five Macro-F1 and undefined F1
set to zero. Pool nights within each participant, then average participant
Macro-F1 for selection; retain pooled Macro-F1 separately. Balanced accuracy
averages observed truth classes; also report fixed-five recall with absent
classes zero. Undefined kappa/recall is null, and absent truth classes are
explicit. A participant with no scored validation epoch blocks selection.
Accuracy, class F1, confusion matrices, NLL, Brier and 15-bin ECE are recorded.
No calibration fitting or target access occurs. Validation inference retains
unscored positions and original physical indices; masking affects loss/metrics.

Local checkpoints, waveform banks, detailed predictions and all `.txt` terminal
logs stay in ignored `mist_transfer_runs/p3a/`. The final public report contains
aggregate validation and provenance only. Stop after P3A; P3B and any reserved
evaluation require separate review/authorization.

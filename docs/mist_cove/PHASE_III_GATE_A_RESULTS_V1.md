# Phase III Gate A v1 — six sealed pauses; Gate B NOT AUTHORIZED

**Gate A completed and passed its measured budget gate. All six seed-123 runs are paused at epoch 1. Gate B has not started and is not authorized.** The user's actual review explicitly approved Gate A only, including the instruction: “Do not start Gate B automatically, even if the estimated cost is below 24 GPU hours.” That restriction supersedes the earlier proposal's conditional continuation wording.

The user's message is recorded verbatim in ignored local storage, with a machine-readable transcription of that actual decision. No additional user approval was generated. Approved proposal SHA256: `8282be8a6d7124a0fd6e83a4ae8b8e273919fc0b7976a756389e8088d3ec38b0`. Actual user-review record SHA256: `d2c38ce22d1f6460eab84504a764173c8c754b3fa1617c8897a3381b34bead62`; authorization-message SHA256: `ca6f162c5cee85fe79ed9e806f6c41dcda78a3f62c972ef478296360a57043f3`.

[Safe aggregate receipt](../../reports/mist_cove/phase3_gate_a_v1.json) contains all measured timings, first-epoch aggregate metrics, resources and output hashes. [Prior preflight report](PHASE_III_PREFLIGHT_RESULTS_V1.md) remains historical evidence of the preceding gate. All detailed source proofs, checkpoints, user-message transcription, logs and participant metrics stay under ignored `mist_transfer_runs/mist_cove/phase3_source_v1/`.

## Actual execution and frozen identity

- Branch: `research/mist-cove-phase3-real-eeg-v1`. Launch HEAD: `58a12895bc8ea363041a1e6e91f0b7c065346267`; frozen tested implementation: `a601e8e004b9e0c2942e83d4517fb8a47856de72`. Subsequent commits add aggregate reports only.
- Implementation SHA256: `d08254fade1b8ba91715143c44c7043f88956eba937984557a011609f0a6bda9`.
- Config canonical SHA256: `af81748149c1bfb3c18dcf5c009c0a9f95bcabdc5a623ecc1e2b7d0b7420df46`; file SHA256: `525fc8152e78219a9bee51ad84a4d067761b22c873d10199d25e426ff012883d`.
- Frozen split file SHA256: `1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5`; fingerprint: `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84`.
- Same 15 TRAIN / 4 VAL people, 29/8 recordings, calibrated original Fpz-Cz uV EEG, 100 Hz / 3,000 samples / 30 seconds. No legacy NPZ substitution.
- Full recipe remains 60 max epochs / patience 12, AdamW 3e-4 / 1e-4, core/encoding batch 16, gradient clip 5, deterministic FP32, AMP/TF32 off, radius 0 / no CRF. Operational pausing did not shorten this configuration.
- Actual device: **NVIDIA GeForce RTX 4080 SUPER**, 16,718,168,064 reported bytes; driver 580.95.05, PyTorch 2.5.1, CUDA build 12.1, cuDNN 90100. Python 3.12.9, imported/selected NumPy 1.26.4; two CPU threads and CUBLAS workspace `:4096:8`.

Each model consumed exactly **79,025 scored TRAIN epochs / 4,978 optimizer updates**. Validation inferred all **21,796 physical epochs**, including 7 unscored positions, before metrics used the 21,789 scored positions. No nights or unscored gaps were joined for supervision. Six runs total **29,868 optimizer updates**, not six independent participant experiments.

## Measured timing and capacity

Epoch wall includes measured TRAIN, full-grid VAL, finite-state checks and checkpoint writing. I/O times are subsets of TRAIN/VAL wall time. The estimate excludes enclosing launch setup/proof verification and observer overhead; total enclosing CLI wall was not separately instrumented. Device allocator peaks exclude CUDA context/driver/display memory. CPU RSS in the receipt is lifetime process high-water, not per-model incremental memory.

| Model | TRAIN s | VAL s | Epoch s | Updates | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 27.767 | 6.627 | 34.401 | 4,978 | 65.99 | 68.00 |
| availability | 27.686 | 6.655 | 34.364 | 4,978 | 65.99 | 68.00 |
| background | 27.147 | 6.727 | 33.881 | 4,978 | 65.99 | 68.00 |
| rounded_q1 | 27.651 | 6.625 | 34.299 | 4,978 | 65.99 | 68.00 |
| raw_context | 49.295 | 4.994 | 54.303 | 4,978 | 185.03 | 232.00 |
| evidence | 78.219 | 10.662 | 88.908 | 4,978 | 186.48 | 246.00 |

| Model | Scored TRAIN epochs/s | Physical VAL epochs/s | Parameters | Trainable parameters |
|---|---:|---:|---:|---:|
| q1 | 2846.0 | 3289.1 | 75 | 75 |
| availability | 2854.3 | 3275.1 | 75 | 75 |
| background | 2911.0 | 3239.9 | 75 | 75 |
| rounded_q1 | 2858.0 | 3289.9 | 75 | 75 |
| raw_context | 1603.1 | 4364.8 | 34,197 | 33,957 |
| evidence | 1010.3 | 2044.3 | 13,556 | 13,556 |

Background instantiates 75 parameters but its active decision path uses 40. No capacity-parity claim is made against the historical encoders.

Sum of measured initial epochs: **280.155691 seconds**. Under the frozen maximum budget (three seeds × 60 epochs per model):

```text
maximum-budget estimate = 180 × sum(six measured epoch durations) / 3600
                        = 14.007785 GPU hours
ceiling                 = 24 GPU hours
budget gate             = PASS
Gate B authorization    = ABSENT — separate user authorization required
```

This is a single-device elapsed-workload estimate based on seed-123 initial epochs, not measured active-kernel utilization or a guarantee about all future epochs/seeds. It includes these initial epochs and assumes no early stopping. It does not authorize Gate B, establish final VRAM peaks, or tune the recipe.

## Post-run integrity verification

After the training process exited, a read-only audit reloaded only the authorized development roles and verified all original EDF/annotation hashes again. TRAIN normalization, trusted completion receipts, all three bank hashes and exact source crops match the preflight. The actual imported environment, split/config digests and complete implementation snapshot remain unchanged.

For each run the audit reconstructed the source/model/config/environment/bank-bound sealed identity, verified every checkpoint digest **before** trusted deserialization, and checked that:

- `EPOCH_SEAL.json`, immutable `gate_a_boundary.json` and `PAUSED.json` agree on epoch 1, signature and output digests;
- `last.pt`, `best.pt`, history and provenance match their seals; the two epoch-1 model states match exactly;
- history has exactly one entry, 4,978 updates and 79,025 supervised epochs; all recorded model/optimizer tensors and gradients/loss diagnostics are finite;
- AdamW state steps are 4,978; Python, NumPy, CPU Torch and CUDA RNG state are present;
- full-grid VAL coverage is 21,789 scored + 7 unscored; no `COMPLETE.json` or `COMPLETE_SEAL.json` exists for these paused runs.

All checks passed. No real paused run was advanced to test resume: doing so would exceed this authorization. Reviewed six-method synthetic CUDA exact-resume receipts still bind the unchanged implementation/environment. The existing CPU suite **381 passed / 0 failed / 11 CUDA skipped** and separate CUDA suite **11 passed / 0 failed / 0 skipped** were digest-reverified rather than unnecessarily rerun for a report-only change. No code or configuration changed during training or audit. No competing compute job was interrupted; the device has no remaining project compute process after exit.

## First-epoch scientific diagnostics — preliminary, no tuning

These are source-VAL checkpoint-selection metrics after **one epoch**, not completed comparisons, independent test results or evidence of transfer. Stage F1/recall, pooled metrics, confusion, NLL, Brier and ECE are retained in the safe aggregate receipt; participant-specific metrics remain only in private checkpoints. All models used the same eligible samples and update counts.

| Model | Participant-average Macro-F1 | Pooled Macro-F1 | Largest predicted-class fraction |
|---|---:|---:|---:|
| q1 | 0.159969 | 0.160251 | 1.000000 |
| availability | 0.159969 | 0.160251 | 1.000000 |
| background | 0.159969 | 0.160251 | 1.000000 |
| rounded_q1 | 0.160135 | 0.160496 | 0.999954 |
| raw_context | 0.518051 | 0.558619 | 0.719996 |
| evidence | 0.425991 | 0.471580 | 0.746661 |

q1, availability and background predict Wake for every scored VAL position after this epoch. Rounded q1 predicts Wake for all but one position. Both historical baselines predict no N1 at this first checkpoint. These collapse/absent-stage results are retained; no learning rate, architecture, stopping rule, eligible sample set or candidate list was changed. They do not settle the final accuracy or evidence-mechanism hypothesis. The previous H1 failure, donor sensitivity and distractor limitations remain unchanged. No real waveform-physiology or counterfactual conclusion has been drawn.

## Commands and retained local artifacts

Executed from the repository root, using the actual recorded user review:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  mist_transfer_runs/mist_cove/phase3_source_v1/environment/bin/python -u \
  scripts/run_mist_cove_phase3.py --authorize-source-train-val --mode gate-a \
  --proposal mist_transfer_runs/mist_cove/phase3_source_v1/launch_proposal_v1.json \
  --review mist_transfer_runs/mist_cove/phase3_source_v1/gate_a_actual_user_review_v1.json \
  --output mist_transfer_runs/mist_cove/phase3_source_v1/training_v1 \
  > mist_transfer_runs/mist_cove/phase3_source_v1/gate_a_console_v1.txt 2>&1
```

The runner refuses to replay existing Gate A pauses. Do not run this command again against the same output. The local `verify_gate_a_v1.py` performs the documented read-only post-verification; its SHA256, console/receipt hashes and every checkpoint/seal hash are retained in the aggregate receipt. Private audit output: `gate_a_postverification_private_v1.json`; safe projection: `gate_a_safe_aggregate_v1.json`. The published report includes no source paths, subject-level records, waveform arrays or weights.

**PASS:** six genuine one-epoch source runs, full-grid validation, measured budget under ceiling, finite state and all post-run provenance/seal checks.

**FAIL / retained diagnostics:** three analytic variants have complete Wake prediction collapse and rounded q1 is nearly collapsed after one epoch. This is an observed early diagnostic, not a prespecified final hypothesis test or a reason to alter the comparison.

**NOT TESTED:** completed 18-run results, final source performance/evidence utility, real interrupted-versus-uninterrupted training beyond the paused boundary, final calibration/clinical event validity, reserved or independent-cohort generalization.

**BLOCKED:** Gate B lacks separate user authorization, regardless of the passing budget estimate. All six runs remain paused under their original full-training identity. No Gate B command or further optimizer update has been executed.

RESERVED EEG/annotations/descriptors/statistics, Sleep-Telemetry, SHHS and legacy NPZ were not accessed. No normalization or bank was fitted/refitted, no split changed, and historical Phase I/II/P3B implementation/results/checkpoints remain untouched.

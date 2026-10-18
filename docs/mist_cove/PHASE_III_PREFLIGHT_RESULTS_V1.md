# Phase III source preflight v1 — STOP FOR INDEPENDENT LAUNCH REVIEW

The authorized real Sleep-Cassette TRAIN/VAL preflight is **PASS**. **No EEG training, optimizer update, source-VAL accuracy evaluation or checkpoint selection has occurred.** Phase III is incomplete: the six-model/three-seed comparison remains **BLOCKED pending independent review**, as required by the user and the Phase II contract. Source usefulness and the 24 GPU-hour maximum-budget feasibility are **NOT TESTED**.

## Identity and authorization

- Branch: `research/mist-cove-phase3-real-eeg-v1`, based directly on verified Phase II `97720bc1a31c8aa97841a2eb1bfb2a61d75b8d41` after fetching origin. Original worktree was clean; no existing branch/worktree was overwritten.
- Frozen, tested implementation: `a601e8e004b9e0c2942e83d4517fb8a47856de72`. Later report-only commits do not change source/config identities.
- Implementation SHA256: `d08254fade1b8ba91715143c44c7043f88956eba937984557a011609f0a6bda9`.
- Execution config file SHA256: `525fc8152e78219a9bee51ad84a4d067761b22c873d10199d25e426ff012883d`; canonical SHA256: `af81748149c1bfb3c18dcf5c009c0a9f95bcabdc5a623ecc1e2b7d0b7420df46`.
- Authorization: user's 2026-10-18 Phase III master prompt, sections 3/8 and final instruction. This authorizes the concrete preflight and conditionally authorizes training after review; the agent has not independently approved its own receipts.
- [Frozen source protocol](PHASE_III_PROTOCOL_V1.md), [execution config](../../configs/mist_cove_phase3_source_v1.json), [safe aggregate receipt](../../reports/mist_cove/phase3_preflight_v1.json).

## PASS — real original EDF preflight

Only the 29 TRAIN and 8 VAL descriptors were loaded using `read_frozen_split` and `load_development`. Frozen split SHA256 `1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5` and fingerprint `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84` match. All 74 permitted source/annotation files match their recorded hashes. Identity, night grouping, calibration and complete physical grids pass the unchanged source contracts. Legacy NPZ was not substituted.

| Role | People | Recordings | Physical epochs | Scored epochs |
|---|---:|---:|---:|---:|
| TRAIN | 15 | 29 | 79,984 | 79,025 |
| VAL | 4 | 8 | 21,796 | 21,789 |

Every physical position was streamed through the genuine lazy loader in batches of at most 16, including 959 TRAIN and 7 VAL unscored epochs. All normalized FP32 samples were finite. No label-based trimming or gap compression occurred. Streaming took 5.646 s TRAIN and 1.537 s VAL, with a 192,000-byte peak waveform batch. Full preflight wall time was 9.006 s. These cached/local data-path measurements are **not training throughput or a maximum-budget estimate**.

Trusted normalization SHA256 `0679b44c76e8c6667c4daba0b1915420eb04b02310be68536130f807899c5742` matches the original P3A completion and P3B fit receipt. TRAIN-only subjects/source hashes, 79,984 physical fitting epochs and label-independent fitting proofs pass. All three existing seed banks match trusted hashes and exact original TRAIN waveform crops under safe `weights_only=True` loading. No statistic or bank was fitted or regenerated. Detailed source paths/proofs and fitted artifacts remain ignored locally.

The first preflight stopped on the example configuration's `Sleep_edf_fulldataset` spelling. The frozen original path uses `SLeep_edf_fulldataset` (capital L); only that spelling exists locally. Both first authorized file hashes matched. The new path check was corrected and re-tested to follow exact frozen original paths, record the discrepancy, and reject any changed hash. No original file, frozen descriptor, old config or split was rewritten. All final receipts cover this correction.

## Environment reconciliation and correctness

The old shared prefix contains both NumPy 2.0.1 and 1.26.4 distribution records, while its import loads 1.26.4. It was not repaired in place. A new ignored venv inherits shared packages and installs a local pinned NumPy 1.26.4, so imported version and selected distribution metadata both report 1.26.4. This is a checked overlay, not a hermetic container. [Pinned overlay requirements](../../configs/mist_cove_phase3_environment_v1.txt) and the private full package freeze document reconstruction.

During assembly, missing pytest/pyEDFlib/MNE were installed locally. An unconstrained plotting dependency initially selected NumPy 2.4.6; NumPy 1.26.4, matplotlib 3.10.7 and contourpy 1.3.3 were pinned before correctness/source checks. No scientific experiment used that temporary assembly state. Relevant imported Python/dependency versions, paths, metadata, implementation and GPU identity are bound in the private receipts. Future environment changes invalidate launch/resume.

Actual imported Python is 3.12.9, PyTorch 2.5.1, NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2, pyEDFlib 0.1.42, MNE 1.10.2 and pytest 8.4.2. Each imported package version matches its selected metadata.

- Design metadata validator: PASS, 9 experiment definitions / 4 hypotheses, no data access.
- Final complete CPU regression: **381 passed, 0 failed, 11 CUDA skipped, 1 existing PyTorch warning**, including 24 new Phase III tests. Skipped cases were separately executed on CUDA.
- Separate CUDA pytest: **11 passed, 0 failed, 0 skipped, 64 deselected**.
- Standalone generated CUDA probes: all six methods pass exact interrupted/uninterrupted model, optimizer, Python/NumPy/CPU/CUDA RNG, early-best state and semantic validation-history recovery, plus completed resume.
- New mock integration checks freeze all 18 calls, preserve the 60/12/16 recipe and seed-123 resume, calculate the undiscouted maximum budget, and reject Gate B before source opening if the estimate exceeds 24 hours. Other tests cover changed hashes, unauthorized roles before file access, user-review rejection, private paths, GPU contention and unscored preflight positions.

## Actual GPU measurements — no training claim

Device: **NVIDIA GeForce RTX 4080 SUPER**, 16,718,168,064 reported bytes, capability 8.9, driver 580.95.05, CUDA build 12.1, cuDNN 90100. Deterministic FP32, AMP/TF32 off, 2 threads, CUBLAS workspace `:4096:8`. No other GPU compute job was interrupted and no CPU fallback was used.

The real-input probe used the first physical batch of 16 from one authorized TRAIN and one VAL recording. TRAIN masked loss/backward passed; VAL inference did not consult labels. Parameters were checked unchanged and **no optimizer was constructed or stepped**. Five-stage logits/probabilities were finite, intervals stayed within original epochs, and additive residual was at most 2.3842e-7.

| Method | Parameters | Trainable parameters | Real batch-16 peak allocated MiB |
|---|---:|---:|---:|
| q1 | 75 | 75 | 65.99 |
| availability | 75 | 75 | 65.99 |
| background | 75 | 75 | 65.99 |
| rounded_q1 | 75 | 75 | 65.99 |
| raw_context | 34197 | 33957 | 182.53 |
| evidence | 13556 | 13556 | 184.41 |

These are process PyTorch allocator peaks, excluding driver/context allocations. Background instantiates 75 parameters but its active decision path uses 40. Batch feasibility does not validate full-epoch runtime, reserve memory or the entire source workload.

## Independent review packet and next operation

Private root: `mist_transfer_runs/mist_cove/phase3_source_v1/` (ignored):

- `preflight_final_v1/preflight.json`: actual source/annotation/bank/normalization proofs, masks, environment and numerical checks.
- `checks_final_v1/correctness.json`, `cpu.xml`, `cuda_pytest.xml` and console logs.
- `checks_final_v1/cuda/receipt.json`: exact synthetic CUDA recovery and resources.
- `environment_freeze.txt`: actual selected environment inventory.
- `launch_proposal_v1.json`: **unapproved** review proposal binding all preceding receipts.

Proposal SHA256: `8282be8a6d7124a0fd6e83a4ae8b8e273919fc0b7976a756389e8088d3ec38b0`.

Independent user review must reference this exact proposal and approve Gate A before any training. The contract source is [Phase II source readiness](PHASE_II_SOURCE_READINESS_V1.md): **“Run an authorized preflight only, then review its result before training.”** No `review.json` approval file has been created. A CLI flag alone cannot launch training. A reviewer identity/reference in the eventual review record must come from the actual user response; this software record is not cryptographic proof or a claim of peer review.

After review, Gate A runs the six seed-123 first epochs under the complete frozen recipe, pauses at sealed boundaries and measures actual full-grid TRAIN/VAL throughput. It estimates `180 * sum(six epoch seconds) / 3600` GPU hours with no early-stop discount. If >24 hours, stop and report. Otherwise Gate B resumes those six runs and executes all 18 variant/seed combinations. Real waveform evidence analyses follow completed frozen training; they are not performed on untrained models to suggest clinical success. Source usefulness remains unanswered until those results exist.

## Reproduction

Run from the repository root. Preserve old directories; fresh repetitions need unused receipt paths. Commit implementation before the correctness command. The new environment was created with the historical interpreter using `-m venv --system-site-packages`, then the pinned overlay requirements were installed together. Keep detailed outputs local.

```bash
PHASE3_PY=mist_transfer_runs/mist_cove/phase3_source_v1/environment/bin/python
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PHASE3_PY" scripts/check_mist_cove_phase3.py \
  --output mist_transfer_runs/mist_cove/phase3_source_v1/checks_final_v1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PHASE3_PY" scripts/run_mist_cove_phase3.py \
  --authorize-source-train-val --mode preflight \
  --output mist_transfer_runs/mist_cove/phase3_source_v1/preflight_final_v1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PHASE3_PY" scripts/prepare_mist_cove_phase3_launch.py \
  --preflight mist_transfer_runs/mist_cove/phase3_source_v1/preflight_final_v1/preflight.json \
  --cpu mist_transfer_runs/mist_cove/phase3_source_v1/checks_final_v1/correctness.json \
  --cuda mist_transfer_runs/mist_cove/phase3_source_v1/checks_final_v1/cuda/receipt.json \
  --output mist_transfer_runs/mist_cove/phase3_source_v1/launch_proposal_v1.json
```

Only after actual review is recorded, use `scripts/run_mist_cove_phase3.py --mode gate-a --authorize-source-train-val --proposal <reviewed-proposal> --review <actual-user-review-record> --output mist_transfer_runs/mist_cove/phase3_source_v1/training_v1`. Gate B uses the same arguments/output with `--mode gate-b`; it rejects a failed/missing/tampered Gate A. Redirect training console output into the private run root. Do not create a fictitious approval record to execute these commands.

## Scientific status and boundaries

**PASS:** source identities, TRAIN-only artifact reuse, full-grid/mask/finiteness contracts, numerical source GPU feasibility and synthetic exact resume.

**FAIL:** no new scientific comparison was executed. Preserve historical H1 FAIL, partial H2 support, lack of demonstrated gating necessity, donor-dependence failure in five of six q1 synthetic task/seed runs, and confident distractor errors. Software correctness does not reverse them.

**NOT TESTED:** all new source-VAL classification/calibration metrics; stage and participant variation; comparison against newly trained raw_context/evidence/background; real waveform morphology, redundancy and donor sensitivity; complete workload VRAM/runtime; RESERVED or independent-cohort generalization. Historical raw_context 0.723511 / evidence 0.718031 participant-average F1 remain labeled historical source-VAL references, not Phase III measurements.

**BLOCKED:** Gate A/B await independent launch review. The 24 GPU-hour feasibility condition is unmeasured. Four VAL people cannot establish broad population performance. Source classification and explanation validity will be assessed separately; a weak 75-parameter analytic representation cannot falsify all morphology-grounded methods.

RESERVED SC EEG, annotations, descriptors and derived statistics were not accessed. ST, SHHS and legacy NPZ were not accessed. No real normalization/bank was fitted. Historical P3B, Phase I/II source, reports, checkpoints and split remain unchanged. All real waveform data, detailed provenance, fitted artifacts and diagnostics remain local/ignored; only safe aggregate receipts are added to Git.

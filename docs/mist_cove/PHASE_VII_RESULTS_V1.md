# Phase VII v1 — integrity stop, scientific study incomplete

**BLOCKED at Gate VII-0.** Six historical full M1 weight files exist, but none has a trusted historical byte-digest seal. The Phase VII prompt requires verifying seals before loading and investigating the original frozen encoder before training robustness controls. No historical model was loaded or retrained. This is an integrity limitation, not a falsification of H7.1–H7.4.

## Verified identity and preservation

Starting/completed Phase VI commit: `8ec83fed182f4ca69889db17272eec84d05e9771`. New branch: `research/mist-cove-phase7-distractor-robustness-v1`. Inspection implementation commit: `9522a6771f0f91087b1d39b0736fd6e81ed81ac3`. Integrity source/test/config inventory SHA256: `79fcc61b7f46c652c1c4237d41db19046eabba0cc094500e1db1672589f5944d`.

**PASS:** specified ancestry; unchanged Phase VI source/config/generator/preregistration/publication inventories; unchanged repaired CI; completed private aggregate-result digest `d0bc73eecc21a5e2aa613c1aa3310ed82d32814281038e761c6fd6b3eaf1fe4c`; 72 original fit records, 21,600 updates, 345,600 exposures and zero matched-head updates. The historical collective robustness FAIL and every original negative result remain unchanged. Frozen Phase VI source/test inventory remains `9bca5bc8b1a372c75ff8472ae7e06882bbd181b57a968f7ba18b69b3492c4e09`; generator bytes remain bound to its execution manifest.

The inspection verifies manifests against the completed Git tree before trusting their file hashes. The [safe integrity receipt](../../reports/mist_cove/phase7_v1/integrity_receipt_v1.json) contains all checked digests and current observations for `single_{123,456,789}_full.pt` and `multiple_{123,456,789}_full.pt`. Each file is 19,875 bytes; no conventional sidecar is present. The complete private confirmatory directory contains 72 weight files. The completed sealed metrics receipt has no checkpoint hash entries, and the scientific runner uses plain atomic state-dictionary writes without seals.

**BLOCKED:** proving these observed weights are byte-identical to the original run states. Their current hashes remained unchanged across this Phase VII inspection and CPU tests; that establishes only within-session preservation. It does not prove original identity. The four sealed tiny CUDA correctness models have different provenance and were not substituted. A new seal created today would not repair the missing historical proof. No evidence of corruption is asserted.

## Engineering results and actual resources

**PASS:** full CPU regression **709 passed, 0 failed/errors, 11 CUDA skips**, including 36 new integrity cases, one existing PyTorch warning; pytest console wall 12.39 seconds, JUnit suite time 12.384 seconds. Tests cover role rejection before path conversion; traversal and symlink rejection; historical Git anchoring; changed source plus rewritten-manifest rejection; result tampering/accounting; absent/wrong-model checkpoints; refusal to treat observed hashes/new sidecars as historical proof; zero-fit config enforcement; and atomic receipt publication including racing-writer preservation. Existing synthetic model, full-grid/mask, additive and exact-resume contracts remain covered by the unchanged regression suite. New feature-stage/pooling/common-encoder scientific tests are not claimed because those implementations cannot proceed before VII-0.

The first full-suite shell launch failed before pytest because its new log directory did not yet exist. The inspection created that directory; the subsequent full-suite invocation passed. No test or scientific failure was concealed or removed.

CPU interpreter: Python 3.12.9, torch 2.5.1+cpu, NumPy import/metadata both 1.26.4; constrained dependency environment reused read-only. The separate environment probe configured deterministic FP32, two threads, TF32 off. It is an environment record, not a scientific fit or CUDA correctness receipt.

The read-only integrity function measured **0.086311 seconds**, excluding interpreter/package import and CLI receipt publication. Peak RSS for that process was not measured. **0 scientific fits, 0 optimizer updates, 0 generated-example exposures, 0 checkpoint deserializations.** Generated toy fixtures used by correctness tests are engineering checks, not the requested scientific comparisons.

**BLOCKED CUDA:** actual NVIDIA GeForce RTX 4080 SUPER, driver 580.95.05, 16,376 MiB total; query showed 14,334 MiB used / 1,611 MiB free, with another compute process using 14,260 MiB. No job was interrupted, no GPU model operation ran, no fallback was called CUDA validation, and no credentials were requested or stored. New CUDA tests, smoke timing, peak allocated/reserved memory and resume checks are **NOT TESTED** in this phase. Prior Phase VI CUDA PASS receipts remain historical results only. See [correctness/environment receipt](../../reports/mist_cove/phase7_v1/correctness_receipt_v1.json).

CI dependencies/workflow were preserved. Actual final-head workflow verification and its URL are recorded in the draft PR, avoiding a self-referential report commit. Local correctness does not by itself establish a GitHub Actions PASS.

## Scientific outcomes and final questions

| Question | Answer and status |
|---|---|
| Q1: Why do distractors impair classification? | **NOT TESTED** in Phase VII. Encoder contamination, pooling interference, head sensitivity and training mismatch remain plausible, potentially coexisting explanations. The unique cause is unresolved; no feature drift/probe/head-intervention measurements were taken. |
| Q2: Do exposure or robust aggregation improve it? | **NOT TESTED.** R1/R2 were not fitted, no new per-seed/task scores or oracle upper bounds exist, and R2 was not selected before the required diagnostic predecessor. |
| Q3: Complete conventional robustness prerequisite? | **NOT TESTED** for Phase VII. The unchanged historical Phase VI collective result is **FAIL**, five of six seed/task competitive comparisons below .70. That is not a new Phase VII result. |
| Q4: Additive advantage on an identical encoder? | **NOT TESTED.** VII-4 did not run; VII-5 has zero fits/updates, no matched encoder hashes and no localization/attribution superiority claim. |
| Q5: Faithful explanations? | **NOT TESTED** in this phase. Oracle generator deletion, internal bookkeeping and donor replacement remain different estimands. The original 210 rejected donors stay rejected; accepted-donor sensitivity is unsupported. No matched diagnostic/control intervention was executed. |
| Q6: Another real EEG experiment justified? | **No recommendation to launch.** First recover historical provenance or explicitly revise the study identity, complete frozen-model diagnostics, then preregister and test matched robustness controls. No new evidence here justifies a real-data campaign. |

**INCONCLUSIVE:** the historical performance gap does not identify a unique mechanism. **FAIL:** no new scientific hypothesis was tested and failed; retain Phase VI's original robustness FAIL without relabeling it. **PASS** applies only to the named integrity subchecks and CPU engineering. The overall study is **BLOCKED/incomplete**, not a completed negative robustness experiment.

## Reproduction, scope and next prerequisite

Use the constrained CPU interpreter described in `docs/CI_CPU_V1.md`. `PYTHON` below denotes that environment; detailed interpreter paths and logs remain private.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" -m pytest tests -ra
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" scripts/run_mist_cove_phase7.py \
  --output mist_transfer_runs/mist_cove/phase7_v1/NEW_UNUSED_RECEIPT.json
```

The second command intentionally returns exit code **2** with a BLOCKED receipt. It is inspection-only; no training mode or receipt flag bypass exists. Existing output files are never overwritten. Public reports contain safe synthetic hashes and aggregate/accounting metadata, not weights, complete waveform arrays, RNG states, participant data or private source locations.

The hypotheses/diagnostic draft and deferred robustness protocol are labeled **not executable/not preregistered**. The config allows zero scientific fits. Do not pretend that deferred method selection or missing exposure/count choices are a completed freeze. An existing independently trustworthy manifest/backup receipt binding all six original weights is the immediate missing prerequisite; its location was requested while independent engineering work continued. No such evidence was provided during this execution.

No actual TRAIN/VAL EEG, annotations, data-bearing source descriptors, real normalization/banks, RESERVED, Sleep-Telemetry, SHHS or legacy NPZ was opened. No historical source result, split, checkpoint or training implementation changed. Even after synthetic recovery, a new real EEG study needs separately reviewed source-only protocol and explicit authorization.

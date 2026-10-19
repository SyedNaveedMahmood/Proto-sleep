# Phase VII prospective study v3 — completed at the failed robustness gate

**Decision: preserve the controlled negative result; do not begin additive-head confirmation or real EEG.** Historical Phase VI checkpoint authentication remains BLOCKED. The user explicitly authorized a new prospective reproduction; this study used newly created, sealed models and never loaded the six original unsealed checkpoints.

## Provenance and chronology

Base Phase VI commit: `8ec83fed182f4ca69889db17272eec84d05e9771`; branch `research/mist-cove-phase7-distractor-robustness-v1`; draft PR #9 targets Phase VI. The initial blocked integrity report remains preserved. The bounded known original run-log/receipt search found no independent scientific weight hashes; no backup was supplied. Present-day original-file inventory hashes are not historical seals. All six original files remain byte-unchanged since that inventory. No claim is made to have exhaustively searched external backups.

Actual subsequent user authorization is recorded by `prospective_authorization_reference_v1.json`, not an agent-generated approval. Prospective protocol/source freeze: `48c36f6`; corrected diagnostic freeze: `a3d0502453a4713c0d1a3a12f298a6f76e3a52b2`; findings documented at `842aaf7` before robustness design; locked robustness freeze: `e3fb36e7f8cf955cea30a9f3b4e85f266ca7d5d6`. Diagnostic source/config inventory SHA256 `4ffa501a62d753d4da495346ad5a10f6c08dfd052dcb88d6cdc948181967c704`; robustness inventory `da49d788f1e2ecab338d374768d287cef07287d967d27caac523a9ddfc7a9f5f`. Manifests were committed and verified before their respective scientific execution.

The first prospective invocation completed/sealed single/123 P0, then stopped on a batched localization-reporting interface error. The model, optimizer/RNG, creation proof, failed log and 3.080431-second STOPPED receipt are preserved. v2 corrected only per-example aggregation, added an end-to-end audit regression, pinned the creation proof and reused the completed 300 updates. It did not retrain that model, change a scientific threshold or remove a failed scientific seed. A separate initial CPU shell redirection failure occurred before pytest and was corrected after creating its output directory.

## Mechanism findings: Q1

H7.1 is **INCONCLUSIVE for diagnostic information loss**, with measured local susceptibility: oracle-local nonlinear probe competitive F1 remains≥.80 in all six cases, despite diagnostic-containing feature drift and classification changes. This contradicts complete destruction for the tested privileged decoder, not every form of representational degradation. Probes use their own generated TRAIN representations, a disclosed50/50 nuisance-exposure mixture, TRAIN-only standardization and privileged region selection; they are not matched to the unchanged head in exposure/information.

H7.2's prespecified **dominant non-diagnostic-path prediction is CONTRADICTED**: diagnostic-overlapping candidate changes dominate the two controlled substitution orders in every case. Exact telescoping and linear head-feature logit differences reconcile numerically. This is a restricted model-internal estimand, potentially off-manifold, and does not prove pooling is irrelevant. Candidate windows overlap; diagnostic-containing sets can also contain nuisance.

H7.3 is **INCONCLUSIVE as a universal head explanation**: TRAIN-fitted pooled nonlinear probes improve some weak heads by >.10, but not all comparisons. Extra probe capacity and nuisance exposure confound a unique head-only causal reading. Privileged oracle pooling often worsens unchanged-head predictions; that is not an information upper bound. No exclusive encoder/pooling/head cause was identified.

Strong artifacts independently matter in the frozen stress construction. Matched500-example clean-versus-artifact draws retain diagnostic/background/noise/nuisance and raise only the box amplitude. P0 mean F1 falls .993317→.746011(single) and .978005→.605796(multiple), while artifact confidence remains .912063/.853084. This supports artifact sensitivity in the constructed model, not a unique explanation of combined interference or physiological artifacts. The newly added high-frequency-nuisance condition is different from original combined corruption; easier nuisance-only success never replaces the original endpoint.

## Matched robustness: Q2 and Q3

P0 is the newly authenticated clean-training reference, not historical R0. R1 changes training exposure only. R2 adds one class-blind TRAIN-derived candidate-clipping operation, with the same encoder/head design, initialization and mixed exposure. All neural models have 3,593 trainable parameters, 3,408 in the encoder; R2 adds 36 fixed statistic values from P0 TRAIN features. Their function classes differ. Common P0 fitting/statistics are disclosed and counted once, not hidden as free compute. Linear probes have 95/185 parameters (local 18D/pooled 36D); 32-unit nonlinear probes 773/1,349. Oracle-region probes receive privileged information.

| Task | Seed | P0 clean / combined F1 | R1 clean / combined F1 | R2 clean / combined F1 | Primary R1 full gate |
|---|---:|---:|---:|---:|---|
| single | 123 | 1.000000 / 0.442956 | 0.871084 / 0.739770 | 0.723966 / 0.648867 | FAIL |
| single | 456 | 0.974464 / 0.743687 | 0.756882 / 0.646028 | 0.846699 / 0.749677 | FAIL |
| single | 789 | 0.999500 / 0.621209 | 1.000000 / 0.975442 | 1.000000 / 0.979802 | PASS |
| multiple | 123 | 0.934782 / 0.342265 | 0.708072 / 0.664722 | 0.699901 / 0.667510 | FAIL |
| multiple | 456 | 0.986491 / 0.698406 | 0.834730 / 0.705184 | 0.847017 / 0.715034 | FAIL |
| multiple | 789 | 1.000000 / 0.551413 | 0.978518 / 0.847281 | 0.988006 / 0.867563 | FAIL |

| Task / model | Mean clean F1 | Mean combined F1 | Mean added-nuisance F1 |
|---|---:|---:|---:|
| single / P0 | 0.991321 | 0.602617 | 0.851255 |
| single / R1 | 0.875989 | 0.787080 | 0.792383 |
| single / R2 | 0.856888 | 0.792782 | 0.788555 |
| multiple / P0 | 0.973758 | 0.530695 | 0.810305 |
| multiple / R1 | 0.840440 | 0.739063 | 0.776650 |
| multiple / R2 | 0.844975 | 0.750036 | 0.787219 |

**FAIL collective VII-4: five of six primary comparisons fail.** Only single/789 passes every clause. Single/123 fails strong paired criteria; single/456 fails clean/combined/added F1 and pairs; multiple/123 fails F1 and pairs; multiple/456 fails diagnostic pairs; multiple/789 fails strong diagnostic/nuisance pairs. No full-input collapse or specified nuisance-equivalence failure occurred. R2 is never substituted after viewing scores.

H7.4 shows **mixed exposure effects**, not universal robust success. R1 combined-corruption mean improves by .184462 (single) and .208368 (multiple), while clean means fall by .115332/.133318. Single/456 combined F1 falls .743687→.646028. Thus the preregistered no-negative-seed exposure-support rule fails. R2's universal≥.03 gain over R1 also fails; some seeds improve and others worsen. These failures do not prove every robustness training recipe or robust aggregator is ineffective. No extra updates or method changes were made.

All methods retain per-class recall/F1/confusions, balanced accuracy/kappa, NLL/Brier/ECE, collapse, strong paired outcomes and full stress/localization records in the safe JSONL. Stress covers weak/removed, repeated/correlated, amplitude sweep, shifts, artifacts, eventless, and multiple-only redundant/complementary/adjacent/overlap conditions. Synthetic classes are not clinically scored sleep stages.

## Reliability and evidence: Q4 and Q5

**NOT TESTED VII-5: zero matched additive-head fits or updates.** The prerequisite failed even though an isolated seed passed. No shared-encoder explanatory advantage, clinical evidence discovery, native-score necessity or transferability is claimed. The conditional machinery/tests use original 14 windows/K4 and verify signed accounting/common encoders when invoked; correctness checks are not scientific evidence of benefit.

Under combined corruption, mean incorrect-confidence≥.9 falls from 20.55% to 0% (single) and 11.42% to 1.33% (multiple) for R1. That improvement does not establish trustworthiness: removed accuracy remains about .19, with confidence .554937/.602975. Lower catastrophic confidence can coexist with clean-performance losses and a failed paired-use gate. There is no abstention mechanism or clinical calibration claim.

The prospective frozen P0 diagnostics retain signed individual/joint exact generator-deletion effects, feature/path decomposition and localization. Known component removal is valid for this synthetic generator, not physiological causation. Conventional M1 has no native additive event contribution; all its pooled logits are background in the old additive interface. Controlled same-construction diagnostic/nuisance component audits are implemented and tested as engineering, but their confirmatory learned-head comparison did not run. **Donor robustness remains NOT TESTED**; no new donor procedure or accepted replacement was manufactured, no guard was weakened, and the historical0/210 remains unchanged.

## Actual resources and correctness

**42 unique scientific fits / 12,600 optimizer updates / 201,600 example exposures**: six new P0 fits, 24 restricted probes and 12 R1/R2 fits. P0 rows in the robustness comparison are evaluations of existing prospective fits, not another six fits. Each actual fit has 300 batch-16 updates. The reused failed-audit model is counted once. Diagnostics corrected wall 40.854400 s plus failed attempt 3.080431 s; robustness wall197.193080s; known total241.127911s, below 1,800 s. Two CPU threads, deterministic FP32, AMP/TF32 off. Diagnostic RSS peak 1,388,460 KiB; robustness 1,611,564 KiB. Model-fit measurements include checkpoint work where instrumented; checkpoint IO and evaluation overhead were not independently profiled. Import/setup time outside internal budget timers and GPU-kernel time are not claimed as measured experiment wall.

**PASS full CPU:** 738 passed, 0 failed/errors, 11 CUDA skips, one existing warning, 12.55 s; 65 new cases since Phase VI. Tests cover original full physical grid/masks and role guards, exact matching and nuisance independence, feature paths, clipped geometry/capacity, same-construction controls, strict collective gating, atomic receipts/tamper rejection, and exact M1/R2 model/optimizer/global/minibatch-RNG interruption/resume. Eighteen new neural model checkpoints carry optimizer/RNG and creation-bound identity/seals; weights remain private. All original Phase VI source/publication/result hashes and original observed weight bytes remain unchanged.

**BLOCKED CUDA:** another process still used 14,260 MiB on the actual RTX 4080 SUPER (driver 580.95.05); query total 16,376 MiB, used 14,334 MiB, free 1,611 MiB. No job was interrupted, no credentials were used/stored, no new CUDA model operation/test/smoke/resume ran, and no memory/runtime estimate is presented as a measurement. GPU feasibility beyond historical receipts remains NOT TESTED.

Python 3.12.9, torch 2.5.1+cpu, NumPy import/metadata 1.26.4. Final source/test inventory and JUnit/log hashes are in `prospective_correctness_v3.json`. The unchanged actual final-head Actions check/link is recorded in Draft PR #9; a local PASS is not substituted for CI.

## Recommendation: Q6 and stopping point

**Do not launch another real EEG experiment on these claims.** Partial robustness improvement does not satisfy the complete conventional prerequisite or establish additive evidence benefit. A future synthetic study could separately test training-distribution/optimization tradeoffs with matched budgets; it needs a new identity and cannot rescue these frozen results. Historical Phase III–VI negatives remain intact. No real TRAIN/VAL EEG/annotations/source descriptors, real fitted artifacts, RESERVED, ST, SHHS or legacy NPZ was opened. No source split, historical checkpoint or prior experiment was modified. No target adaptation or external transfer was evaluated.

Reproduction uses the constrained CPU environment and the archived private creation receipts. The corrected diagnostic launcher intentionally reuses its pinned newly sealed first baseline from the preserved failed attempt; replaying that exact operational history on another machine requires that private prospective artifact. A clean from-scratch replay elsewhere needs a new experiment identity/freeze, not a fabricated match to private checkpoint-container hashes. The deterministic model/optimizer/RNG algorithm is tested separately.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" scripts/run_mist_cove_phase7_prospective.py --output mist_transfer_runs/mist_cove/phase7_prospective_v1/NEW_DIAGNOSTIC_OUTPUT
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" scripts/run_mist_cove_phase7_robustness.py --output mist_transfer_runs/mist_cove/phase7_prospective_v1/NEW_ROBUSTNESS_OUTPUT
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" -m pytest tests -ra
```

These show the executed CLI shapes; all current outputs are sealed/preserved and must not be overwritten. Robustness execution verifies the fixed local prospective baseline receipts; changing output names does not create permission for unregistered sweeps. The read-only collector verifies private result/checkpoint digests and projects safe aggregate results; it refuses existing public outputs. No automatic real-data gate follows.

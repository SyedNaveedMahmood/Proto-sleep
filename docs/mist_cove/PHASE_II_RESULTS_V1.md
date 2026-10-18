# MIST-COVE Phase II witness v1 — results and stop gate

Phase II synthetic research and source-runner software preparation are complete. **q=1 remains the primary reference.** Exact path accounting, redundant-group dependence and the rounded selector’s predefined tie-stability comparison pass. **Donor agreement fails for q=1 in five of six task/seed runs**, and strong in-band competition exposes poor diagnostic robustness. Individual waveform necessity and physiological validity remain unestablished. No real EEG training or data access occurred.

## Identity, protocol and preserved evidence

- Date: 2026-10-18, Asia/Dhaka. Authorization: user Phase II request, restricted to code, generated experiments, mock integration and synthetic CUDA checks.
- Base: verified Phase I `1a95a54f95edc6f694836a97091336e1e745e4f4`. Branch: `research/mist-cove-phase-ii-witness-v1`; draft PR base: `research/mist-cove-implementation-v1`.
- The clean checkout was branched directly from the fetched Phase I reference; no reset, destructive cleanup or merge was used.
- Original protocol preregistration commit: `62c3970`; original experiment implementation: `addce7b6327f1a42e3cbbcc99af024a2751300bd`.
- Symmetric-factor amendment preregistered at `24f8f8a`; final corrected scientific implementation: `60a48336dc11fe57263df41cf7b40a15e59dfff2`.
- Final engineering implementation: `34019f3f5b0025ae424a7abd06f09e1880f69e5d`; only GPU total-versus-trainable parameter counting changed after the scientific run. Correctness receipt verifies identical model/generator/audit/training files and identifies the one script difference.
- Original config SHA256: `e6b6a2392735a1898bdb7f37b257e99da5efa900c2110c494ccb645fe1637910`.
- Corrected config SHA256: `ed3536190f348fb14aa894fd367fd2e2d8e5ffe6a081d3e9cccf951f4daf47d7`.
- Prepared source config SHA256: `1e336b0fd166ce3f06ecc7bb23ecd3e704572f7378921592f2de4b4519b8b0e0`.
- Phase I source, configs, reports, results and hypotheses are unchanged. H1 remains FAILED, H2 partially supported, and H3 does not establish gating necessity. Frozen P3B code/config/results and exact split files are unchanged. No checkpoint identity guard was bypassed.
- [Decision/preregistration](PHASE_II_DECISION_V1.md), [generator amendment](PHASE_II_GENERATOR_AMENDMENT_V2.md), [witness specification](PHASE_II_WITNESS_SPEC_V1.md), [source readiness checklist](PHASE_II_SOURCE_READINESS_V1.md).

Aggregate receipts: [original v1](../../reports/mist_cove/phase2_synthetic_original_v1.json), [corrected v2](../../reports/mist_cove/phase2_synthetic_corrected_v2.json), [CUDA resources/resume](../../reports/mist_cove/phase2_cuda_v1.json), [correctness and log hashes](../../reports/mist_cove/phase2_correctness_v1.json). Weights, generated waveform arrays, mock prediction rows and console logs remain local/ignored.

v1 had an asymmetric complementary single-factor generator that always retained A. That policy did not enforce the intended loss of factor identity. Its full results are retained; the flaw is explicitly documented. v2 randomly retains A or B without supplying role identity to the model. Every threshold, model, optimizer budget, seed list and donor comparison is unchanged. The amended protocol was frozen before v2 evaluation. A second v2 execution reproduced every shared training, metric, counterfactual, stability, budget and hypothesis result exactly; extra report metadata and timings were excluded from that comparison.

## What was implemented and tested

The 75-parameter analytic spectral reference retains signed local evidence and a linear interpretable background. q=1 removes fixed availability attenuation. The availability model, background-only and original availability-based fixed-patch function are controls. A 0.5-uV rounded-ranking selector adds no parameters and retains K=4. The report refinement separates A additive bookkeeping, B individual full-model effects, C joint predeclared-group effects, and D set/score stability. It exposes background, fixed-witness internal paths, reselection, signed five-stage terms, donor sensitivity, redundancy and unsupported replacements. It does not force individual scores to equal waveform counterfactual effects.

Each protocol has two separately trained artificial tasks, three seeds, five methods, 150 training examples and 180 full-batch updates per run: **30 runs / 5,400 updates / 810,000 synthetic training exposures per execution**. This is a synthetic recipe, not the source training recipe. Seven cases each have 50 fresh generated examples; counterfactual analyses audit five examples per case (one per class) under three erasers plus a separate phase-preserving negative control. Groups are generator-declared before attribution; clinical group discovery is not tested. Seeds are repetitions, not participant sample sizes.

## PASS — exact estimands and redundancy mechanisms

For original selection S0, replaced selection S1, background B and event total E, the signed full effect reconciles as:

```text
full_effect = [B(X)-B(X~)] + [E(X,S0)-E(X~,S0)] + [E(X~,S0)-E(X~,S1)]
            = background_change + fixed_event_path_effect + reselection_change
```

All 30 runs pass the 1e-5 reconciliation contract; maximum absolute residual is **0.000001907**. Internal fixed-witness/path and score-deletion effects are labeled internal interventions, not physiological counterfactuals. Membership and rank-order changes are distinguished.

W1 material reselection and W2 greater joint than individual dependence pass in all six q=1 task/seed runs. Background-only has zero event reselection, as expected for the mechanism-disabled control. On correlated-copy fixtures, removing one member barely changes the score because alternatives remain, while jointly removing the whole declared group has a large effect:

| Task | Seed | Mean absolute individual true-stage effect | Absolute joint effect | Absolute reselection term |
|---|---|---|---|---|
| substitutable | 123 | 0.0139 | 9.561 | 1.762 |
| substitutable | 456 | 0.0172 | 9.709 | 1.752 |
| substitutable | 789 | 0.0103 | 9.863 | 1.919 |
| complementary | 123 | 0.0132 | 8.811 | 2.322 |
| complementary | 456 | 0.0201 | 8.119 | 2.627 |
| complementary | 789 | 0.0132 | 7.986 | 2.540 |

A useful interval therefore need not be individually necessary in this construction. Joint effects remain donor-conditional and are not equal to the sum of individually recomputed effects. Reordered group effects can also change through global background and selection. Neither a unique individual causal allocation nor clinically meaningful group membership follows from these results.

## PASS — narrow tie-stability refinement; no causal-identification claim

W3 requires at least 0.05 mean Jaccard-distance reduction under tiny noise/ranking jitter on tied copies, with no dual-case accuracy loss over 0.02. All six paired comparisons pass:

| Task | Seed | Hard instability | Rounded instability | Reduction | Dual accuracy loss |
|---|---|---|---|---|---|
| substitutable | 123 | 0.794 | 0.054 | 0.740 | 0.000 |
| substitutable | 456 | 0.749 | 0.021 | 0.728 | 0.000 |
| substitutable | 789 | 0.782 | 0.004 | 0.778 | 0.000 |
| complementary | 123 | 0.776 | 0.009 | 0.768 | 0.000 |
| complementary | 456 | 0.796 | 0.009 | 0.787 | 0.000 |
| complementary | 789 | 0.763 | 0.017 | 0.746 | 0.000 |

These are nearly equivalent generated witnesses: hard-selection identities often changed while logits barely changed. Rounded ranking makes provenance less arbitrary in this specific tie regime; it also favors earlier windows and retains bin discontinuities. Temporal shifts, physical amplitude changes, calibrated inverses, budget changes and reordered witnesses are reported separately. This result does not establish unique witness necessity, clinical robustness, or that rounded selection should replace the primary q=1 reference.

## FAIL — donor agreement and distractor robustness

W4 requires mean full-effect spread across bridge/background/noise erasers at most 0.25. q=1 passes only one of six runs:

| Task | Seed | Mean eraser spread | W4 |
|---|---|---|---|
| substitutable | 123 | 0.287 | FAIL |
| substitutable | 456 | 0.216 | PASS |
| substitutable | 789 | 0.286 | FAIL |
| complementary | 123 | 0.334 | FAIL |
| complementary | 456 | 0.521 | FAIL |
| complementary | 789 | 0.466 | FAIL |

Elementary numerical donor guards passed, and endpoint-preserving blending introduced no boundary-jump change. Nonetheless energy, spectral content, background features and the replacement event pool changed, with substantial donor-dependent effects. Passing continuity checks does not certify physiology. The availability control had lower eraser spreads (about 0.017–0.031) in these fixtures; this is a different estimand and **does not reverse the frozen H1 failure**. Phase scrambling retains spectral diagnostic information and is reported separately rather than treated as an eraser.

All q=1 dual and overlap cases were classified correctly. Strong randomized in-band candidates competing for the event budget reduced q=1 accuracy to **0.20–0.26**, while incorrect predictions could remain confident. Corrected complementary single-factor accuracy was 0.50–0.56 on 50-example sets: loss of the unobserved factor leaves two compatible class codes. High confidence after erasing correlated groups is also exposed in the receipt; no abstention/calibration benefit is claimed.

The strict 1e-5 mean calibrated-gain tolerance fails for the availability control at complementary seeds 123 and 789 (about 1.43e-5 and 1.72e-5). These small FP32/tie-sensitive failures are preserved; the tolerance was not relaxed. Primary q=1 passes all six corrected calibration checks.

## Simple alternatives and calibration

Means below are descriptive across three optimization/generator seeds, not participant uncertainty. All methods instantiate 75 parameters; background-only has 40 active parameters. Historical raw_context/evidence accuracy was not measured on real data; their unchanged definitions were tested only on generated adapter/resume fixtures.

| Task | Method | Dual accuracy | Dual NLL | Dual Brier | Dual ECE15 | Competition accuracy |
|---|---|---|---|---|---|---|
| substitutable | q1 | 1.0000 | 0.0020 | 0.0000 | 0.0020 | 0.2400 |
| substitutable | availability | 1.0000 | 0.0040 | 0.0000 | 0.0040 | 0.2800 |
| substitutable | background | 1.0000 | 0.0419 | 0.0030 | 0.0406 | 0.3067 |
| substitutable | fixed_patch | 1.0000 | 0.0330 | 0.0020 | 0.0322 | 0.5000 |
| substitutable | rounded_q1 | 1.0000 | 0.0020 | 0.0000 | 0.0020 | 0.2333 |
| complementary | q1 | 1.0000 | 0.0381 | 0.0030 | 0.0372 | 0.2200 |
| complementary | availability | 1.0000 | 0.0599 | 0.0067 | 0.0579 | 0.2400 |
| complementary | background | 0.8867 | 0.4096 | 0.2357 | 0.1944 | 0.2667 |
| complementary | fixed_patch | 0.9200 | 0.2715 | 0.1410 | 0.1609 | 0.3400 |
| complementary | rounded_q1 | 1.0000 | 0.0369 | 0.0027 | 0.0361 | 0.2067 |

Background-only already solves the clean substitutable task and reaches 0.86–0.90 on complementary dual pairs. Fixed-patch can outperform q=1 under competitive distractors. These outcomes preserve simple non-morphological explanations; synthetic accuracy does not establish physiological waveform reasoning, transfer, novelty or superiority. Full fixed-five metrics, recall/confusion, kappa, NLL/Brier/ECE, signed counterfactual terms, stability and budget probes are in the aggregate receipts.

## Engineering correctness and actual GPU resources

- Final CPU suite: **357 passed, 0 failed, 11 CUDA skipped, 1 existing PyTorch warning** (6.68 s final engineering run). New Phase II tests contribute 104 CPU passes; the existing 253 passes are preserved.
- Separate CUDA pytest execution: **11 passed, 0 failed, 0 skipped, 64 deselected** (4.07 s), including six new exact-resume cases and five historical CUDA tests.
- Standalone CUDA checks: **six methods PASS** for generated batch-16 finite logits/backward and exact interrupted-versus-uninterrupted recovery of model, optimizer, Python/NumPy/CPU/CUDA RNG, early-best state and semantic validation history. Completed resume is verified too. Synthetic resume uses two epochs and six optimizer steps per uninterrupted fixture; it does not shorten the future source recipe.
- Atomic-save interruption preserves previous bytes; per-output locks, digests, missing/invalid seals, altered config/model/source indices/labels/bank/provenance and forbidden roles are tested. Source mocks verify all 18 future variant/seed invocations use the frozen maximum recipe, full-grid masks, and train/val loaders only. No real-data source artifacts were exercised.
- Corrected scientific execution: **104.07 s CPU wall**, **528,856 KiB process peak RSS**. CUDA standalone checks: **2.03 s wall** for tiny software fixtures, not a throughput benchmark.
- Actual GPU: **NVIDIA GeForce RTX 4080 SUPER**, 16,718,168,064 bytes reported total memory, capability 8.9, driver **580.95.05**, PyTorch **2.5.1**, CUDA build **12.1**, cuDNN **90100**. Deterministic FP32, AMP/TF32 off, CUBLAS workspace `:4096:8`, 512 MiB allocator cap; no CPU fallback.

| Method | Total parameters | Trainable parameters | Batch-16 peak allocated MiB | Resume training peak allocated MiB |
|---|---|---|---|---|
| q1 | 75 | 75 | 64.19 | 65.15 |
| availability | 75 | 75 | 65.80 | 65.15 |
| background | 75 | 75 | 65.80 | 65.15 |
| rounded_q1 | 75 | 75 | 65.80 | 65.15 |
| raw_context | 34197 | 33957 | 182.20 | 94.15 |
| evidence | 13556 | 13556 | 184.16 | 94.96 |

These are this process’s PyTorch allocator peaks, excluding driver/context allocations and real-data I/O. A background-only model still instantiates its unused event head. A model may have parameters that require gradients yet receive none in an ablation; active counts are separately recorded in the scientific receipt. Actual real-source throughput, full-run memory and the future maximum-budget compute cost remain NOT TESTED.

Initial GPU checking was blocked by memory contention and resumed after the user freed the device. The first available-device run passed the four new candidates but exposed a smoke-harness invocation error for historical feature encoders; it was corrected to use encode→emissions. Total versus trainable parameter reporting was also corrected (raw_context has 34,197 total / 33,957 trainable). No baseline definition or scientific threshold changed. The initial background-control metadata mismatch was repaired before confirmatory execution. No unresolved software test failures remain.

Environment caveat: loaded NumPy is **1.26.4**, while installed distribution metadata reports **2.0.1** in the same prefix. Both are recorded, rather than silently equated. Reconcile this environment and regenerate proofs before a real source freeze; historical environments were not modified.

## NOT TESTED / BLOCKED and next gate

**NOT TESTED:** clinically scored sleep accuracy, real waveform physiological validity, real-data redundant-group discovery, source/target calibration, real-source full-grid I/O, real-data GPU throughput, RESERVED generalization and independent-cohort transfer.

**BLOCKED:** actual source preflight/training, RESERVED access, ST and SHHS operations. Phase II grants no such authorization. Real launch requires separate user approval, an independently reviewed launch receipt, unchanged source split proofs, trusted TRAIN-only normalization/banks, reconciled environment and environment-bound correctness/GPU evidence. A synthetic PASS receipt cannot serve as source authorization.

No real TRAIN/VAL EEG, RESERVED SC EEG/annotations/descriptors/statistics, ST or SHHS was accessed. No real normalization or evidence statistic was fitted. Only generated EDF/NPZ regression fixtures and in-memory/mock views were used. The frozen P3B split, implementation, configuration, checkpoints and reports remain untouched. No real source experiment started.

Next: review the donor and distractor failures, decide which predeclared candidates merit a separately authorized source comparison, freeze a new source config/version if changing its list, verify the exact 15/4/55 roles and split digests without opening RESERVED descriptors, bind verified TRAIN-only normalization/banks and actual runtime versions, run the authorized source preflight, then review its result before any training. Preserve seeds 123/456/789, AdamW 3e-4/1e-4, max 60 epochs, patience 12, core/encoding batch 16, clip 5, FP32/no AMP/TF32, radius zero/no CRF and earliest-best participant-average fixed-five VAL F1 improvement >1e-8. Keep unchanged raw_context and evidence as comparisons. Source-only completion never unlocks RESERVED or external cohorts.

## Reproduce without real data

```bash
python scripts/validate_mist_cove_design.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests -ra
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/mist_cove/test_phase2_training.py \
  tests/transfer/test_full_experiment.py tests/evidence/test_model.py -k cuda -ra
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/run_mist_cove_phase2_synthetic.py --protocol-version 2 \
  --output mist_transfer_runs/mist_cove/reproduce_phase2_v2/results.json
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/check_mist_cove_phase2_cuda.py \
  --output-dir mist_transfer_runs/mist_cove/reproduce_phase2_cuda --allocator-cap-mib 512
```

Use fresh output paths. For original v1 reproduction, select `--protocol-version 1` with another unused output. No source-runner launch command is part of this reproduction protocol.

## Changed-file inventory

All 34 paths below are new Phase II files. The source readiness note was subsequently extended within this branch; no Phase I file was modified.

- `configs/mist_cove_phase2_synthetic_v1.json`
- `configs/mist_cove_phase2_synthetic_v2.json`
- `configs/mist_cove_source_prepared_v1.json`
- `docs/mist_cove/PHASE_II_DECISION_V1.md`
- `docs/mist_cove/PHASE_II_GENERATOR_AMENDMENT_V2.md`
- `docs/mist_cove/PHASE_II_RESULTS_V1.md`
- `docs/mist_cove/PHASE_II_SOURCE_READINESS_V1.md`
- `docs/mist_cove/PHASE_II_WITNESS_SPEC_V1.md`
- `experiments/mist_cove/phase2_comparison_matrix_v1.json`
- `experiments/mist_cove/phase2_comparison_matrix_v2.json`
- `experiments/mist_cove/phase2_preregistration_v1.json`
- `experiments/mist_cove/phase2_preregistration_v2.json`
- `reports/mist_cove/phase2_correctness_v1.json`
- `reports/mist_cove/phase2_cuda_v1.json`
- `reports/mist_cove/phase2_synthetic_corrected_v2.json`
- `reports/mist_cove/phase2_synthetic_original_v1.json`
- `scripts/check_mist_cove_phase2_cuda.py`
- `scripts/run_mist_cove_phase2_synthetic.py`
- `scripts/run_mist_cove_source_prepared.py`
- `src/mist_cove/phase2/__init__.py`
- `src/mist_cove/phase2/adapter.py`
- `src/mist_cove/phase2/audit.py`
- `src/mist_cove/phase2/experiment.py`
- `src/mist_cove/phase2/model.py`
- `src/mist_cove/phase2/provenance.py`
- `src/mist_cove/phase2/report.py`
- `src/mist_cove/phase2/source.py`
- `src/mist_cove/phase2/synthetic.py`
- `src/mist_cove/phase2/synthetic_runtime.py`
- `src/mist_cove/phase2/traces.py`
- `src/mist_cove/phase2/training.py`
- `tests/mist_cove/test_phase2_source_mock.py`
- `tests/mist_cove/test_phase2_training.py`
- `tests/mist_cove/test_phase2_witness.py`

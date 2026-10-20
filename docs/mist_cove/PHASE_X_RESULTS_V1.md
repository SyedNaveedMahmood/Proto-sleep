# MIST-WIC Phase X — dependency interface and feasibility results v1

**Decision: engineering interface verified on synthetic mocks; not ready for real-source launch.** Full native identity and hidden-sample noninterference pass. Exact source-geometry circuits work. Random untrained historical-architecture mocks are mostly prior-driven, while constructed global controls require the whole epoch. Historical checkpoint authentication remains BLOCKED; actual-source/GPU feasibility and clinical interpretation remain untested.

## Scope, identities and preserved findings

Started exactly at Phase IX `ac611572c2efea1acfed48465fd482e6715ec57e` after a clean-tree inspection and fetch. New branch: `research/mist-cove-phase10-source-feasibility-v1`. Historical implementations/reports and Phase IX next_steps.md remain unchanged; plan Git blob `e40e646cb027de163d43220e47e51a491316bce2`. Phase X scope/config/geometry foundation7c27037; operator committed3bb3ba0 before identity probes; fixture/response freeze9f9c84d before outcomes. All new source/data are code or generated mocks. No real EEG, annotations, data-bearing descriptors, fitted banks/scalers, actual checkpoint weights or real EEG directories were accessed. No new training, optimizer update, VAL evaluation, RESERVED/ST/SHHS/legacy-NPZ/target access occurred.

Phase IX negative findings are preserved: clean additive superiority, incremental query-consistency robustness, fixed-witness stability, admissible donors and removed-information reliability were unsupported. The historical raw_context source-VAL mean F1 .723511 is an existing four-person development result, not a new or independent clinical test. Phase X makes no new model-performance or novelty claim.

## Dependency and operator findings

Actual default raw_context geometry is169 windows:59 at100 samples,57 at200,53 at400; stride50, dim32. Every support is computed from half-open native unfold intervals. For example, a400-sample window at50 intersects three200-sample atoms. The209 features are192 pooled local channels plus16 summary values and1 amplitude value. Local centering/RMS and GroupNorm depend on the complete individual window, not other windows/batch items. Raw-control inference bypasses anchor similarity and anchor values; bank shapes, encoder/head parameters and amplitude buffers still need provenance.

The partial operator requires all intersected atoms per window, uses mean with fixed native P, excludes invalid windows from max, and uses fixed neutral when no window survives. All17 whole-epoch features are visible only at the full mask; no global RMS/spectral bypass remains. Zero is primary; .01 is a frozen sensitivity control. Fixed source TRAIN affine normalization is assumed before adapter input, never fitted here. Original RawTemporal radius0/no-CRF head and five-stage ordering are retained. See [dependency audit](PHASE_X_DEPENDENCY_AUDIT_V1.md), [operator specification](PHASE_X_OPERATOR_SPEC_V1.md) and [limitations](PHASE_X_OPERATOR_LIMITATIONS_V1.md).

## Mandatory identity/noninterference gates

| Gate | Actual mock result | Limits |
|---|---|---|
| Full-input identity |9 configs (3 seeds × full/no-amplitude/no-summary),4 inputs each; all feature vectors and logits bitwise equal; maximum absolute/relative errors0 |Random synthetic parameters/buffers; no actual source state |
| Hidden-sample noninterference |252 feature/logit comparisons:3 seeds ×6 masks ×7 perturbations ×2 neutrals; all bitwise equal, maximum errors0; every retained embedding checked |Finite probes supplement static support proof |
| Broken global negative controls |15 injected bypass comparisons expose hidden-dependent summaries |Deliberately leaky implementation is rejected, not deployed |
| Native classifier preservation |34,197 total /33,957 originally trainable parameters; zero new trainable parameters; no parameter or definition change |No retraining or calibration fitting |

Tolerance was frozen at atol=rtol=1e-6 before these probes; relative error uses reference magnitude floored at1e-12. Inputs include flat/constant, noise/sinusoids, hidden impulses up to1e4, DC/RMS changes, overlapping/boundary changes, empty/single/alternating/full masks. Full-input identity is not physiological masking validity.

## Exact circuits and support/coverage

| Fixture | Kind | Target | True masks | ROBDD nodes | Signed status | Minimum active retained seconds | Signed / positive coverage |
|---|---|---:|---:|---:|---|---:|---:|
| localized | privileged analytic control | 0 | 16384 | 1 | COVERED | 2 | 1.000000/1.000000 |
| distributed | privileged analytic control | 0 | 4096 | 3 | COVERED | 6 | 1.000000/1.000000 |
| global_amplitude | privileged analytic control | 0 | 1 | 15 | COVERED | 30 | 1.000000/1.000000 |
| high_frequency_nuisance | privileged analytic control | 0 | 16384 | 1 | COVERED | 2 | 1.000000/1.000000 |
| broadband_interference | privileged analytic control | 0 | 16384 | 1 | COVERED | 2 | 1.000000/1.000000 |
| redundant | privileged analytic control | 0 | 24576 | 2 | COVERED | 2 | 1.000000/1.000000 |
| complementary | privileged analytic control | 0 | 8192 | 2 | COVERED | 4 | 1.000000/1.000000 |
| disruptor | privileged analytic control | 0 | 16384 | 4 | COVERED | 4 | 1.000000/0.500000 |
| eventless | privileged analytic control | 1 | 32768 | 0 | PRIOR_ONLY | 0 | 1.000000/1.000000 |
| whole_epoch | privileged analytic control | 0 | 1 | 15 | COVERED | 30 | 1.000000/1.000000 |
| mixed_global_local | privileged analytic control | 0 | 1 | 15 | COVERED | 30 | 1.000000/1.000000 |
| neutral_bias | privileged analytic control | 0 | 32768 | 0 | PRIOR_ONLY | 0 | 1.000000/1.000000 |
| aggregation_sensitivity | privileged analytic control | 1 | 32768 | 0 | PRIOR_ONLY | 0 | 1.000000/1.000000 |
| incorrect_explainable | privileged analytic control | 0 | 16384 | 1 | COVERED | 2 | 1.000000/1.000000 |
| native_123 | random native architecture | 2 | 32768 | 0 | PRIOR_ONLY | 0 | 1.000000/1.000000 |
| native_456 | random native architecture | 4 | 32767 | 15 | PARTIAL_SUMMARY | 2 | 0.937529/0.937529 |
| native_789 | random native architecture | 4 | 32768 | 0 | PRIOR_ONLY | 0 | 1.000000/1.000000 |

All17 tables enumerate32768 masks (557,056 availability queries); every ROBDD reconstructs every row. All displayed signed/positive clauses are universally verified and subset-minimal. Tests independently verify source-geometry OR, AND, blocker, whole-epoch and constant truth tables; XOR and unsupported-response fixtures also pass. No monotonicity shortcut is used. A no-support toy remains NO_SUPPORT rather than being retargeted. Full per-mask tables and graph archives stay private.

Redundant controls give alternative atoms2 or10. Complementary support requires both. Distributed support requires2,10,13. The disruptor is exactly `(A AND B) OR ((A OR B) AND NOT C)` with A=2,B=10,C=6; its signed summary covers all true states, versus .5 for positive-only. The original all-retained active reason requires A AND B, while A AND NOT C and B AND NOT C describe other states. Negative literals are not original active waveform events. The incorrect control predicts0 for declared constructed label4 and still has an exact localized explanation.

Whole-epoch/global/mixed controls require all15 atoms (30s), so exactness does not deliver sparse localization. Eventless, neutral-bias and aggregation-sensitivity controls are prior-only for their original class. Random native mocks123 and789 are also prior-only;456 is supported at every nonempty mask, with four displayed positive clauses covering .937529 of true states. These are untrained biases, not evidence of sleep-stage understanding. Summary coverage1 in a prior-only circuit claims no waveform support. Four clauses are bounded incomplete summaries, not minimum-cardinality proofs or all prime implicants.

## Same-model controls and costs

| Fixture | Native-gradient ranking atoms | Ranking canonical/universal | Greedy seconds | Greedy universal | Random same-size canonical | Positive/signed coverage |
|---|---:|---|---:|---|---|---|
| localized | 1 | True/True | 2 | True | False | 1.000000/1.000000 |
| distributed | 3 | True/True | 6 | True | False | 1.000000/1.000000 |
| global_amplitude | 15 | True/True | 30 | True | True | 1.000000/1.000000 |
| high_frequency_nuisance | 1 | True/True | 2 | True | False | 1.000000/1.000000 |
| broadband_interference | 1 | True/True | 2 | True | False | 1.000000/1.000000 |
| redundant | 1 | True/True | 2 | True | False | 1.000000/1.000000 |
| complementary | 2 | True/True | 4 | True | False | 1.000000/1.000000 |
| disruptor | 3 | True/True | 2 | False | False | 0.500000/1.000000 |
| eventless | 0 | True/True | 0 | True | True | 1.000000/1.000000 |
| whole_epoch | 15 | True/True | 30 | True | True | 1.000000/1.000000 |
| mixed_global_local | 15 | True/True | 30 | True | True | 1.000000/1.000000 |
| neutral_bias | 0 | True/True | 0 | True | True | 1.000000/1.000000 |
| aggregation_sensitivity | 0 | True/True | 0 | True | True | 1.000000/1.000000 |
| incorrect_explainable | 1 | True/True | 2 | True | False | 1.000000/1.000000 |
| native_123 | 0 | True/True | 0 | True | True | 1.000000/1.000000 |
| native_456 | 1 | True/True | 2 | True | True | 0.937529/0.937529 |
| native_789 | 0 | True/True | 0 | True | True | 1.000000/1.000000 |

Native-head gradient×feature attribution uses one cached-head backward per nonconstant score (16 actual backwards across17 rows; the literal constant uses none), propagates mean/max contributions to physical supports approximately, and reports global attribution separately. It is not exact nonlinear reconciliation, a waveform-gradient method or an official external explainer reproduction. Greedy is the unchanged Phase IX SIS-inspired canonical search; random subsets use one frozen seed and matched ranking size, not a distributional efficacy estimate. Query methods share verified tables; standalone canonical queries are still counted by their logical query cost. Generated oracle intervals score alignment only in privileged analytic controls, never select ranking/greedy/random outputs. Random native mocks have no oracle or accuracy labels.

In the interference control the greedy2s canonical subset succeeds only while C is absent; it is not universally positive-sufficient. The signed certificate retains that withheld condition explicitly. A3-atom native-gradient ranking succeeds canonically/universally but includes the disruptive region. This illustrates different fidelity domains, not a general superiority result. Constructed detectors/templates know generated locations and cannot establish learned EEG localization. Strong broadband/high-frequency fixtures are single engineered examples, not robustness benchmarks.

The first comparator export stopped at a literal constant head because autograd had no input graph. Reporting correction91e6639 explicitly returns zero feature attribution for a mathematical constant, versions the freeze, and preserves every prior table/receipt digest. No model, seed, waveform, operator, margin or endpoint changed, and no new response table was generated for recovery. Partial failed backward count and isolated failure time are unavailable; the enclosing3.389663s command is retained as a conservative upper timing record. Successful export processes all17 rows; there is no outcome-driven rerun. Accounting freeze v3 corrects the constant row backward counter from1 to0 and retains all table/attribution values without rerunning them.

## Provenance, feasibility and access boundaries

The metadata review selects original P3B raw_context123 prospectively and records its published best-state digest and frozen split/source identifiers. Actual checkpoint authentication is BLOCKED: the original serialized byte/creation/environment/TRAIN normalization/bank chain has not been independently verified. Separate published Phase III seals may support a different reviewed lineage, not silent substitution. No new source artifact checksum was calculated. See [provenance review](PHASE_X_SOURCE_PROVENANCE_REVIEW_V1.md).

Intel Core i7-14700K,2 threads: benchmark internal wall 4.085524s, peak process RSS 662,532KiB. One default cache21,700bytes; full Boolean grid491,520bytes. Median native forward .002494s; cached masked query .001569s;512 masks .004640s; complete table .303184s. Nine complete timing tables plus three .01 tables add393,216 queries. Study plus benchmark total950,272 complete-table availability queries, not full-network passes. Encoders run at cache/native construction; masked queries reuse local features. No optimizer updates or fitted source values. See [resource report](PHASE_X_RESOURCE_FEASIBILITY_V1.md).

Conditional150-example CPU projections are46.895s fast,47.406s median,115.886s stress scenario. No150-input run occurred. Compiler stress uses a random Boolean table with4,318 nodes, not a worst-case trained model. CPU numbers exclude actual source IO/authentication, and cannot establish one GPU-hour. Neutral .01 changes1/32768 response bits at123 and none at456/789; full targets remain unchanged. The primary policy is unchanged.

Timed behavior/table/comparator/benchmark study stages sum 7.409471s; this is not total phase elapsed, CPU core time or proof/test/setup/report time. CUDA remains **BLOCKED** because another process occupied the device; zero new CUDA model operations and no interruption. No historical CUDA receipt is substituted. Actual trained-source CPU/GPU feasibility is NOT_ESTABLISHED.

Every supported source role is rejected before any path inspection; absolute/symlink paths, bank/checkpoint inputs and attempted release flags cannot unlock readers. Source/model readers and torch/numpy deserialization were patched to fail while exercising the generated mock factory/adapter. Default CLI verifies metadata only. Atomic outputs refuse overwrite, reject nonfinite values, detect tampering and exclude sensitive schemas/paths. This is an accidental-access boundary, not an OS sandbox against arbitrary Python. Future real access needs separately reviewed implementation and release decisions.

## Regression and reproducibility

Complete CPU suite: **968 passed / 11 CUDA skips / 0 failures / 0 errors**, 17.293s JUnit. All880 prior tests remain;88 new tests cover geometry, identity, noninterference, circuits, degeneracy, provenance and access controls. One historical PyTorch warning persists. No prior tracked file is modified.

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CPU_PYTHON=mist_transfer_runs/mist_cove/phase4_v1/ci_cpu_environment/bin/python
"$CPU_PYTHON" scripts/check_mist_cove_phase10.py
"$CPU_PYTHON" -m pytest tests -ra
# Explicit generated-only probes: behavior, circuits, comparators.
# Benchmark script runs in-memory mocks only; --cuda-smoke is idle-only.
# Completed output directories refuse overwrite; preserve every prior attempt.
```

A full-history checkout is needed for pinned Git-object/provenance and historical-diff verification. Default verification never constructs a source reader or loads fitted artifacts. Public reports include only code, geometry, generated aggregate findings and safe metadata. Private synthetic per-mask archives remain ignored. Final pushed commit and exact-head GitHub Actions URL are recorded in the draft PR/final response to avoid self-referential hashes.

## Scientific decision and next gate

X-A architecture supports PASS; X-B mock identity/noninterference PASS; X-C exact finite circuits/controls PASS; X-D metadata review COMPLETE with actual authentication BLOCKED; X-E CPU mock measurements/access denial COMPLETE with CUDA BLOCKED and actual GPU feasibility unmeasured; X-F protocol/report/tests prepared for human review. The execution ledger records all16 tasks, their dependencies, checks, findings and limitations.

Review the [future TRAIN-only protocol](PHASE_X_TRAIN_ONLY_PROTOCOL_V1.md) and operator limitations. It proposes at most150 first-two-per-stage/person original TRAIN epochs, frozen order/coordinates and missing strata, while preserving15 TRAIN/4 VAL/55 RESERVED. Before launch it requires independently authenticated checkpoint and TRAIN-fit lineage, exact dependency/identity/noninterference/neutral review, separately authorized two-input actual-resource pilot, frozen sample/compute/privacy plan and an explicit final human launch decision.

**Stop here.** Zero real EEG or fitted source artifacts were opened. The engineering interface may proceed to human review; this package does not authorize actual artifacts, EEG, VAL, RESERVED/ST/SHHS, model retraining, clinical event claims or Phase XI.

# Phase VIII robust evidence study v1 — research closeout

**Primary B1 robustness FAIL: four of six task/seed cases fail. Confirmatory additive explanation quality NOT TESTED (zero fits). Donor compatibility accepts zero of1,500 attempts; accepted-donor stability NOT TESTED. No new real EEG experiment is justified.**

## Engineering and scientific decisions

Engineering:28 requested deliverables complete; Task18 is correctly gated NOT TESTED; Task23 CUDA BLOCKED. All30 tasks are addressed individually below. 825 CPU tests pass, 11 CUDA skips, zero failures/errors, one existing warning. The final pushed-head Actions URL and exact final commit are recorded in the draft PR and final response to avoid self-referential source receipts.

Scientific: better mean corruption performance and calibration do not satisfy complete clean/robust/paired criteria. Secondary pooling success is heterogeneous. Exact additive/feature accounting and software completion are separate from scientific superiority.

## Identities, chronology and authorization

Base4262a1a9680f9e8c1747363553204aaeb1ce976b. Foundation/diagnostic freezee15d4e8; reporting-only amendment51bbfac; confirmatory training/evaluation implementation2bcfdff3a1b85b3016552f8ef53a13a755f3b2bd; descriptive endpoint freeze72676da. The first diagnostic attempt failed before any completed task/seed row after one150-update probe fit; that attempt is retained/counted. No generator/threshold/model was revised after a scientific failure. The descriptive appendix was committed after locked scores and is explicitly not preregistered confirmation. Source/config inventories and private result hashes are in the execution and freeze manifests.

All18 P0/R1/R2 prospective checkpoints match creation/seal/source/environment identities and regenerated TRAIN streams; R2 fixed buffers reproduce exactly. The first P0 reporting-source exception is independently archived and documented. Original six Phase VI checkpoints remain unauthenticated and were never deserialized. No real TRAIN/VAL EEG, data-bearing annotation/descriptor, real fitted statistic, RESERVED, ST, SHHS, legacy NPZ, independent cohort or adaptation was accessed.

## Every seed/task classification result

Each cell is clean / original combined competitive Macro-F1. Five output codes are synthetic; they are not scored clinical sleep stages. Locked evaluation uses1,000 balanced examples/condition, identical inputs across methods.

| Task | Seed | P0 | R1 | R2 | B1 primary | A1 | T1 | B1 full gate |
|---|---:|---|---|---|---|---|---|---|
| single | 123 | 1.000000 / 0.433349 | 0.869854 / 0.731286 | 0.715794 / 0.638450 | 0.753671 / 0.678202 | 0.550341 / 0.492418 | 0.856215 / 0.811722 | FAIL |
| single | 456 | 0.981976 / 0.722616 | 0.747855 / 0.627467 | 0.840142 / 0.716827 | 0.926616 / 0.816445 | 0.740544 / 0.547910 | 0.624494 / 0.544566 | FAIL |
| single | 789 | 0.998000 / 0.632271 | 1.000000 / 0.982263 | 1.000000 / 0.983236 | 1.000000 / 0.975467 | 0.999000 / 0.873658 | 0.934256 / 0.941097 | PASS |
| multiple | 123 | 0.931719 / 0.348685 | 0.733430 / 0.670399 | 0.711395 / 0.671585 | 0.754244 / 0.659384 | 0.724277 / 0.639221 | 0.865222 / 0.755810 | FAIL |
| multiple | 456 | 0.981974 / 0.662492 | 0.833696 / 0.702889 | 0.838422 / 0.735889 | 0.787818 / 0.636023 | 0.844077 / 0.621033 | 0.921108 / 0.737555 | FAIL |
| multiple | 789 | 0.994000 / 0.531956 | 0.973969 / 0.834179 | 0.984979 / 0.853056 | 0.990024 / 0.900650 | 0.991029 / 0.827479 | 0.936434 / 0.904591 | PASS |

| Task/method | Clean mean ± seed SD | Competitive mean ± seed SD | Worst competitive seed | Worst taxonomy/strength F1 |
|---|---:|---:|---:|---:|
| single/P0 | 0.993325 ± 0.009880 | 0.596079 ± 0.147991 | 0.433349 | 0.066667 |
| single/R1 | 0.872569 ± 0.126095 | 0.780339 ± 0.182414 | 0.627467 | 0.229320 |
| single/R2 | 0.851979 ± 0.142472 | 0.779504 ± 0.180737 | 0.638450 | 0.265201 |
| single/B1 | 0.893429 ± 0.126473 | 0.823372 ± 0.148754 | 0.678202 | 0.348983 |
| single/A1 | 0.763295 ± 0.225193 | 0.637996 ± 0.205967 | 0.492418 | 0.135213 |
| single/T1 | 0.804988 ± 0.161109 | 0.765795 ± 0.202215 | 0.544566 | 0.209097 |
| multiple/P0 | 0.969231 ± 0.033038 | 0.514378 ± 0.157640 | 0.348685 | 0.066667 |
| multiple/R1 | 0.847032 ± 0.120823 | 0.735822 ± 0.086715 | 0.670399 | 0.316085 |
| multiple/R2 | 0.844932 ± 0.136908 | 0.753510 ± 0.092010 | 0.671585 | 0.348601 |
| multiple/B1 | 0.844029 ± 0.127545 | 0.732019 ± 0.146505 | 0.636023 | 0.478536 |
| multiple/A1 | 0.853128 ± 0.133606 | 0.695911 ± 0.114304 | 0.621033 | 0.483949 |
| multiple/T1 | 0.907588 ± 0.037482 | 0.799319 ± 0.091624 | 0.737555 | 0.417390 |

## Primary gate failures and uncertainty

- single/123: clean_floor, clean_penalty, competitive, diagnostic_pairs.
- single/456: clean_penalty, diagnostic_pairs, nuisance_pairs.
- single/789: all clauses pass.
- multiple/123: clean_floor, clean_penalty, competitive, diagnostic_pairs.
- multiple/456: clean_floor, clean_penalty, competitive, diagnostic_pairs, nuisance_pairs.
- multiple/789: all clauses pass.

The worst primary original competitive score is .6360229148234718 (multiple456), not a mean of successful seeds. Original competitive class-swap rates range .52–.93; nuisance agreement .76–.93. Primary clean F1 losses versus P0 exceed .03 in four cases. Seed SDs are descriptive optimization/generator variation. All taxonomy transformations of a draw remain paired;500-resample accuracy-difference intervals resample original draws rather than transformations. An independent100-example generated competitive draw is secondary descriptive evidence, not a second model-selection split. No human population significance, independent clinical replication or universal corruption threshold is claimed.

## Scientific answers

1. **Distractor failures:** local/mixed candidate feature contamination, pooling/head interactions and training exposure all have evidence. Diagnostic-overlap versus disjoint interventions differ. Exact logit sensitivity is affine in pooled features, but neither arithmetic decomposition nor oracle pooling uniquely identifies an encoder/pooling/head waveform cause.
2. **Balanced robustness:** B1 improves mean combined corruption but fails the prespecified no-clean-damage/paired-use contract in four of six. It is not an unqualified improvement.
3. **Damaging artifacts:** at strength1, locked P0 mean F1 is .609035 for transients and .631295 for broadband, versus much easier disjoint HF. B1 remains weakest on weak diagnostics (.705646), broadband (.746831) and overlapping nuisance (.751516). The weakest family depends on model/intensity/seed; these amplitudes are not all energy-matched across distinct families.
4. **Robust pooling:** T1 improves three competitive/clean pairs sufficiently but loses .302122 clean and .271879 competitive F1 at single456. Universal H8.4 FAIL.
5. **Calibration:** B1 reduces original competitive NLL/Brier and incorrect confidence>=.9 in every case versus P0. Single-task incorrect high-confidence mean falls .206333→0; multiple .114000→.007667. Temperature scaling sometimes worsens locked corrupted NLL despite development fitting; robustness and calibration remain distinct.
6. **Information loss:** H8.6 fails multiple123 confidence decrease (.089874<.10). B1 removed accuracy .17–.22 with retained confidence. The development-fit policy accepts89% of removed multiple123 cases at risk .820225; the same policy nearly abstains completely for seed789. This is not reliable recognition of absent information or clinical safety.
7. **Sparse additive explanations:** NOT TESTED because the complete primary prerequisite failed; exact shared-encoder engineering tests provide no superiority evidence.
8. **Donors:** all1,500 candidate attempts fail fixed guards; per-example acceptance coverage0 and conditional agreement/effect spread null. Spectral/energy incompatibility dominates. Erasing an intentional spectral diagnostic can conflict with preserving the original interval spectrum; a new estimand is needed, not retrospective relaxation.
9. **Unresolved:** prioritize sparse diagnostic preservation under overlapping artifacts, stable blind aggregation, then conditional additive utility and diagnostically neutral donor feasibility; clinical construct/transfer validity remains independently unresolved.
10. **Real EEG:** no new launch is scientifically justified on these results. Complete robust prerequisite, relevant matched-head science, donor/construct review, new CPU/CUDA preflight, source provenance/resource review and separate human authorization remain blockers.

## Compute and engineering limits

Known scientific internal wall including the failed probe attempt and descriptive completion: **253.681273s** (confirmatory171.921811s, corrected diagnostics13.099149s, failed attempt.870420s, descriptive completion67.789894s). The confirmatory plus descriptive wall239.711704s is below3,600s. 18 new representation fits +6 completed probes +1 failed-attempt probe =25 scientific fits,6,450 updates and103,200 exposures. The20-update/320-exposure resource pilot is separate; engineering/resume test updates are not scientific fits. 18 prospective reference states are reused without retraining.

Measured confirmatory training32.583958s; per-row evaluation/generation/audits110.365948s; checkpoint writes.318416s; exact read-back.027192s; reference checkpoint restore.042858s. Confirmatory process high-water RSS1,237,764KiB; appendix511,360KiB. Pilot inference100 examples (64+36) .016184s. Import/setup, report rendering and replay verification are outside those campaign wall timers; replay has its own receipt. These are wall measurements, not profiled CPU core/kernel time. GPU was occupied and no Phase VIII allocation, speed, memory or resume result is claimed.

## Per-task audit

### Task 1 — Working branch and identity: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Inspect clean status, fetch Phase VII, verify ancestry, create requested branch and preserve history. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task1. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Exact base4262a1a9680f9e8c1747363553204aaeb1ce976b; diagnostic freezee15d4e8, corrected51bbfac, confirmatory2bcfdff; no historical merge/reset/force operation. **Failures:** No branch-integrity failure.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No branch-integrity failure. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Review draft PR and final-head CI.

### Task 2 — Consolidated foundation: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Compare Phase III–VII reports, public aggregates, source and generated tests. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task2. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Five-phase audit separates source clinical development from synthetic codes and records every historical negative. **Failures:** Clinical and external validation remain absent.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Clinical and external validation remain absent. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use foundation to constrain future claims.

### Task 3 — Trusted prospective states: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Git-anchor creation/source/config receipts; byte-check before restore; regenerate TRAIN streams and clipping buffers. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task3. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 18/18 sealed P0/R1/R2 references verified; model/optimizer/global and minibatch RNG restored; no original Phase VI deserialization. **Failures:** First P0 requires archived reporting-source exception; training/restore ASTs match.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. First P0 requires archived reporting-source exception; training/restore ASTs match. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Retain private creation receipts and historical authentication blocker.

### Task 4 — Distractor taxonomy: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Separate extension of unchanged Phase VI generator with independent class-blind stream. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task4. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 15 named families, four strengths, original3000 samples/fourteen candidates; paired latent/label/geometry tests pass. **Failures:** Multiple and competing constructions intentionally duplicate; grid geometry overlaps.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Multiple and competing constructions intentionally duplicate; grid geometry overlaps. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use fixed matrix without retrospective redesign.

### Task 5 — Graded strength analysis: INCONCLUSIVE

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Matched strength sequences, full metrics, paired draw bootstrap and descriptive localization appendix. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task5. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** All36 states evaluated over60 taxonomy/strength cells; P0 added-HF prediction-switch fraction rises0→.005→.02333→.09833, B1 0→.00833→.02167→.04333. **Failures:** Threshold shapes differ by seed/family; no universal collapse threshold confirmed.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Threshold shapes differ by seed/family; no universal collapse threshold confirmed. **Scientific interpretation:** All36 states evaluated over60 taxonomy/strength cells; P0 added-HF prediction-switch fraction rises0→.005→.02333→.09833, B1 0→.00833→.02167→.04333. **Next decision:** Inspect prespecified curves; freeze any new threshold hypothesis separately.

### Task 6 — Encoder contamination: INCONCLUSIVE

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Frozen P0 features, support-defined candidate sets, privileged TRAIN-only local linear probes, norms/cosines/Fisher separation. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task6. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Diagnostic-overlap strength1 F1 .7986 versus disjoint .9598 in exploratory six-case mean; privileged local probe .5691 versus .8243. **Failures:** Probe capacity/exposure differs from Phase VII nonlinear study; no universal information-preservation result.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Probe capacity/exposure differs from Phase VII nonlinear study; no universal information-preservation result. **Scientific interpretation:** Diagnostic-overlap strength1 F1 .7986 versus disjoint .9598 in exploratory six-case mean; privileged local probe .5691 versus .8243. **Next decision:** Test one limited amplitude/interference hypothesis on new draws.

### Task 7 — Aggregation failure analysis: INCONCLUSIVE

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Same frozen features and unchanged head: mean, max, meanmax, trimmed, bounded and privileged oracle. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task7. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Transient mean F1 .5683→.6079 with trimming; broadband .6266→.6785; many sparse regimes deteriorate. **Failures:** Alternative pooling causes head-distribution mismatch; oracle is no guaranteed bound.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Alternative pooling causes head-distribution mismatch; oracle is no guaranteed bound. **Scientific interpretation:** Transient mean F1 .5683→.6079 with trimming; broadband .6266→.6785; many sparse regimes deteriorate. **Next decision:** Use matched trained T1 as secondary evidence.

### Task 8 — Decision-head sensitivity: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Exact affine feature-to-logit decomposition, two substitution orders, aggregate gradient norms. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task8. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Original diagnostic reconciliation max7.6294e-6; descriptive severe-strength max1.14441e-5 under separately frozen3e-5 arithmetic tolerance. **Failures:** Exact arithmetic cannot uniquely identify waveform causality or separate all interacting stages.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Exact arithmetic cannot uniquely identify waveform causality or separate all interacting stages. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Keep internal and waveform estimands distinct.

### Task 9 — Preregistered comparison: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Commit primary/candidate/optimizer/stream/resource/gate contracts before candidate training. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task9. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** B1 fixed primary;18 fits train before locked scoring; confirmatory source/config manifest validates. **Failures:** No model selection from locked Pareto results.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No model selection from locked Pareto results. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Preserve frozen identity and negative results.

### Task 10 — Balanced exposure training: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** B1 fixed four-stratum exposure, same encoder/head/updates/batch/seed as references. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task10. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Mean competitive F1 improves .596079→.823372 single and .514378→.732019 multiple; clean means fall .993325→.893429 and .969231→.844029. **Failures:** Complete primary gate fails4/6; seed-specific clean penalties and paired use fail.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Complete primary gate fails4/6; seed-specific clean penalties and paired use fail. **Scientific interpretation:** Mean competitive F1 improves .596079→.823372 single and .514378→.732019 multiple; clean means fall .993325→.893429 and .969231→.844029. **Next decision:** No claim of robust improvement without clean damage.

### Task 11 — Artifact-focused training: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** A1 changes only strong stratum to broadband; retain fixed artifact subtypes and equal optimization. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task11. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Single/multiple mean competitive F1 .637996/.695911; no all-case artifact/clean superiority over B1. **Failures:** Distribution-specific gains do not establish universal artifact robustness.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Distribution-specific gains do not establish universal artifact robustness. **Scientific interpretation:** Single/multiple mean competitive F1 .637996/.695911; no all-case artifact/clean superiority over B1. **Next decision:** Retain failed secondary and avoid broad augmentation claims.

### Task 12 — Compact robust pooling: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** T1 fixed feature-wise trimmed mean/max under identical B1 input/minibatch exposure. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task12. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Competitive/clean improvement criterion passes3/6; single456 clean/competitive deltas -.302122/-.271879. **Failures:** Other seeds improve; sparse signal trimming can be destructive.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Other seeds improve; sparse signal trimming can be destructive. **Scientific interpretation:** Competitive/clean improvement criterion passes3/6; single456 clean/competitive deltas -.302122/-.271879. **Next decision:** Do not replace B1 after locked outcomes.

### Task 13 — Clean/robust Pareto analysis: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Report both axes, worst condition/seed/task, calibration, capacity and measured cost. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task13. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 36 rows and per-task seed mean/SD/worst summaries; all3593 trainable parameters. **Failures:** A reporting frontier cannot select a confirmatory winner.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. A reporting frontier cannot select a confirmatory winner. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use tradeoffs to plan one future hypothesis.

### Task 14 — Three seeds and two tasks: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Full unrounded fixed-five compound gate over all six primary rows. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task14. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Only single789 and multiple789 pass every clause; worst competitiveF1 .636022915 (multiple456). **Failures:** No seed averaging rescues primary; missing-case gate fails.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No seed averaging rescues primary; missing-case gate fails. **Scientific interpretation:** Only single789 and multiple789 pass every clause; worst competitiveF1 .636022915 (multiple456). **Next decision:** Keep additive confirmation gated.

### Task 15 — Calibration and reliability: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Separate500-example calibration, fixed61-temperature grid, ECE15, NLL/Brier, confidence/error and rejection curves. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task15. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** B1 corrupted NLL/Brier and >=.9 incorrect confidence improve versus P0 in all six cases; scaling has mixed locked effects. **Failures:** Calibration improvements do not prove robust classification or explanations.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Calibration improvements do not prove robust classification or explanations. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Retain raw/scaled results and coverage/risk.

### Task 16 — Information-loss detection: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Seven named diagnostic-loss regimes; calibrated development-fit rejection policy. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task16. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** B1 deletion-confidence drop >=.10 passes5/6, fails multiple123 at .089874; removed accuracy .17–.22 and confidence remains substantial. **Failures:** Removed/high-energy conditions share one baseline; ordered partial-factor rule remains identifiable.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Removed/high-energy conditions share one baseline; ordered partial-factor rule remains identifiable. **Scientific interpretation:** B1 deletion-confidence drop >=.10 passes5/6, fails multiple123 at .089874; removed accuracy .17–.22 and confidence remains substantial. **Next decision:** Do not claim absence recognition or clinical safety.

### Task 17 — Matched-head engineering: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Generated-fixture E0/E1/E2/E3 with shared frozen encoder, matched exposure/K4 and exact signed accounting. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task17. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** All four engineering paths pass; no eventless fabricated witnesses; disabled pathway verified; exact encoder hashes match. **Failures:** Fixture checks cannot demonstrate explanation superiority.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Fixture checks cannot demonstrate explanation superiority. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Retain ready pipeline for a future successful parent.

### Task 18 — Confirmatory explanation quality: NOT TESTED

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Require collective B1 robustness PASS before any head fit. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task18. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 0 confirmatory E0/E1/E2/E3 fits and0 updates. **Failures:** Primary prerequisite FAIL; no replacement head scores manufactured.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Primary prerequisite FAIL; no replacement head scores manufactured. **Scientific interpretation:** 0 confirmatory E0/E1/E2/E3 fits and0 updates. **Next decision:** Require new successful preregistered parent before execution.

### Task 19 — Independent donor methodology: FAIL

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** TRAIN/calibration-derived envelope; three fixed donor populations; energy/spectrum/boundary guards. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task19. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 0/1500 attempts accepted across300 task-specific draws; spectrum and energy dominate rejection; boundary bridge continuity alone insufficient. **Failures:** Accepted-donor spread/agreement NOT TESTED; task inputs share nuisance streams; numerical neutrality differs from diagnostic neutrality.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Accepted-donor spread/agreement NOT TESTED; task inputs share nuisance streams; numerical neutrality differs from diagnostic neutrality. **Scientific interpretation:** 0/1500 attempts accepted across300 task-specific draws; spectrum and energy dominate rejection; boundary bridge continuity alone insufficient. **Next decision:** Investigate compatibility-versus-erasure conflict prospectively.

### Task 20 — Failure-analysis suite: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Explicit overlapping categories with independent100-draw audit, fixed first-per-class and first-error synthetic previews. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task20. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** B1 localized-but-wrong109/600; high-confidence errors3/600; oracle-correct/full-wrong48/600; 36 private synthetic previews. **Failures:** Internal IoU hit is lenient; categories overlap and repeat draws across models.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Internal IoU hit is lenient; categories overlap and repeat draws across models. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use reproducible error groups rather than curated successes.

### Task 21 — Statistical/sensitivity analysis: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Per-seed sample SD/worst cases, independent secondary draws, 500 paired draw-cluster bootstrap, position/overlap/amplitude/count analyses. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task21. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Every strength retains paired accuracy intervals; independent competitive draw scores and all task/seed results published. **Failures:** No participant significance; shared transformations/tasks are correlated; no full F1 population CI.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No participant significance; shared transformations/tasks are correlated; no full F1 population CI. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Require fresh generator-family and external validity before broad inference.

### Task 22 — Regression/provenance expansion: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Preserve all prior tests; add pairing, metrics, gating, resumes, donors, publication, tamper and scope checks. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task22. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 825 CPU passed, 11 CUDA skipped; no failures/errors; one existing warning. **Failures:** CPU correctness cannot substitute for CUDA behavior.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. CPU correctness cannot substitute for CUDA behavior. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Require green final pushed-head CI.

### Task 23 — Bounded CUDA correctness: BLOCKED

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Query occupancy before any Torch allocation; bounded checker prepared. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task23. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** RTX4080SUPER occupied by another Python process,15898MiB;0 new CUDA model operations. **Failures:** Forward/backward/determinism/resume/memory for Phase VIII NOT TESTED.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Forward/backward/determinism/resume/memory for Phase VIII NOT TESTED. **Scientific interpretation:** RTX4080SUPER occupied by another Python process,15898MiB;0 new CUDA model operations. **Next decision:** Leave other job intact; rerun only when an appropriate device is idle.

### Task 24 — Compute efficiency: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Two-thread deterministic CPU; actual fit/eval/I/O/RSS and remaining-budget appendix timing. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task24. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Confirmatory171.921811s; descriptive67.789894s; known scientific/failed-attempt internal wall253.681273s;25 fits including failed probe,6450 updates/103200 exposures; pilot20 updates/320 exposures separately. **Failures:** Interpreter import/setup, source replay/report rendering and test CPU time are separate; no GPU/kernel measurement.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Interpreter import/setup, source replay/report rendering and test CPU time are separate; no GPU/kernel measurement. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use measured pilot for any new scope.

### Task 25 — Comprehensive report: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Machine-readable task audit plus versioned Markdown and separate engineering/scientific decisions. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task25. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** All30 tasks addressed;28 engineering deliverables completed, one conditional experiment NOT TESTED and CUDA BLOCKED. **Failures:** A completed failed study is not a scientific PASS.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. A completed failed study is not a scientific PASS. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Review complete negative results.

### Task 26 — Reproducible visualization: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Deterministic charts from aggregate receipts; detailed traces stay private. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task26. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 16 public aggregate charts cover requested14 themes plus group quantiles/localization; input/hash manifests;36 private synthetic previews. **Failures:** Plots reuse correlated draws; no private model/EEG/waveform exports.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. Plots reuse correlated draws; no private model/EEG/waveform exports. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Read labels and estimator limitations.

### Task 27 — Offline decision dashboard: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** JSON and static Markdown, verified aggregate inputs and gate recomputation. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task27. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** All candidates, task/seed scores, worst cases, calibration, donor, provenance, CPU/CUDA and authorization exposed. **Failures:** No telemetry/server; early snapshot preserved and final snapshot versioned.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No telemetry/server; early snapshot preserved and final snapshot versioned. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Use dashboard as receipt index.

### Task 28 — Architecture-gap analysis: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Compare documented compact and historical source architectures; separate all hypothesized mechanisms. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task28. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** Representation/receptive-field/context/pooling/exposure/artifact/ranking/calibration/donor gaps documented. **Failures:** No direct synthetic-versus-clinical F1 ranking; larger model is not automatically justified.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No direct synthetic-versus-clinical F1 ranking; larger model is not automatically justified. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Prioritize one sparse-signal-preserving interference control.

### Task 29 — Future source-only proposal: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Write reviewable gates, candidate principles, frozen source contract, provenance and resource requirements. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task29. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** 15TRAIN/4VAL/55RESERVED, originalFpz-Cz/100Hz/3000 physical samples and TRAIN fitting preserved; repeatedly used VAL disclosed. **Failures:** No real loader invoked; proposal does not authorize execution.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No real loader invoked; proposal does not authorize execution. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Separate human authorization required after scientific blockers.

### Task 30 — Research roadmap: PASS

**Experimental identity:** phase8-robust-evidence-v1; diagnostic-v2; confirmatory-v1; descriptive-completion-v1 as applicable. **Input data source:** Generated Phase VI waveforms and separate Phase VIII interventions; historical Git/public reports for foundation/proposals; no real data access.

**Method:** Rank three possible directions by evidence, identifiability and feasibility. **Frozen hypotheses:** H8.1–H8.8 as applicable; no scientific superiority inferred from engineering Task30. **Metrics:** Explicit scores/counts/timings in actual_results and linked aggregate receipts; no invented missing endpoint.

**Actual results:** A robust representation first; B conditional evidence organization; C clinical validity/transfer later. **Failures:** No automatic Phase IX or parallel launch of all directions.

**Uncertainty:** Seed/task variation and draw-cluster paired intervals where measured; engineering status is scope-limited. **Limitations:** Synthetic family only; no physiological/clinical/external claim. No automatic Phase IX or parallel launch of all directions. **Scientific interpretation:** Defined deliverable complete within scope; see distinct scientific gate decisions. **Next decision:** Review next two or three limited questions.

## Artifacts and reproduction

Full aggregate rows, per-class recall/confusions, NLL/Brier/ECE, calibration curves, risk/coverage, independent draw scores, gate clauses, donor rejection, candidate metrics and group-effect quantiles are in [the report directory](../../reports/mist_cove/phase8_v1/). [Final dashboard](../../reports/mist_cove/phase8_v1/dashboard_v2.md), [foundation](PHASE_VIII_RESEARCH_FOUNDATION_V1.md), [architecture gaps](PHASE_VIII_ARCHITECTURE_GAPS_V1.md), [future source proposal](PHASE_VIII_FUTURE_SOURCE_PROPOSAL_V1.md) and [roadmap](PHASE_VIII_RESEARCH_ROADMAP_V1.md) separate future hypotheses from authorization. Detailed models, optimizer/RNG, logits, complete generated traces and36 trace previews stay ignored locally.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
CPU_PYTHON=mist_transfer_runs/mist_cove/phase4_v1/ci_cpu_environment/bin/python
"$CPU_PYTHON" -m pytest tests -ra
# Existing completed output paths refuse overwrite. Fresh replay needs the sealed
# prospective reference artifacts, or a separately identified authorized reproduction.
"$CPU_PYTHON" scripts/run_mist_cove_phase8_diagnostics.py --output mist_transfer_runs/mist_cove/phase8_v1/NEW_DIAGNOSTIC
"$CPU_PYTHON" scripts/run_mist_cove_phase8_robustness.py --output mist_transfer_runs/mist_cove/phase8_v1/NEW_CONFIRMATORY
# Aggregate regeneration is reviewed through the archived private result hashes.
```

These are reproduction interfaces, not permission for repeated model-selection sweeps. Public code/config/data-generation streams are portable; exact private Torch checkpoint-container hashes require preserved private artifacts. A clean-room new reference reproduction must carry a new prospective identity, never fabricated historical seals.

# Experimental plan and decision rules

**Status at branch creation:** no real EEG run of MIST-Evidence has been completed. Software tests and synthetic runs are not sleep-science results. These rules are proposed before new branch results, not an assertion of formal registry preregistration.

## 1. Correct the evaluation unit

All 20 Sleep-EDF-20 subjects have participated in the previous development program, often in training for other rotating folds. Changing the fold number does not create a fresh, independent confirmatory cohort. Use this dataset for debugging and architecture selection only. Do not pool previous positive screens as if they were untouched test results.

The implemented development loader opens only train and validation records for the current manifest. It has no test-scoring switch. That protects the CURRENT split, but cannot undo the historical adaptive use of this small cohort.

For final claims, freeze a new source-training cohort, source-validation cohort, and an external cohort never used to choose this architecture. Record recording IDs, subject/family IDs, montage, sampling, units, label mapping, trim policy, gap indices, and overlap exclusions. Sleep-EDF-20 versus overlapping Sleep-EDF-78 is not an independent external-transfer comparison.

## 2. Primary hypotheses

H1, evidence value: source-anchored waveform matches improve mean subject Macro-F1 beyond explicit spectral/amplitude features alone under the same temporal decoder.

H2, local geometry value: the constrained local neural comparison improves over direct observed comparisons without reducing expert-rated match validity.

H3, context value: explicit temporal contributions and CRF decoding improve staging beyond independent epochs, without hiding local errors behind sequence priors.

H4, interpretability: reported emissions are exactly explained by the named inputs, and high-scoring waveform matches agree with independent event annotation where available.

H5, transfer: after the recipe is frozen, the evidence model reduces performance degradation on an external recording environment relative to properly matched baselines.

H1-H3 are development decisions. H4 contains an exact software property and a separate unproven clinical property. H5 is the original MIST-Sleep motivation and remains untested by this branch.

## 3. Implementation inventory

| Variant | What changes | Available now |
|---|---|---|
| `evidence` | Source waveforms + constrained learned comparison + summary + additive context + CRF | Yes |
| `summary_only` | Removes all waveform-match inputs; summary features and identical decoder remain | Yes |
| `observed_only` | Neural dissimilarity contribution is zero | Yes |
| `no_summary` | Only prototype match evidence, no direct summary/amplitude path | Yes |
| `no_amplitude` | Removes the explicit training-scaled amplitude variable | Yes |
| `no_context` | Radius zero and no CRF; independent epochs | Yes |
| `no_crf` | Keeps additive context; removes output-chain dependency | Yes |
| `raw_context` | Uses unconstrained local embeddings and nonlinear temporal convolutions | Yes |
| Random anchors | Change `--anchor-method medoid` to `random`, otherwise same model | Yes |
| Legacy AttnSleep | Existing repository implementation; matched new evaluation adapter | Adapter planned |
| WaveSleepNet / ProtoSleepNet | Authors' implementations and matching single-channel protocol | Planned, not substituted with ours |
| Modern full-night reference | U-Sleep/U-Time or a current reproducible single-EEG reference under matched data | Planned |
| MorphSpec initialization/auxiliary loss | Not in default architecture; new matched ablation would be required | Deferred |
| Annotated event benchmark | Clinician-labeled waveforms and detection metrics | Planned; labels unavailable here |
| External locked evaluation | Frozen train/validation/test protocol and scoring program | Planned; development test access deliberately disabled |

`raw_context` is a capacity/representation control, not a reproduced AttnSleep or SOTA result. Its local encoders share the same initialization as the evidence model under the same seed, but the model is not parameter-matched. Parameter counts are exported. A capacity-matched reference must be added before attributing a final accuracy difference solely to interpretation constraints.

## 4. Stage E0: software and input audit

Run the tests and synthetic end-to-end smoke on CPU, then on the PC's CUDA device. Required checks include exact CRF enumeration, padding, source identity, prototype immutability after training, input locality, halo/full-segment equality, additive score completeness, constant signals, split isolation, epoch-gap handling, save/load, and deterministic interrupted resume.

Run data preflight. Do not run temporal experiments when removed epochs have lost their original indices. Obtain those indices from preprocessing or use epoch-only operation. Record the actual units and montage before making any clinical waveform interpretation.

Stop on non-finite loss/gradients, mismatched fingerprints, unexpected input shapes, source/validation overlap, or failed completion verification. Do not automatically restart from a corrupt checkpoint.

## 5. Stage E1: hardware and pipeline smoke

On fold 0, seed 123, run `evidence`, `summary_only`, and `raw_context` for two epochs with an explicitly separate output directory. Inspect epoch logs, batch timings, memory, predictions, and exported waveform overlays. This is an integration check, not a paper result or a winner-selection experiment.

Measure seconds per epoch on the actual PC before estimating a multi-fold runtime. Do not reuse prior tiny cached-vector timings to estimate this raw-window model.

## 6. Stage E2: first architecture comparison

Initial development comparison: folds 0-4, seeds 123/456/789, fixed recipe. Start with evidence, summary-only, and raw-context controls. Average seeds within each validation subject/fold first. Report every subject, not only pooled epochs.

Provisional retention rule for the evidence bottleneck: mean improvement over summary-only at least 0.005 absolute Macro-F1 and positive in at least 4/5 development folds. This threshold is an engineering decision, not proof of statistical significance. It is stricter than "one positive run".

If evidence fails this comparison, do not call the waveform bank useful. Inspect training-only coverage, redundancy, local match validity, and amplitude dependence. At most one documented repair may be selected on development data before a new fixed comparison. Do not advance to an external claim by removing a difficult validation subject.

The raw-context comparison measures the price of the evidence constraint. A deficit greater than 0.01 on mean subject Macro-F1 triggers a design review before pursuing SOTA. A smaller deficit may still support a useful interpretable model, but not a SOTA claim.

## 7. Stage E3: components must earn their place

Only after E2 is mechanically healthy, run observed-only, no-summary, no-amplitude, no-context, no-CRF, and random-anchor controls on the same development protocol. Do not cherry-pick the most favorable subset of folds.

- Remove the learned metric if it adds no consistent gain or makes visual matches less credible.
- Remove CRF if it adds no consistent gain, excessive boundary errors, or collapses short transitions.
- Remove amplitude evidence if apparent gains are explained by recording-unit shortcuts or unstable gain sensitivity.
- Remove measured summaries from the headline explanation only if the anchor-only model actually supports doing so. Otherwise retain and disclose them.
- Drop the prototype claim entirely if summary-only matches or exceeds the evidence model consistently.

For positive component retention, use the same provisional 0.005 mean / 4-of-5 direction rule. For accuracy-neutral simplification, prefer the simpler model; record the decision rather than search until a difference appears. Exact score accounting must always pass.

No mandatory auxiliary losses are included. A teacher-student/distillation extension is not implemented or assumed to fix a deficit. Any such change would create a new registered model version and require comparison to the same teacher trained without the morphology constraints.

## 8. Stage E4: validate what the waveforms mean

Before naming a prototype clinically, obtain at least two independent expert reviews of source anchors and high-matching windows from OTHER subjects. Sample displayed cases according to a fixed policy and include low-score, false-positive, artefact, and error cases. Do not show only visually attractive examples.

Evaluate event-level precision, recall, duration/interval agreement, inter-rater agreement, and stability across subjects where annotated spindles, K-complexes, or slow waves are available. Define the annotation manual, matching tolerance/IoU rule, and acceptance thresholds BEFORE reading event-test labels. Automatic detectors can provide development labels but are not independent clinical ground truth.

Mechanistic audits must distinguish:

1. Exact emission-score reconstruction from every displayed feature/lag term.
2. Model-feature deletion: zero a selected evidence variable and recompute outputs.
3. Raw-signal interventions: remove or replace a supported interval and recompute ALL affected overlapping windows and sequence predictions.
4. Random-location and matched-duration interventions, including energy-preserving replacements, to detect trivial corruption effects.
5. Observational expert agreement on event identity.

Only (1) is guaranteed algebraically. None of the other tests establishes clinical causality on its own. Mean similarity is not spindle density or event count.

## 9. Stage E5: freeze and evaluate for accuracy and transfer

Freeze the complete code hash, subject manifests, preprocessing, bank-construction seed, supervised seeds, checkpoints, epoch selection rule, metric definitions, comparison list, and clinical naming policy. Use at least three training seeds, but do not count them as independent test subjects.

Compare actual reproduced models under the SAME EEG channel(s), epoch inclusion/trim policy, sampling, annotation mapping, context availability, pretraining data access, and test subjects. Keep EEG-only comparisons separate from EEG+EOG+EMG foundation-model comparisons. Use author code/checkpoints where license and input conventions permit; otherwise explicitly label reimplementations.

Primary statistic: paired difference in mean subject Macro-F1 on the independent evaluation cohort. Report uncertainty across subjects, seed variation separately, per-stage F1, accuracy, kappa, calibration, transition/boundary performance, and inference cost. Bootstrap subjects/families as appropriate, not epochs. Correct multiple confirmatory comparisons with a prespecified procedure. Overlapping development folds do not supply independent test evidence.

SOTA is allowed only if the independent comparison against the strongest eligible reproduced reference supports a positive primary effect with the prespecified uncertainty criterion and no hidden data/modalities advantage. There is no universal score that proves SOTA on Sleep-EDF-20.

If the model is accurate but its waveform explanations fail validation, call it an evidence-based classifier without clinical interpretation. If it is interpretable but weaker, report the accuracy/interpretability tradeoff. If neither holds, drop the architecture. Do not present expected outcomes as results.

## 10. What to do now

Run E0 and E1, not all stages overnight. The branch has not been tested on the main PC or its EEG files. New architecture selection stays on already-used development data. No new test cohort is consumed just to answer whether the code runs.

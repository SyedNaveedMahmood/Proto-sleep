# Witness claims and local audit interface

The default reference is q=1, with exactly the Phase I spectral encoder and 75 trainable head parameters. The fixed availability comparison remains available, with no claim that it estimates stage-pair observability. Phase I findings and artifacts remain unchanged. The primary scientific change is the meaning and scope of the evidence report, not a more complex predictive architecture.

## What one waveform can truthfully establish

An additive contribution is the score the model assigns to a selected local basis element under its selection rule. A full counterfactual effect is dependence on replacing a specified part of the input under a specified donor procedure. With equivalent witnesses, individual necessity need not follow from a large selected score. Several different local assignments can support the same total score. The appropriate report is therefore conditional and plural: show original scores, actual individual interventions, declared joint interventions, and their stability and donor sensitivity separately. A group must be specified by its membership or a generator rule before evaluating its effect; selecting a group to maximize a desired outcome is not validated here.

Let X~ be the jointly replaced waveform, S0 the original witness identities, S1 the newly selected identities, B background logits and E summed local evidence. Each term below is a signed five-stage vector:

```text
full_effect = [B(X) - B(X~)]
            + [E(X,S0) - E(X~,S0)]
            + [E(X~,S0) - E(X~,S1)]
            = background_change + fixed_event_path_effect + reselection_change

fixed_witness_effect = background_change + fixed_event_path_effect
```

The second equation holds background and selection mechanisms distinct. Fixing S0 preserves original identities and validity even if a replacement is flat; it is an explicit **internal model intervention**. Removing the original score C(G) while leaving every other score untouched has effect C(G) by construction, but it is not waveform replacement. There is no general identity equating C(G), a full joint waveform effect, or the sum of individually recomputed effects. The implementation checks the identities that actually hold to absolute tolerance 1e-5.

## Independent API

`phase2.model.WitnessModel` provides the original hard selector and a rounded-score alternative with no added parameters. `WitnessSet` explicitly carries half-open local sample intervals and validity for an internal fixed-identity forward pass. It rejects wrong geometry, off-grid intervals, duplicates, overlap, budgets, dtypes and devices. No model path opens data or fits a normalizer. The source adapter restores physical uV from separately verified frozen TRAIN statistics.

`phase2.audit.audit_group(model, x, intervals, ...)` computes original, replaced, and replaced-with-fixed-identities outputs. Interval unions ensure overlapping groups replace every physical sample once. All replacements are constructed from the original waveform before applying the joint change, so donor selection cannot depend on earlier replacements. Stage contributions remain signed. Background change, fixed-path change, reselection and reconciliation error are exposed separately. Both selection order and membership change are reported; the compatibility field `selection_changed` includes order changes, while `selection_membership_changed` is the set comparison. W3 Jaccard calculations always use sets.

`phase2.report.local_evidence_report` combines explicitly named quantities:

- **A_additive_contribution:** the original selected score vector, counting each intersecting selected witness once.
- **B_individual_counterfactual_effects:** one full-model replacement per declared interval.
- **C_group_counterfactual_effect:** one joint replacement of the declared union, with its own complete re-evaluation.
- **D_attribution_stability:** selected-set Jaccard distance, signed event-total and logit changes under caller-declared transformations. Absent transformations are marked NOT_TESTED.

The report includes donor-wise estimands, per-stage donor spread, score-versus-full disagreement, material reselection, unsupported donor guards and a warning when individual necessity is not established. It exposes these diagnostics independently of probabilities. The redundancy warning is descriptive, not a physiological test. The simpler `audit_group` and experiment stability functions can also be used directly; reports never export waveform bytes.

`phase2.traces.source_ledger` joins metadata-only subject/recording/night identities, original epoch positions and scoring masks to exact inferred physical-input hashes. Raw source-file hashes and physical-input hashes are distinct fields. Channel is Fpz-Cz, units uV, indices are half-open and absolute sample offsets are `3000 * original_epoch_index + local_offset`. Reserved/external roles are rejected. Real source provenance verification belongs to the separately authorized loader, and real traces must remain local/ignored.

## Donor interpretation

Bridge, explicit same-epoch background and independently generated background noise are separate erasure procedures. Endpoint-preserving blends avoid introducing a new boundary jump but can retain part of the original signal and alter global spectral features. Energy mismatch and spectral distortion are measured, not assumed absent. The noise scale is a synthetic fixture constant, never fitted from real data. Finite replacements that fail elementary energy/slope checks remain visibly unsupported in diagnostic audits; no physiological conclusion follows from passing those checks.

Phase scrambling is an information-retaining negative control for a spectral model. It is deliberately **not** grouped with erasers when judging donor agreement. Replacement-dependent effects must be reported as such. If donor sensitivity fails the frozen criterion, report FAIL; do not tune donors to match advertised contribution magnitudes.

## Boundaries of the improvement

Rounded ranking is a deterministic 0.5-uV score bin with physical-time tie breaking. It can reduce arbitrary provenance switching, but it also favors earlier equivalent windows and has bin discontinuities. It does not solve individual identifiability under redundancy, group discovery in clinical data, artifact rejection, physiological validity, or generalization. Temporal-shift diagnostics use generated signals away from boundaries, zero padding, and separate physical-sample alignment metrics; they do not authorize relabeling shifted clinical epochs. Budget probes use K=2/4/8 and hold the per-event scale at 2 so that score normalization does not confound the selection comparison.

The v1 generator's asymmetric complementary-factor removal was corrected in a separately preregistered v2 amendment. Both receipts are preserved. The results report distinguishes numerical/software contracts, supported mechanisms, contradicted prespecified hypotheses, untested clinical questions and blocked source access.

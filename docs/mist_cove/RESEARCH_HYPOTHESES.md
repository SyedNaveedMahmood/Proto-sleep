# Falsifiable research hypotheses

These are constructed proposals, not confirmed discoveries or literature-novelty claims.

## Generative framing
Let latent physiology Z, participant biology U, and acquisition D generate recorded EEG X = F_D(Z,U)+noise. Scored stage Y also depends on time/history H, possible unavailable modalities V, and convention A. Therefore X- and Y-shift are distinct, and not all true distinctions are identifiable from one EEG lead.

With class-conditional event distributions P_a and P_b, a common information-erasing channel cannot increase total-variation distinguishability. With balanced class priors, Bayes pairwise error equals (1-TV(P_a,P_b))/2. This is a theoretical bound under these assumptions, **not** a promise that measured EEG likelihoods are known. A shared invertible transformation preserves the likelihood ratio with correctly transformed densities; noninvertible filtering may not.

## H1 — conditional observability (primary mechanism)
Claim: event- and stage-pair-specific evidence availability under acquisition changes yields better *selective calibration* than indiscriminate representation consistency.
- Manipulation: known benign scale/calibration, controlled targeted attenuation, and matched-power nuisance changes.
- Distinguishing prediction: pairwise risk/reliability degrades selectively on distinctions using attenuated evidence; good performance on harmless changes is retained.
- Competitors: raw_context, evidence, q≡1 ablation, global quality/entropy baseline, unconditional consistency using matched budget.
- Falsifiers: availability tracks distortion power only; pairwise estimates are no better than entropy and simple quality; worsened clean accuracy or risk at identical coverage.
- Caveat: a per-class gate is not automatically a valid pairwise information estimator. Directly test the mapping; no causal interpretation without interventions.

## H2 — faithful, source-traceable witnesses
Claim: selected waveform windows support the advertised signed class evidence in model-level counterfactual tests.
- Manipulation: remove/replace highlighted intervals with background-compatible donors; compare length/energy-matched random and similar nonselected intervals; vary donor strategy.
- Prediction: claimed positive evidence decreases the relevant stage logit more than controls. Negative evidence behaves oppositely.
- Falsifiers: highlighted events have negligible effects, are indistinguishable from controls, or attribution depends strongly on OOD replacement artifacts.
- Caveat: faithful model use != true physiological causation. Exemplar similarity != stage diagnosis.

## H3 — substitutable evidence
Claim: two different valid witnesses can support the same stage without forcing the same local patch correspondence.
- Manipulation: in constructed dual-witness examples, separately attenuate A, separately B, and both; match energy of controls.
- Prediction: reliance shifts from A to B or vice versa when one is lost; both lost increases uncertainty.
- Competitors: fixed-patch evidence, raw_context, architecture-matched independent event weighting.
- Falsifiers: routing does not change, shifts to nuisance patterns, or keeps unjustified certainty with neither witness.

## H4 — optional epoch-phase mechanism
Claim: segmentation origin introduces non-physiological variation into local witness extraction.
- Manipulation: shift analysis grid of **continuous synthetic** EEG first; compare event location/reconciliation.
- Prediction: continuous event representation is less phase-sensitive at boundaries.
- Falsifiers: effect negligible after equal-context windowing.
- Critical: altered 30-second grids are not automatically assigned the old human labels; any real-label test needs correct rescoring/eligibility and separate authorization.

## Negative outcomes and interpretation
- If source evidence gain does not beat raw_context, do not call morphology necessary.
- If external zero-shot gains fail, do not claim transfer even if witness metrics improve.
- If q head learns stage shortcut, remove/restrict it rather than claiming observability.
- If a matched simple calibrated model wins, use it; novelty is not the selection criterion.
- If external cohort cannot be verified/accessed, external hypothesis remains untested.

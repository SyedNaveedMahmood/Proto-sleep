# MIST-Morph v2: morphology transport and evidence grammar

**Status:** implemented research candidate. It has not yet been executed on real EEG or CUDA at this commit. No novelty, SOTA, clinical-event, or external-transfer claim is established by implementation alone.

## Why v2 exists

MIST-Evidence v1 passed its first real-PC two-epoch epoch-only bring-up on Sleep-EDF development fold 0: evidence MF1 0.7128, summary-only 0.5745, and raw-context 0.6965 for seed 123. That smoke is encouraging but is not a scientific performance result. It is one fold, one seed, two epochs, and no temporal context.

The broad idea "waveform prototypes drive sleep staging" is already occupied by prior work such as WaveSleepNet and later prototype-based sleep-staging systems. V2 therefore does not claim novelty from having prototypes. It tests a narrower mechanism: **global competition among real waveform correspondences within each epoch, with explicit background mass and additive score accounting.**

## Core hypothesis

Independent prototype similarities can allow one local EEG structure to support several similar prototypes simultaneously. V2 asks whether an explicit assignment plan is a better morphology representation.

For every 1, 2, or 4 second scale, v2 uses the same immutable real training-waveform anchors and constrained local morphology cost as MIST-Evidence v1. Let `C[i,k]` be the cost between query window `i` and real anchor `k`. A background column is appended so ordinary/unmatched EEG is not forced onto a morphology anchor.

V2 solves a **semi-unbalanced entropic transport** problem. Query-window source masses are exact and uniform. Anchor/background target masses are only softly regularized toward a prior. The implemented generalized Sinkhorn updates use exact source scaling and a relaxed target exponent

`tau_target = rho / (rho + epsilon)`.

The default mechanism is frozen for the first comparison:

- epsilon = 0.08
- target KL strength rho = 0.20
- 30 Sinkhorn iterations
- background morphology cost = 0.40
- background target prior = 0.35

These are engineering starting values, not physiologically derived constants and not guaranteed optimal.

## Evidence variables

For each real anchor at each scale the classifier receives:

1. **transported mass**: total normalized assignment to that anchor;
2. **transport-weighted affinity**: local morphology similarity weighted by transported mass;
3. **position moment**: transported mass weighted by normalized within-epoch window position.

Each scale also contributes explicit background mass. The same 16 named summary features and one training-scaled amplitude feature from v1 remain visible as separate evidence.

With 16 anchors at each of three scales, the default v2 local representation has:

`3 scales * (16 mass + 16 affinity + 16 position + 1 background) + 17 summaries = 164 named variables`.

These variables feed the same additive temporal score model and optional CRF as v1. There is no unrestricted hidden epoch embedding in the main v2 path.

Transport mass is **not an event count**. Query windows overlap. A high mass means the normalized assignment plan places substantial evidence on an anchor, not that a clinician would count that many events.

## Visual explanation

For an anchor-related score contribution, the HTML report displays:

- the actual validation waveform interval carrying the largest transport mass for that anchor;
- the actual immutable training waveform anchor;
- total anchor transport mass;
- transport-weighted affinity and position moment;
- background mass at the corresponding scale;
- observed shape/spectrum/envelope distances;
- neural distance and its bounded fraction;
- exact signed contribution to the selected class contrast.

Clinical names remain absent until independently annotated event data and expert review support them.

## What is structurally guaranteed

Subject to floating-point tolerance and the tested code path:

- anchors remain actual immutable training snippets;
- every query window allocates exactly `1/P` source mass across anchors plus background;
- flat windows are preferentially routed to background;
- target anchor usage is relaxed rather than hard-forced;
- the local classifier uses only named transport/summary variables;
- additive emission contributions reconstruct the model score;
- reserved test files are not opened by the development runner.

None of these guarantees biological meaning, accuracy, robustness, or novelty.

## First falsification experiment

Do **not** tune transport hyperparameters before the first comparison.

Because the current Sleep-EDF NPZ files lack verified original epoch indices, start in `--epoch-only` mode. Compare on the same development fold/seed:

- `transport`: v2 assignment mechanism;
- `evidence`: v1 independent peak/mean similarities;
- `summary_only`: measured features without waveform anchors;
- `raw_context`: unconstrained learned local control (epoch-only means no temporal context).

First run two epochs only for software/runtime verification. If mechanically healthy, the next scientific comparison must use the same training budget and seeds for v1 and v2. Do not select transport settings from the two-epoch scores.

### Keep/drop logic

- If transport is materially worse than v1 across the fixed development comparison, drop the transport mechanism.
- If transport behaves nearly identically to v1 and offers no measurable assignment/explanation advantage, prefer v1.
- If transport improves accuracy or robustness while preserving source-waveform fidelity, advance it.
- If summary-only matches both morphology models, drop the headline waveform-bank claim.
- If the unconstrained control is substantially stronger, quantify the accuracy/interpretability tradeoff rather than hiding a neural bypass.

## External novelty and SOTA claims

V2 combines established ingredients (real exemplars, morphology distances, entropic transport, additive models, CRFs). Its candidate contribution is the **specific source-anchored morphology transport evidence path for sleep staging and the associated audit protocol**. A complete novelty claim requires a deeper full-text prior-art review.

Sleep-EDF-20 has been adaptively reused throughout this project. It is development evidence. SOTA or cross-dataset claims require a separately frozen external evaluation protocol and matched reference implementations.

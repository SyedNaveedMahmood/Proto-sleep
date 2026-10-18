# Prespecified experimental protocol — design proposal only

No data evaluation is authorized by this document. The future implementing agent must freeze an executable v1 protocol before inspecting additional real scores.

## 1. Cohorts, gates, data identity
- SC 74/146 verified physical scope; 397,832 full 30-second epochs; 1,454 unscored inference-only positions. Fpz-Cz, 100 Hz, calibrated physical microvolts. Subject exclusions and signal-calibration rules are inherited from `docs/DATA_CONTRACT.md`.
- Frozen 15 TRAIN / 4 VAL / 55 RESERVED. Split fingerprint `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84`. P3B uses only 79,984 physical TRAIN (79,025 scored), 21,796 physical VAL (21,789 scored). TRAIN-only fit and anchor selection; VAL only checkpoint selection/calibration if separately prespecified. Preserve original epoch sample/time offsets and scoring masks.
- SC RESERVED = within-study unseen-participant test, sealed until single separately approved evaluation; no repeated model search.
- SC→ST = conditional exploratory cross-study study, must independently pass metadata/channel/scoring compatibility and treatment/participant grouping review, then freeze transfer protocol before scores. Never promote ST exploration to independent external proof.
- SHHS or another fully independent external cohort = unavailable/unverified at design time. Access, license, montage/scoring comparability and participant independence must be verified. Zero-shot has no target fitting, even unlabeled.
- EDF-20 and EDF-78 share participants; they are not independent domains.

## 2. Estimands and baselines
Initial study is epoch-only, radius 0, no CRF and equal 30-second local inputs. Match existing P3B fixed reference seeds 123/456/789; AdamW lr 0.0003, wd 0.0001, max 60 epochs, patience 12, core and encoding batches 16, clip 5, deterministic FP32, AMP/TF32 disabled. Fixed five-class participant-average VAL Macro-F1 selects earliest best on improvement >1e-8.
Report realized early-stop epochs, parameter totals, runtime, optimizer updates and GPU memory. Exact optimization-step parity may not be possible if early stopping differs; never claim otherwise.
Reference baselines: attnsleep, evidence, summary_only and raw_context. The key new comparison is against raw_context as well as an equal-capacity q-free/event-free version of the proposed backbone; do not cherry-pick AttnSleep as the only comparator.
An offline witness auditor must not get an unfair extra tuned training budget.

## 3. Required ablations
A0: unchanged raw_context control (historical checkpoint, no recipe change).
A1: unchanged evidence (historical checkpoint, no recipe change).
A2: new backbone with background head only.
A3: same backbone + signed event score, q≡1.
A4: A3 + availability q under permitted source-only labels/synthetic interventions.
A5: A4 + optional source-training exemplar index *without* changing predictions (audit only).
A6: conditional consistency loss against matched unconditional-consistency control, **only after source-only preliminary tests indicate a useful distinction**.
A7: optional evidence substitution; synthetic tests before real-data training.
MoE and temporal context are optional later v2 candidates, not v0 requirements.

## 4. Perturbation family
- Benign control: physically calibrated gain/conversion with accurate inverse and controls for amplitude-sensitive labels; transform both event and background.
- Evidence-loss: predeclared attenuation of an **identified** simulated diagnostic component, preserving alternative event in dual-witness fixtures.
- Matched corruption: same energy/statistical distortion as evidence-loss without erasing the chosen discriminant.
- Negative control: synthetic acquisition nuisance deliberately randomized independently of stage and subject.
For real clinical data, never automatically claim all transforms preserve the annotated label or source physiology. Avoid spectral-altering transforms outside measured acquisition assumptions; annotate reasoned inclusion/exclusion. No perturbation hyperparameters based on RESERVED/ST/external test.

## 5. Metrics/decisive criteria
- Primary source *selection* metric: participant-average fixed-five Macro-F1 across 4 VAL people. This is not an unbiased generalization estimate.
- Proposed eventual **external** primary metric: participant-average fixed-five Macro-F1 on a prespecified verified independent cohort, only if access approved. If absent, mark outcome NOT TESTED.
- Supporting: per-stage F1, confusion, balanced accuracy (report observed-truth and fixed-five versions), κ, NLL, Brier, 15-bin ECE, paired subject-level differences, risk–coverage at preregistered coverage levels 100%, 95%, 90%, 80%, class-specific coverage and retention, counterfactual witness influence, evidence-source diversity.
- For abstention, report coverage and error together; no filtering of hard cases to inflate apparent accuracy. Reliability tests use risk at identical coverage and matched corruption levels.
- Select a prespecified score threshold and aggregate method on permitted source data only; use uncertainty intervals from *participants*, not epochs or seeds masquerading as participants.
- Prespecify practical vs statistical superiority margin before external evaluation (v0 deliberately leaves it unfilled; this is a freeze blocker, not permission to tune afterward).
- No stage-label-specific augmentations from target and no subject-identity leakage.

## 6. Sequential decisions
D: metadata-only document/schema tests, no EEG.
I: new implementation + synthetic contracts only, separately approved.
S: source TRAIN/VAL training and matched comparisons, separately approved; choose one frozen candidate without reserved/target exploration.
R: predeclared single SC RESERVED evaluation with source-only selected model; separately approved.
T: independent ST and true external access/scoring gates; no ST→SHHS access shortcut. Only independent cohort establishes genuine external transfer.
F: few-shot protocol with disjoint support/test and budget K∈{0,25,100,500}, repeated support sampling and random/bias/temperature/adapter comparators. No zero-shot test data used for support selection.

## 7. Stop/invalidity conditions
Halt if immutable P3B code or split fingerprint changes; forbidden role opened; normalization/bank fits VAL; unscored positions compressed; overlapping subject or repeated-night leakage; pre-fit on target for inductive zero-shot; unverified montage; test-informed architecture selection; unrealized config/provenance hash; synthetic witness donor OOD confound dominates.

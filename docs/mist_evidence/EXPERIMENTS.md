# Experiment registry

Status at branch creation: no real-EEG result exists for MIST-Evidence. Only synthetic execution and numerical invariants have been tested. Earlier MorphSpec numbers are not results for this model.

## Dataset policy

Sleep-EDF-20 is development-only. All 20 subjects participated in earlier iterative experiments, including training roles. Rotating a subject into a new validation role does not create a project-wide untouched subject. None of the old validation-selected scores should be presented as final test results.

The new manifest declares train/val/test subjects before loading arrays. Training, normalizers and exemplar selection use train only. Checkpoints are selected on val. The initial implementation intentionally exposes no test-loading command. A future independent test release must freeze code, configuration, preprocessing, checkpoint IDs, metrics and the analysis plan before test access, then be reviewed separately. Do not simply relabel old EDF-20 data as a clean test set.

For external validation, use an independently governed cohort and account for overlapping Sleep-EDF subsets. Source/target montage, units, sampling rate, scoring convention, demographic differences and pretraining exposure must be documented. A generic converted 100-Hz NPZ manifest is supported, not a claim that SHHS/MASS raw-data conversion has been implemented.

## E0. Numerical and execution gate [implemented]

Run all new tests plus the synthetic CLI smoke on CPU and then on the main PC's CUDA environment. Require finite forward/backward/updates, exact local margin accounting, CRF enumeration agreement, forbidden test access, training-only anchors, gap-safe chains and epoch-boundary resume equivalence. The test archive is deliberately invalid and must remain unopened.

A passing test does not establish clinical correctness, comparable preprocessing, GPU speed or predictive performance. The existing repository's full test suite must also be run on the main PC.

## E1. Real-data bring-up [implemented]

Create one EDF-20 development manifest, inspect label/shape/sample-rate contracts, and train two epochs in a separate runtime directory. This checks actual loader/optimizer/checkpoint/report paths. Do not report these two-epoch scores as research results. Verify epoch counts and optimizer steps. No old MorphMAE checkpoints are required.

## E2. Local information and bottleneck ablation [implemented]

Run three fixed seeds for each arm on identical train/validation subjects, using the same new optimization budget:

| Arm | Inputs to classifier | What it tests |
|---|---|---|
| descriptor | Measured morphology averages | Do handcrafted signal statistics already suffice? |
| dense | Same crop CNN, unconstrained max/mean embedding plus measurements | Cost of the evidence bottleneck |
| evidence | Live exemplar max/mean similarities plus measurements | Proposed local evidence model |
| attnsleep | Existing AttnSleep class, same new trainer | Stronger project reference |

The new AttnSleep arm is not an exact replay of the paper's or earlier runs' optimizer settings. Do not compare unmatched checkpoints as a causal ablation. Dense/evidence share encoder architecture and seeded initialization, not classifier dimensionality or total parameter count. Report counts, compute and supervised data for every arm.

Primary local development comparison: evidence minus dense Macro-F1; practical reference: evidence minus AttnSleep. Report subject-level outcomes and stage F1, including negative results. Seeds are repeated optimization runs, not additional independent subjects.

## E3. Context ablation [implemented]

Fit the same optional CRF to each local arm. Compare each arm with and without context. This prevents attributing a generic temporal benefit to morphology prototypes. Do not create chains from shuffled epochs. If original epoch-index metadata is absent, first establish continuity from preprocessing records. `--assume-contiguous` is a recorded assertion by the operator, not a verified fact.

No fixed medical transition graph, minimum stage duration or forced smoothing is used. Report boundary-near errors and per-stage regressions. The current CRF accounts for adjacent-stage dependence, not long-range cycles. A more powerful sequence module is a later candidate only if this explicit context model fails; it is not silently present in the claimed evidence-only architecture.

## E4. Ingredient ablations [configuration-supported; not run]

Freeze the comparison list before viewing scores: measured-distance rho=0 versus 0.5 versus 1; single-scale 200 samples versus 100/200/400; auxiliary morphology weight 0 versus 0.1. All are exploratory on EDF-20. Do not form an unrestricted Cartesian hyperparameter search and then report the best run as a predeclared result. Use new output folders; never reuse checkpoints across recipes.

A rho=1 run is still an exemplar-distance model. The descriptor-only arm is a separate necessary control. If extra descriptor regression does not improve matched results, keep it off. No need to revive raw waveform MAE, WCO or spherical orthogonality without a new independently testable hypothesis.

## E5. Explanation quality [partially implemented]

Implemented: raw exemplar provenance, bank embedding redundancy audit, full signed feature ledger, exact local and contextual class-margin reconstruction, raw interpolation versus same-length random-window interventions, and an expert review CSV template.

Still required on real data: systematic reports over all validation subjects, blinded expert ratings, coverage of incorrectly staged and low-confidence epochs, and artifact/quality analysis. Do not select only attractive examples. Use the same examples for competing methods. Mathematical score completeness and expert agreement are separate endpoints.

## E6. Clinical morphology validation [scorer implemented; data/training adapter pending]

Use independent expert event annotations, for example a suitably licensed MASS/DREAMS event subset. The source annotations, montage and scorer disagreement must be reviewed. Stage labels and pseudo-labels from a rule detector are not independent clinical ground truth.

Establish prototype/event naming on development annotations. Freeze any mapping, thresholds, merging/NMS policy and evaluation tolerances before event-test use. Report both event-level precision/recall/F1 and localization behavior, stratified by subject and event type. `score-events` implements one-to-one interval matching within recording and event category, maximizing valid match count before IoU. It does not itself convert prototype scores into validated events or train an event detector.

Only after this gate may the interface use verified clinical labels. Until then it says 'unreviewed waveform exemplar' and shows measurements, not 'detected spindle'.

## E7. Independent staging and transfer [planned, not implemented as an unlocked test]

Reproduce a predeclared set of credible baselines under identical channels and target test subjects: the project AttnSleep reference, the closest interpretable WaveSleepNet/ProtoSleepNet variant where source and input setting permit, and a stronger contemporary sequence baseline. Foundation-model comparisons must disclose larger pretraining corpora, modalities and subject exposure.

Primary endpoint: subject-averaged Macro-F1 against the strongest matched baseline. Report pooled epoch metrics separately, per-stage F1, accuracy, kappa, calibration, latency, memory and trainable parameter counts. Average seeds within subjects before paired inference; do not bootstrap EEG epochs as independent subjects. Training overlap and model-selection uncertainty remain limitations of ordinary rotating-fold intervals.

A proposed confirmatory success criterion, to be frozen before genuinely independent testing, is at least +0.005 Macro-F1 against the strongest matched baseline with a positive subject-level paired 95% interval, plus acceptable event-evidence validation. This is a proposed practical threshold, not a universal definition of SOTA. A SOTA label additionally requires the relevant contemporaneous benchmark comparisons, not merely beating AttnSleep.

## Stop/change policy

Every change gets a new config and run identity. Preserve failed runs. Keep the model if evidence is useful; drop a component if its controlled ablation shows no benefit. A new paper claim must not be created by changing endpoints after a failed gate. Current code is suitable for development and falsification, not a license to claim novelty, SOTA, clinical deployment or cross-dataset success.

# Architecture specification: MIST-Evidence v1

## 1. The scientific decision

MIST-Sleep remains the broader goal of morphology-informed sleep transfer. The present candidate is **MIST-Evidence**, a classifier whose input evidence can be traced to actual EEG windows and training examples. It is not described as an established state-of-the-art architecture.

The previous project found useful improvements from MorphSpec in frozen-feature comparisons. It did not establish improvement over the fully trainable AttnSleep reference. In the final folds 15-19, MorphSpec minus frozen random was +0.061066 mean validation Macro-F1, versus +0.038392 against frozen original MorphMAE, but -0.020025 against trainable A1. These are prior user-supplied results, not measurements of this branch. Staged transfer also failed its primary endpoint. The reconstruction-focused MorphMAE objective already included spectral and derivative terms, so calling that baseline "waveform-only" would be inaccurate.

Three earlier interpretations need correction:

- A frozen probe improvement is evidence about a specified representation/classifier protocol, not proof of clinical morphology or domain invariance.
- Frozen and fine-tuned comparisons on different rotating folds do not establish that fine-tuning erased morphology. That requires matched same-subject before/after evaluation or controlled interventions.
- A new rotating fold is not a wholly new cohort. Subjects used for validation in one fold occurred in training in other folds. All Sleep-EDF-20 results are now development evidence.

### Remove from the default core

Mandatory reconstruction pretraining, prototype-distribution MAE, transition-aware masking, WCO, spherical separation penalties, and unvalidated event names are removed. They add costs or assumptions without sufficiently strong support from the recorded experiments. This does not prove these ideas never work. It means they do not receive a privileged place in the new core.

### Retain

The source/validation separation, fixed comparison seeds, morphology/spectral measurements, explicit source provenance, and the principle of comparing each component with a simpler control are retained.

## 2. Data and scope

The implemented input contract is one 30-second EEG epoch with 3,000 samples at 100 Hz. NPZs contain `x`, `y`, optional `fs`, and preferably `epoch_indices`. Labels must already map to Wake/N1/N2/N3/REM = 0/1/2/3/4. Resampling, montage selection, unit conversion, annotation mapping, and removal of artefacts are NOT silently performed.

Input units remain "input file units" unless calibrated upstream. Neither the code nor its figures invent microvolt units. Absolute clinical amplitude thresholds are not inferred from standardized or unknown-unit recordings.

The existing EDF-20 path and subject convention are supported. An explicit recording manifest supports other preprocessed datasets with the same tensor contract. Subject IDs in multi-dataset manifests must be globally namespaced. Family-level splits, where applicable, must be built by the data curator.

If original epoch indices are absent, the temporal model refuses to assume that surviving epochs are adjacent. `--epoch-only` is a safe alternative. `--assume-contiguous` records a user assertion and is permitted only after continuity is independently verified. Gaps and night boundaries break both temporal filtering and CRF chains.

## 3. Architecture at a glance

```text
Single-channel EEG, 30 seconds
  |
  +-- Independent windows: 1 s, 2 s, 4 s; stride 0.5 s
  |       |
  |       +-- Direct observed comparisons
  |       |     normalized signed shape / relative spectrum / RMS envelope
  |       |
  |       +-- Small local neural comparison
  |             constrained contribution, no full-epoch context in this encoder
  |       |
  |       +-- Compare against actual immutable TRAINING waveforms
  |             16 source exemplars per scale, 48 total
  |       |
  |       +-- Per-exemplar peak and mean similarity: 96 features
  |
  +-- 16 explicit spectral/line-length features
  +-- 1 amplitude feature, normalized with TRAINING statistics
          |
          113 named evidence variables per epoch
          |
     Nonlinear univariate basis for each variable
          |
     Additive lag-specific score contributions, +/-10 epochs
          |
     Five emission scores per epoch
          |
     Linear-chain CRF over each continuous night segment
          |
     Hypnogram + waveform matches + exact score accounting
```

The architecture does not send a free full-epoch embedding around the evidence bottleneck. The measured summary features are a separate, named evidence path. They are not falsely attributed to waveform prototypes. The `raw_context` control intentionally removes this restriction and is explicitly labeled non-interpretable.

## 4. Training-waveform anchors

For each length, candidate windows are sampled from training recordings, balanced across subjects and available sleep-stage labels. Stage labels are used to improve candidate coverage; therefore bank construction is not claimed to be label-free. No clinical event labels are supplied.

A fixed observed descriptor combines downsampled signed shape, square-root band distribution, and RMS-envelope shape. MiniBatchKMeans groups these candidate descriptors. The actual candidate closest to each center is retained, without reusing a candidate. Thus each anchor is a real input snippet, not a cluster centroid plotted as EEG. The alternative `--anchor-method random` controls for selection strategy.

Each anchor records the original file hash, subject, array epoch, original epoch index when known, start sample, duration, and stage of the surrounding source epoch. The clinical event label is null. Source stage is not evidence that a short snippet is a spindle or K-complex.

The anchor buffers remain unchanged throughout learning. Their learned embeddings are recomputed by the same current local encoder as the query. There is no post-training prototype projection step that changes the classifier after selection. This avoids one source of train/display mismatch but does not eliminate redundancy, artefact selection, or misleading learned similarities.

## 5. Local morphology comparison

Let `q` be a query window and `p_k` a source anchor of the same length. Both are centered and normalized by their local RMS, with an epsilon floor. Flat windows receive zero similarity.

Three observed dissimilarities are used:

- `d_shape = (1 - cosine(z(q), z(p_k))) / 2`, retaining polarity;
- `d_spectrum = 0.5 * sum_b (sqrt(P_b(q)) - sqrt(P_b(p_k)))^2`;
- `d_envelope = (1 - cosine(E(q), E(p_k))) / 2`.

`P` is Hann-windowed relative spectral energy in 0.5-4, 4-8, 8-12, 12-16, and 16-30 Hz. `E` contains eight local RMS-envelope bins. These are engineering descriptors, not clinical event detectors. Short windows have limited frequency resolution, particularly in the delta band. The three scales partly address this limitation; they do not solve it by assumption.

A two-layer Conv1d/GroupNorm/GELU encoder supplies unit-length local embeddings, with another bounded cosine dissimilarity `d_neural`. Each batch item is one local window. GroupNorm therefore cannot draw statistics from another window, epoch, or subject. No input outside that window can affect its neural embedding.

For each anchor, observed weights `a_k = 0.1 + 0.7 * softmax(theta_k)` and neural fraction `m_k = 0.5 * sigmoid(eta_k)` give

```text
d_k = (1 - m_k) * sum_j a_kj d_observed_j + m_k d_neural
s_k = exp(-tau_k d_k),    tau_k = 2 + 18 * sigmoid(rho_k).
```

This construction is bounded, nonnegative, and differentiable apart from ordinary max-pooling ties. It is a dissimilarity model, not a claim of a metric satisfying the triangle inequality. At most half of the combined dissimilarity can be supplied by the learned embedding. Each observed component has at least weight 0.1 inside the observed mixture, whose overall weight is at least 0.5. Thus the total dissimilarity is at least 0.05 times any one observed component. That constraint makes the observable components impossible to eliminate completely. It does NOT prove clinical meaning. The cap and temperature range are initial design choices whose value must be tested against `observed_only`, not universal scientific constants.

All signal descriptors, scores, losses, and CRF computations use FP32. AMP is not enabled in v1.

## 6. Evidence variables and amplitude

For each anchor, retain the maximum match and the mean match across all overlapping windows in the epoch. The maximum has an exact matching interval. The mean is an aggregate over all windows; it must not be illustrated as though one displayed peak explains the entire mean. Neither is called an event count.

The measured summary path exposes full-epoch relative band energies, their standard deviations and maxima across six 5-second segments, and a bounded normalized line-length measure. It does not reuse the previous 16-target MorphSpec model or claim numerical equivalence to its log-power targets.

Local RMS normalization can discard clinically relevant amplitude. A separately named amplitude variable therefore uses log whole-epoch RMS, centered and scaled by mean and standard deviation computed ONLY on training recordings, followed by a sigmoid. This gives the classifier access to amplitude without concealing its contribution in a prototype plot. It is not a calibrated microvolt measurement. It can encode acquisition shortcuts, so `no_amplitude` and gain-sensitivity tests are required.

The matching pathway is approximately invariant to positive gain and DC shifts under nondegenerate signal conditions. The FULL model, with its explicit amplitude variable, is not gain invariant. Polarity inversion is not assumed safe. No augmentation has been silently declared physiology preserving.

## 7. Exact additive temporal evidence

For evidence variable `f[t,i]`, use three basis functions:

```text
phi(f) = [f, f^2, ReLU(f - 0.5)].
```

Each basis is univariate. The emission for class `c` is

```text
e[t,c] = b[c] + sum_(lag=-R..R) sum_i sum_j W[c,i,j,lag] phi_j(f[t+lag,i]).
```

`R=10` means the emission sees at most 21 epochs, or 10.5 minutes. This is an offline, bidirectional model. It is not real-time/causal. Boundary padding represents zero additional evidence.

Every source epoch, feature, basis, and lag contribution is directly available. The implementation checks that all contributions plus bias reproduce each emission score. Because `phi(0)=0`, zeroing one evidence coordinate has an exact, testable emission-score effect. This is a feature intervention in the model, not proof that physically deleting an EEG event would produce the same effect.

The univariate/additive restriction may lose useful interactions. That is an intentional interpretability constraint and a potential accuracy bottleneck. The nonlinear `raw_context` control tests the cost of removing it. Do not hide an unrestricted neural residual in the main candidate and continue claiming the whole decision is explained by waveform matches.

## 8. Sequence inference

A linear-chain CRF scores a candidate sleep-stage path:

```text
S(y|x) = start[y_1] + sum_t e[t,y_t]
         + sum_(t>1) A[y_(t-1), y_t] + end[y_T].
p(y|x) = exp(S(y|x)) / Z(x).
```

The implementation uses log-sum-exp forward recursion, exact Viterbi decoding, and forward/backward marginals. The tests compare these against exhaustive enumeration on short chains. Transitions are learned from training labels; no test-label transitions or hand-coded physiological rules are used.

The emission neighborhood is finite, while CRF decoding operates across each complete continuous segment. This does not make the model a full-night Transformer or an explicit duration model. Long constant-stage durations are not modeled separately.

Training minimizes average per-token negative log-likelihood of nonoverlapping core blocks with input halos. Validation decodes whole continuous segments. The training objective is a sum of proper BLOCK conditional likelihoods, not exactly the full-record likelihood. This distinction and its boundary approximation must remain explicit. Core length and CRF removal are included in the experimental plan.

A displayed class contrast separates center-epoch evidence, neighboring-epoch evidence, bias, and CRF neighbor-transition terms. Its conditional path-score difference fixes the neighboring decoded labels. It is NOT a decomposition of a marginal probability or a physiological causal explanation.

## 9. Optimization and reproducibility

The initial recipe is AdamW, LR 3e-4, weight decay 1e-4, gradient clipping at norm 5, 60 epochs maximum, patience 12, and seed 123 for the first development run. There is no learning-rate rescue grid. The primary checkpoint metric is mean subject-level validation Macro-F1; pooled Macro-F1 is also saved.

The likelihood is unweighted. Training stage labels balance anchor candidates, but there is no hidden class-weight change between the implemented controls. An independently justified weighted-loss study would be a new registered ablation, not an undocumented change to the final benchmark.

Logs are printed with flushing. `last.pt` includes optimizer and Python/NumPy/PyTorch RNG states. `best.pt` is not a completion marker. `COMPLETE.json` is written only after the run exits training, reloads the best checkpoint, and saves validation predictions. Resume checks data, implementation, configuration, and bank fingerprints. Mid-epoch interruptions repeat that epoch; resumed epoch-boundary execution is tested against uninterrupted execution.

## 10. Visual outputs and honest semantics

`index.html` contains observed EEG, actual query/source waveform overlays, matching intervals, and signed evidence differences. JSON includes all peak/mean match information needed to audit the displayed snippets, dissimilarity components, and source provenance. The full CSV contains all terms, not just the top examples shown in HTML.

The waveform labels initially read `s100_p00`, etc. They do not say spindle, slow wave, or K-complex. Those names require independent expert annotation and event-level validation. A pretty prototype waveform is not sufficient. Single-channel EEG also cannot establish every criterion available to a full EEG/EOG/EMG sleep scorer.

## 11. What can and cannot be promised

Guaranteed by design and checked in tests: source-anchor identity, local input support, bounded scores, additive emission accounting, proper finite-chain normalization, and explicit split guards, within the tested software paths.

Not guaranteed: improved staging, biological specificity, domain invariance, absence of all software bugs, SOTA performance, or complete novelty. These are empirical or literature-review questions. The complete experimental program is in the companion plan.

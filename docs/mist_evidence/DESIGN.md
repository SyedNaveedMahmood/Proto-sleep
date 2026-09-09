# Design specification

## 1. Evidence that constrains this redesign

The completed project logs show useful MorphSpec frozen features, not superior final staging. In the last five rotating fold roles, MorphSpec exceeded frozen random features by 0.061066 mean validation Macro-F1 and original frozen MorphMAE by 0.038392, but remained 0.020025 below the trainable AttnSleep reference in every fold. The earlier staged-transfer run was also negative against AttnSleep. These are supplied project results, not results from this branch.

Two previous interpretations must be narrowed. Comparing frozen and unfrozen models on different subject/fold groups does not prove catastrophic forgetting. Also, rotating new fold roles are not an entirely unseen cohort: subjects in later validation roles appeared in earlier training roles. All Sleep-EDF-20 subjects have now participated in adaptive development. Old validation-selected scores are not an unbiased final test estimate.

Remove mandatory raw-signal MAE initialization, the free spherical prototype reconstruction bottleneck, transition-biased masking and WCO from the new core. They have not earned a place through the available results. Keep AttnSleep as a benchmark and measured spectral/morphological information as a testable ingredient. The original MorphMAE used spectral and derivative losses too; do not describe it as a pure MSE-only baseline.

## 2. Target and operational definition

MIST-Evidence predicts a stage from scores that explicitly compare short query waveforms with real training waveforms, plus named measured descriptors. It distinguishes local EEG evidence from sequence context. The target is interpretable **model evidence**, not an automatic claim of human-equivalent clinical reasoning.

The candidate contribution is a combination of verifiable invariants: strict crop-local evidence, no discrepancy between a free latent prototype and its displayed exemplar, and complete class-contrastive accounting through local and contextual stages. None of prototypes, morphology targets, exemplar classification or CRFs is individually new. See PRIOR_ART.md.

## 3. Local morphology encoder

Input is a preprocessed single-channel 30-second epoch at 100 Hz: `[B,1,3000]`. No resampling or label remapping is performed implicitly. Scales are 100, 200 and 400 samples. Adjacent crops overlap by half a window; a final crop covers any remainder.

Each crop is centered and RMS-normalized **within that crop** for the learned branch. A shared CNN uses convolution widths 16, 32, 48, strides 2, kernels 11, 9, 7; GroupNorm and SiLU follow each convolution. Four-bin adaptive pooling feeds a 64-dimensional unit-normalized embedding. No BatchNorm state, whole-night normalization or temporal attention can import outside-interval data into this crop embedding.

Default local parameter count is 28,901; this is a deliberately compact first candidate, not evidence that it can match much larger models. A matched dense control determines whether the evidence bottleneck is the limiting factor before any capacity expansion.

## 4. Real, live waveform anchors

For each scale, build a pool using at most 256 labeled epochs per training subject, taking one random crop per selected epoch. Select four exemplars per stage using farthest-point sampling in standardized measured descriptors. Candidate selection balances source subjects where possible. These are **epoch-stage strata**, not micro-event labels.

The bank contains 20 anchors per scale, 60 in total. Each has its original float32 samples, recording ID, subject, epoch index, sample start/end and source-file SHA-256. The samples are buffers, not free trainable waveforms. At each forward pass:

`p_k(theta) = encoder_theta(actual_training_crop_k)`

The query and all exemplars use the same current encoder. There is no post-training nearest-waveform substitution or periodically stale projected prototype. The raw identity remains exact even though its learned representation can change. Redundant embeddings can still occur and are audited; raw anchoring is not an anti-collapse theorem. Diversity sampling can choose artifacts, so expert review remains necessary.

## 5. Measured morphology and declared similarity

For every crop compute eight quantities: log relative power in [0.5,4), [4,8), [8,12), [12,16), [16,30) Hz; log RMS amplitude; RMS-normalized line length; zero-crossing fraction. Power is measured using a Hann-window periodogram. Standardization means and scales are fit from the training candidate pool only.

These are descriptive signal measurements, not automatic event detectors. One-second windows have coarse spectral resolution. RMS is in the supplied NPZ units; it is not a calibrated microvolt measurement. Amplitude robustness must be evaluated rather than assumed.

For unit embeddings z and p, and standardized descriptors d and a, use:

`D_latent = ||z-p||^2 / 4`

`D_measured = 1 - exp(-mean((d-a)^2)/2)`

`D = (1-rho) D_latent + rho D_measured`

`s = exp(-D/tau)`

Default rho=0.5 and tau=0.2 are declared engineering choices, not theoretically optimal values. Both component distances are bounded for finite inputs. The metric's components are included in reports. Similarity in this metric does not imply visual or clinical equivalence.

## 6. No hidden classification bypass

For each anchor, retain its maximum similarity across query windows and mean similarity across query windows. Maximum means strongest match; mean means average matching score, not an event count or a clinical duration. Include the mean of each standardized measured descriptor at every scale. The resulting vector has 144 features.

For stage c:

`local_logit_c = bias_c + sum_j W[c,j] * feature_j`

Weights are signed. Explanations compare the predicted class c with an explicit alternative r:

`local_margin(c,r) = bias_c - bias_r + sum_j (W[c,j]-W[r,j]) * feature_j`

All terms, including unfavorable and measured-descriptor terms, are exported. Displayed raw snippets are not a separate black-box explanation branch. There is no unconstrained latent residual path in the evidence model. The dense CNN exists only as a visibly named non-interpretable control.

## 7. Context without an opaque residual

An optional linear-chain conditional random field models adjacent-stage dependence over each intact recording segment. Its input is the fixed local model's five logits, never true neighboring labels. Unknown labels or missing epoch indices break chains during training. Different nights never share a chain.

Sequence energy is the sum of local logits, learned transition potentials, and start/end potentials. Each potential is bounded using `2*tanh(parameter)` and initialized at zero. Zero potentials recover the independent local classifier exactly. This is a first-order model, not explicit duration modeling or a complete model of full-night physiology.

Train only the CRF parameters on training sequences after selecting the local model. Use exact forward/backward recursions in float64, and marginal-MAP predictions. Inference uses future as well as past observations, so it is offline, not a real-time claim.

For epoch t, the exact conditional class odds decompose as:

`log p(y_t=c|x) - log p(y_t=r|x)`

`= local_margin(c,r) + incoming_context_margin(c,r) + outgoing_context_margin(c,r)`

The report shows all three. Context messages summarize the chain; they do not assign a waveform interpretation to every distant epoch. This identity is numerically checked against exhaustive enumeration on short chains.

## 8. Learning and checkpointing

The initial local objective is training-class-weighted cross-entropy. AdamW, LR 3e-4, weight decay 1e-4, maximum 60 epochs, patience 12, batch 16, gradient clipping 5, FP32. Validation Macro-F1 selects the checkpoint. No mandatory SSL stage is used.

An optional auxiliary smooth-L1 descriptor-regression head is implemented but defaults to weight zero. It must pass an ablation before adoption. In particular, a locally normalized encoder may not recover absolute amplitude, so auxiliary objectives require scrutiny rather than assuming all descriptor targets help.

CRF training minimizes sequence negative log likelihood normalized by the number of training epochs. It uses fixed cached local logits and separate validation selection. Training is therefore two-stage, not joint end-to-end training through the CRF.

`last.pt` is committed atomically at epoch boundaries with optimizer/shuffle/RNG states. `best.pt` is not proof of run completion. A separate COMPLETE.json records the selected model digest. Resume refuses changed data, code or configuration. Resume after a power loss may replay the unfinished epoch, not resume its exact minibatch.

## 9. What the visual interface establishes

The HTML report shows the query crop, actual training exemplar, normalized shape overlay, all window similarities, recording provenance, measured descriptors in JSON, signed class contributions, and the separate context contribution. Mean-pooled evidence explicitly displays the whole similarity curve; a single closest crop does not explain its entire mean.

An optional input intervention interpolates the strongest absolute-evidence window and compares the class-margin change with five equal-length random windows. This is a perturbation audit and may create off-distribution inputs. It is not a biological counterfactual or proof of event causality.

All anchors start as `unreviewed`. Expert annotations and independent event-level tests are required before naming an anchor spindle-like or K-complex-like. Stage labels alone cannot certify those identities. EOG/EMG information is not available in the single-channel input.

## 10. Failure modes and removal rules

- Evidence worse than the matched dense encoder: bottleneck or anchor coverage is a candidate cause, not proof of a broken implementation. Inspect, then change one component on development data.
- Descriptor-only matches evidence: drop the learned similarity branch unless it provides independently measured value.
- Context harms stage boundaries or minority-stage F1: omit CRF; do not hide the regression in overall accuracy.
- High within-bank redundancy: inspect diversity/usage rather than blindly enforcing orthogonality.
- Clinical reviewers reject exemplars: drop clinical event names, regardless of staging accuracy.
- External performance fails: drop the robustness/SOTA claim. Do not tune on the external test labels.

Only a fixed independent benchmark can establish performance, and only a much fuller novelty audit can establish the defensible original contribution. This specification is implementable and falsifiable, not guaranteed to win.

# Code interfaces, evidence ledger and provenance

## Verified existing interfaces on research branch
- `src/mist_transfer/protocol.py`: `read_frozen_split` validates frozen JSON/receipt; `load_development(frozen, role)` **rejects roles other than train/val before descriptor access**. Do not add a permissive development test loader.
- `src/mist_transfer/source_data.py`: `SourceRecording` retains `indices`, `y`, `scoring_mask`, `recording_id`, `subject`, `split`. `LazyEpochArray` lazily supplies [N,1,3000] normalized FP32; source normalization contains physical microvolt mean/std. Synthetic tests must not instantiate real source loaders.
- `src/mist_transfer/experiment.py`: `evaluate_source` predicts *all physical positions* before applying stage-scoring mask and calls `classification_metrics`.
- `src/mist_transfer/metrics.py`: fixed-five classification metrics; undefined F1=0, unobserved class recall=null, balanced accuracy on observed truth classes.
- `src/mist_transfer/baselines.py`: adapter contract `encode(x)` → `emissions(features)`, `predict(logits)`, `loss(emissions, labels, valid)` and `cfg.radius==0`. Do not modify frozen `make_baseline` to add MIST-COVE.
- `src/mist_evidence/model.py`: real source TRAIN-only waveform exemplar bank and multi-scale local windows; original event provenance metadata currently produced by `training_bank`.
- `src/mist_transfer/full_experiment.py`: P3B gated runner, exact identity/fingerprint/resume policy; must remain unmodified.

## Proposed v1 model I/O (not implemented yet)
`forward(x_normalized: Float[B,1,3000], *, return_evidence: bool=False)` → finite logits `Float[B,5]`. Optional evidence structure:
- `event_intervals`: Int[B,K,2] original start/end samples in [0,3000], half-open, with validity mask for fewer than K witnesses.
- `background_logits`: Float[B,5].
- `signed_evidence`: Float[B,K,5].
- `availability`: Float[B,K,5] in [0,1] (or explain/justify a genuinely pair-specific [B,K,5,5] alternative).
- `contributions = availability * signed_evidence` with zeros for invalid windows.
- `logits == background_logits + contributions.sum(dim=1)` within declared floating-point tolerance.
- `selection_policy`, `event_scores`, `acquisition_observability_metadata` distinct from class probabilities.
- `raw_provenance` for permitted real development data is metadata bound to source file hash, recording/night, original epoch index, start sample and channel; never serialize waveform bytes to public reports.

If normalized input is `x_z=(x_uV-mean_uV)/std_uV`, then restore calibrated `x_uV=x_z*std_uV+mean_uV` from TRAIN-only frozen statistics. Reversing a calibration is not the same as imposing gain invariance.

## Witness audit proposal
Audit trace should include, in **local ignored** storage only: source TRAIN identity, original epoch/sample interval, donor-selection recipe/version, optional nearest real TRAIN waveform exemplar identity, signed stage logit contribution, counterfactual Δ, sensitivity across plausible donor strategies, corruption diagnostics.
Code-level exact additive equality proves decomposition, not causal physiology. Counterfactual donor replacements may produce edge discontinuities/OOD artifacts; check signal statistics and donor strategies. Never publish raw trace data or subject-level paths.

## Data and outputs
- Commit: model source/tests/design configs, source-only **aggregate** metrics after consented phases, hash inventories, validation methodology.
- Keep local/ignored: `mist_transfer_runs/mist_cove/` raw waveform examples, per-subject predictions, anchors, fitted scalers, checkpoint states, log streams, diagnostic plots from participant EEG.
- Cache keys: implementation Git SHA, config hash, split fingerprint, input source hashes, environment/dependency and GPU identity, random seed, dataset/montage/preprocessing version. On resume mismatch, hard stop; never silently refit or reuse stale cache.
- Respect existing `SourceProofs` path/hash contracts and atomic-seal/checkpointing patterns; don't copy prior frozen artifacts into a new identity without verifying allowed provenance and version compatibility.
- All class-balanced *training* sampling must preserve source roles and ensure original time indices remain recoverable. Validation/test full physical grid is independent of labels.

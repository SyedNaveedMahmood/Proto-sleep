# MIST-Evidence v1: research branch

**Status: implemented research candidate; CPU execution tested; real-EEG performance and CUDA execution not yet established for this new model. No SOTA or clinical-event claim.**

MIST-Sleep's goal remains morphology-based sleep staging with evidence a person can inspect. This branch replaces the unsuccessful mandatory pretraining/prototype stack with a narrower hypothesis: classify from **live embeddings of real waveform exemplars plus measured morphology**, then add a separately auditable adjacent-stage context model only if it helps.

This is not a renamed MorphSpec checkpoint or a copy of recovered WaveSleepNet. Existing code and main-branch experiments remain intact.

## Read in this order

1. [Architecture, equations and claim boundaries](DESIGN.md)
2. [Experiment registry and decision rules](EXPERIMENTS.md)
3. [Runbook and data contract](RUNBOOK.md)
4. [Closest prior work and novelty limits](PRIOR_ART.md)
5. [Verification record](VERIFICATION.md)

## What is implemented

- Independent 1, 2 and 4-second crop encoding, using GroupNorm.
- Train-only, subject-balanced candidate pools and stage-stratified raw exemplar selection.
- Real waveform anchors re-encoded on every forward pass with the same trainable encoder.
- A declared learned-distance/measured-morphology similarity, maximum and mean pooling.
- Additive local staging scores with exact class-contrastive feature accounting.
- Optional bounded linear-chain CRF, full-night contiguous-chain inference and exact local/left-context/right-context log-odds accounting.
- Descriptor-only and unconstrained local CNN controls; existing AttnSleep reference arm under the same new training loop.
- Atomic checkpoints, completed-run markers, code/data/config fingerprints and epoch-boundary resume.
- Self-contained HTML waveform reports, JSON/CSV ledgers, interpolation interventions, a bank audit and an event-level interval scorer.
- Synthetic full-pipeline smoke and regression tests, including exhaustive CRF checks and interrupted/resumed training equivalence.

## Not implemented or not established

An EEG event detector with independently validated spindle/K-complex names, raw EDF/SHHS importers, event-supervised encoder training, an external test unlock, multi-channel fusion, and replication of all current SOTA baselines are **not** claimed to exist here. Existing 100-Hz single-channel NPZ files are the supported input. General external-cohort manifests can use this same NPZ contract for train/validation, but no external results have been obtained.

A saved waveform is a real observation, not proof that its learned similarity is clinically meaningful. The encoder is still a learned function. The ledger is model-faithful, not a biological causal explanation.

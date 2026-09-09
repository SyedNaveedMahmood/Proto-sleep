# MIST-Evidence v1

**Branch:** `research/mist-evidence-v1-verified`  
**Status:** implemented research candidate with CPU-tested software; not a validated SOTA or clinical system.  
**Parent:** `25f76fe80595f65b019e897de62e6efff3adb879`. Existing experiments on `main` are unchanged. A different commit appeared on the initially created `research/mist-evidence-v1` branch during publication; it was preserved, not force-pushed. This verified branch contains the implementation tested in the verification record.

The new question is: **Can a sleep classifier make its predictions through verifiable local waveform evidence, while exposing rather than concealing the contribution of neighboring epochs?**

This is a new supervised morphology-evidence architecture, not another attempt to rescue MorphMAE by changing its learning rate. No legacy checkpoint or legacy WaveSleepNet installation is needed.

## Read in this order

1. [Architecture and mathematical specification](docs/MIST_EVIDENCE_DESIGN.md)
2. [Experiments, decision rules, and claim limits](docs/MIST_EVIDENCE_EXPERIMENTS.md)
3. [Commands, output files, and resume behavior](docs/MIST_EVIDENCE_RUNBOOK.md)
4. [Prior art and novelty assessment](docs/MIST_EVIDENCE_SOURCES.md)
5. [Implementation verification](docs/MIST_EVIDENCE_VERIFICATION.md)

## What is implemented

- Source-only, immutable waveform exemplars at 1, 2, and 4 seconds.
- A local learned comparison constrained to retain direct waveform, spectral, and envelope dissimilarities.
- Peak and mean match evidence; explicit spectral/line-length and training-scaled amplitude features.
- Exact additive contributions across a 21-epoch input neighborhood.
- A tested linear-chain CRF for sequence likelihood, Viterbi decoding, and posterior marginals.
- Seven ablations/controls in addition to the main candidate.
- Selective train/validation loading from the existing NPZ layout or explicit manifests.
- Original-epoch gap handling, optional safe epoch-only operation, and blocked test access.
- Complete train/validation loops, fingerprints, atomic checkpoints, epoch-boundary resume, and completion verification.
- HTML waveform comparisons, JSON evidence records, and CSV score contributions.

## What is not established or implemented

No new real-EEG staging result has been measured for this branch. Named clinical event detection, external-dataset scoring, third-party SOTA reproductions, and clinician validation are planned studies, not completed features or demonstrated claims. The code deliberately does not expose a development-time test-evaluation switch.

Prototype learning, shapelets, additive models, and CRFs are established ideas. The research contribution is the proposed constrained evidence path and its falsifiable audit protocol. A complete novelty claim would be unjustified.

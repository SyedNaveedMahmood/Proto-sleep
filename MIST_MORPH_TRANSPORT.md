# MIST-Morph v2

**Branch:** `research/mist-morph-transport-v2`  
**Status:** implemented morphology-transport research candidate; not yet validated on real EEG/CUDA at branch creation.  
**Parent:** `research/mist-evidence-v1-verified`.

MIST-Morph v2 keeps the real source-waveform evidence infrastructure from MIST-Evidence v1 and changes the central correspondence mechanism. Instead of treating each prototype similarity independently, overlapping EEG windows compete through a semi-unbalanced entropic transport plan over real training-waveform anchors plus an explicit background sink.

The goal is not to claim that optimal transport itself is new. The research question is whether **source-anchored morphology transport produces a better and more faithful staging representation than independent prototype similarities**.

Read:

1. `docs/MIST_MORPH_TRANSPORT_V2.md` for architecture and equations.
2. `docs/MIST_MORPH_TRANSPORT_EXPERIMENTS.md` for fixed experiments and drop rules.
3. `docs/MIST_EVIDENCE_DESIGN.md` for the inherited v1 waveform-distance and additive-context contracts.

Implemented additions:

- `src/mist_evidence/transport.py`: semi-unbalanced Sinkhorn assignment and transport evidence model.
- `src/mist_evidence/transport_explain.py`: source-waveform transport reports.
- `tests/evidence/test_transport.py`: source-marginal, background, gradient, immutability and score-accounting tests.
- `scripts/run_mist_morph_transport_dev.sh`: controlled development runner.
- CLI variant `transport`, evaluated against the unchanged v1 `evidence`, `summary_only`, and `raw_context` controls.

Transport hyperparameters are fixed for the first comparison. Do not tune them from the two-epoch integration smoke. Sleep-EDF-20 remains development data. Clinical event names, external transfer, novelty and SOTA require separate evidence.

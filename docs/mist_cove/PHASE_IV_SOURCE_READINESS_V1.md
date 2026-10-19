# Phase IV source-only proposal — not authorized or executed

The new `phase4.source.Adapter` implements the existing encode→emissions, masked loss and probability interface, with radius zero and no CRF. Generated recording fixtures preserve original indices, unscored gaps and full-grid inference through the unchanged sealed trainer/evaluator. The prepared JSON names M0/M1/M2/M3/M2_rms and unchanged raw_context/evidence at seeds 123/456/789: a future 21-run comparison, not an executed study. M1_phase remains a synthetic internal phase-information control, not a canonical-waveform source candidate.

All source candidates preserve AdamW 3e-4 / weight decay 1e-4 / max 60 epochs / patience 12 / core and encoding batch 16 / gradient clip 5 / deterministic FP32 / AMP and TF32 off. Earliest best participant-average fixed-five source-VAL F1 must improve by >1e-8. They share the 15 TRAIN / 4 VAL / 55 RESERVED split and canonical raw-EDF pipeline, complete physical grids, source provenance and TRAIN normalization. Source-VAL is repeatedly used development, not an independent test.

The metadata-only command is:

```bash
python scripts/run_mist_cove_phase4_source_prepared.py --mode inspect
```

It does not call a source loader. Real execution additionally requires a separate explicit user authorization flag, an actual Phase IV Gate A/B human review, matching config/implementation identities, fresh real preflight source proofs/grid, environment-bound CPU/CUDA/exact-resume artifact hashes, private outputs and the shared GPU writer lock. No such real proposal/approval receipt is created in this phase. Prior Phase III approvals do not satisfy this gate. Gate A pauses all seven seed-123 candidates at epoch one, estimates all 21×60 epochs without early-stop discount, and blocks if >24 GPU hours. A separate Gate B review is required; its seed-123 resumes retain the exact saved state. The future authorized preflight/operational monitoring must be reviewed before launch. Synthetic timing/memory does not establish full-data feasibility.

Before future source work: independently authorize preflight, verify the existing frozen split via `read_frozen_split`, open only `load_development(train/val)`, verify all exact source/annotation hashes and physical counts 79,984/79,025 TRAIN and 21,796/21,789 VAL, calibration, no crossing nights/gaps, and trusted TRAIN normalization/banks. Missing/corrupt artifacts stop execution; no refit or silent source substitution. Bind new environment/correctness receipts and benchmark real full-epoch throughput before approving the frozen comparison. Save checkpoints, source traces, fitted statistics and participant predictions only under ignored outputs. Never infer RESERVED/ST/SHHS access.

## Separately specified future donor compatibility

The old fixed 3-µV donor guard and Phase III rejection results remain unchanged. The new pure-array prototype `phase4.donors` is a separate numerical-envelope proposal. Its fitting API currently **accepts only explicitly synthetic TRAIN fixtures** and rejects VAL/real scopes before array conversion. No real donor statistic is fitted here.

A future authorized, separately versioned TRAIN-only protocol would partition TRAIN recordings into disjoint fit/calibration portions before waveform inspection. Label-independent candidate windows supply log centered RMS, log RMS first-difference and high-frequency energy fraction. Fit medians/MADs on fit windows, and a fixed 95th-percentile maximum standardized-distance cutoff on the separate TRAIN calibration windows. Use source-proof-bound channel/length-specific envelopes, disjoint donors and explicit continuity/energy/spectral diagnostics. A zero or unsupported calibration envelope must reject, not trigger a permissive fallback. No VAL normalization, fit or cutoff choice is permitted.

This tests numerical distribution compatibility, not true background physiology or diagnostic absence. Passing the envelope cannot certify a physiological eraser, prevent removal of valid alternative evidence, or replace independent clinical criteria. Real accepted-donor coverage, donor disagreement and model effect stability remain NOT TESTED until separately authorized. Mock acceptance proves only software behavior. The future source comparison must not silently adopt this prototype as a validated donor policy.

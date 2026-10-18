# Versioned roadmap with hard stop gates

| Gate | Deliverable | Accept if | Must not do |
|---|---|---|---|
| D0 — DESIGN (current) | Prompt, hypotheses, experiment protocol, matrix, validator | Metadata-only validator + its synthetic tests pass; docs match P3B | No EEG opening, training, reserved or external accesses |
| I1 — SYNTHETIC (future authorization) | `src/mist_cove` standalone model, metadata/evidence dataclasses | Shapes, exact additive decomposition, finite logits/probabilities, no data IO | No historical runner modifications |
| I2 — FALSIFICATION TESTS (future authorization) | Controlled fixtures for H1/H2/H3, donor controls, quantitative synthetic reports | Pair-specific reliability diagnostic, donor-matched comparison, dual witness switch, collapse tests | No real clinical performance claims |
| S0 — SC SOURCE (future authorization) | Versioned source-only runner/config + fairness receipt | Locked P3A roles, reproducible TRAIN-only stats, matched budget, CUDA exact resume and leak guards | No RESERVED/ST/SHHS reading |
| S1 — FREEZE (future authorization) | Selected candidate/config commit and independent-eval preregistration | Limit tuning using four VAL people; log negative baselines, tie rules, acceptance margin | No target-informed selection |
| R0 — SC RESERVED (future separate authorization) | Single sealed 55-person evaluation | Protocol freeze verified, never used to revise model | No repeat evaluation for tuning |
| T0 — CROSS-STUDY (future separate authorization) | ST eligibility + exploratory evaluation | Fixed-clock/treatment/identity/channels reviewed | No independent-cohort claim |
| T1 — EXTERNAL (future separate authorization) | Verified independent cohort, inductive zero-shot evaluation | Permissions, montage/labels/independence proven, external protocol frozen | No unlabeled target fitting |
| F0 — ADAPTATION (future separate authorization) | Disjoint target-support few-shot comparison | Fixed 0/25/100/500 budgets, robust repeated support draws, no test leakage | No test-informed support selection |

## First coding-agent iteration (only after I1 authorization)
1. Document repo state and baseline interface signatures.
2. Create `src/mist_cove/__init__.py`, `model.py`, `evidence.py` with no external data I/O; use torch already in dependency tree.
3. Unit-test fixed 3000-sample input, five logits, optional 4–8 event windows, exact logit reconciliation, sign-preserving contributions and eventless inputs.
4. Implement synthetic perturbation constructors distinct from augmentation training code, record random seeds.
5. Validate under CPU-only CI; measure resource/VRAM only if actually run; a 16 GB RTX 4080 SUPER is a constraint, not a measured guarantee.
6. Open PR showing contract tests and explicit synthetic-only status.
7. Request separate permission before source real training; **do not run next gate automatically**.

## Hypothesis priority and selection
Start H1 conditional observability and H2 counterfactual witness faithfulness separately. Attempt H3 substitution only after baseline evidence representation works. H4 epoch-phase is a separate optional study requiring correctly scoped label semantics. No MoE unless measurable evidence conflicts motivate it and parameter-matched controls are retained.

## Documentation responsibilities
Update this roadmap with what actually passed, list negative outcomes, attach exact config/commit fingerprints, and mark gates BLOCKED rather than retroactively claiming earlier authorization. Never overwrite P3B results.

## 2026-10-18 Phase I execution receipt

The user's subsequent Phase I coding task authorized only implementation and synthetic experiments. Tested implementation commit: `533df355dc07715b042de92896b2d9c21cd6f653`. Config file SHA256: `66a6d51ebd121ffdd36ce3b72a1550c91a0282bd6df57fd463e7582f1a85cee8`.

I1 correctness: **PASS**, 53 new synthetic tests; full CPU regression **253 passed / 0 failed / 5 CUDA skipped**, one existing warning. I2 falsification infrastructure is implemented and executed: **H1 superiority FAIL in all three seeds**, constructed H2 direction/donor checks and H3 substitution checks pass, while counterfactual magnitude agreement and redundant-witness necessity remain limited. q=1 also solves the artificial discrimination fixtures. This is not an all-hypotheses success or evidence of transfer.

S0/R0/T0/T1 remain **BLOCKED pending separate phase authorization and protocol proofs**. No real TRAIN/VAL, RESERVED, ST, or SHHS data were accessed. No frozen P3B implementation, split, configuration, checkpoint, or metric was changed. See [the implementation note](PHASE_I_IMPLEMENTATION_V1.md) and [complete results and next-gate requirements](PHASE_I_RESULTS_V1.md).

# Original-proposal specification and Phase M0 scope

The authoritative source is the [13-page text mirror](original_proposal/MIST_SLEEP_ORIGINAL_PROPOSAL_TEXT.md) from source commit `85caad5e1e6a6c8ef744562e3a979478e7a0d57b`. The [registry](original_proposal/README.md) records the unavailable original PDF's expected byte hash. This document maps the original proposal to the narrower, already frozen Phase M0 experiment; it does not revise the proposal.

| Original section | Scientific requirement | Phase M0 treatment |
|---|---|---|
| 1. Executive Summary | Small-model, compute-efficient cross-dataset transfer question | Source-domain feasibility only; external claim untested |
| 2. Core Research Question | Transferability and label-efficient fine-tuning | Future independent protocol prepared; neither tested here |
| 3. Literature-Based Motivation | Compare AttnSleep, multi-resolution, DA and foundation-model ideas | Historical and public-source audit; no external model fit |
| 4. Refined Gap and Novelty | Define safe invariances and test beyond a parts-list combination | MPI isolated against same architecture and augmentation; novelty remains uncertain |
| 5. Formal Rationale | Separate morphology from nuisance; preserve stage information | Bounded gain/noise/wander test; destructive stress test; no true-domain claim |
| 6. Architecture | Overlap tokenizer, multi-scale filters, shared prototypes, temporal context | 29 physical within-epoch tokens, three scales, 64 prototypes, one-epoch GRU |
| 7. Ablations | Stage only, SupCon, MPI, combined; later term-specific tests | M0–M3 complete; embedding-only and assignment-only fits deferred |
| 8. Augmentation and shift predictions | Small gain/offset/noise/wander, shifts, restricted bandpass; reject destructive training transforms | First four used; shifts/bandpass excluded for exact alignment and physical safeguards; strong stress separate |
| 9. Questions and hypotheses | External gap, MPI beyond SupCon, representation preservation, label efficiency, negative transfer | Only source-development MPI/SupCon mechanisms and controlled nuisance tested |
| 10. Dataset scope | Three independent external targets when feasible; unusual clinical target | Public eligibility review only; zero new external waveform access |
| 11. Baselines | AttnSleep, capacity and multi-scale controls, DA and foundation model | Adapted AttnSleep and matched M0 available; remaining paper baselines deferred |
| 12. Settings | Within-dataset, direct transfer, target fine-tuning, DA, representation analysis | Original SC TRAIN fit/development and fit-only linear probe completed; external settings deferred |
| 13. Decision hierarchy | Fixed external transfer, stage and representation rules | Overnight exploratory +0.02 bounded-nuisance readiness rule evaluated; no confirmatory external claim |
| 14. Morphology validation | Spectral/shape statistics before clinical naming | Prototype occupancy and stage association measured; no clinical naming, no expert event validation |
| 15. Statistics | Five seeds if feasible, subject bootstrap, paired inference | Two fixed development seeds, five-person paired bootstrap; not confirmatory |
| 16. Negative transfer | Expose shift-specific failure | Destructive synthetic stress failure reported; real clinical negative transfer untested |
| 17. Timeline | Multi-month staged external study | Phase M0 is an overnight internal feasibility step only |
| 18. Final statement | Falsifiable compute-efficient transfer study | Paper-level protocol drafted, outcome presently NO-GO for an efficacy claim |

The original proposal's per-subject or per-recording normalization and possible across-epoch context are constrained by the overnight prompt's 10-fit-only statistics and strict within-epoch boundary. The original K=32/K=128 and EMA variants would add scientific fits and were excluded from the fixed ten-run matrix. The PDF binary was not found in the available attachment filesystem, so its hash and page count are registry references, not a locally reverified binary claim.

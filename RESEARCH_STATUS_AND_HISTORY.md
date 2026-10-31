# Research status and history — Proto-sleep / MIST-Sleep

**Last reviewed:** 2026-10-31  
**Scope:** public GitHub branches and committed aggregate reports, not private run directories.  
**Overall status:** research software and several analyses are complete, but **improved zero-shot transfer, clinical event localization, and sparse-MoE benefit are not established**. Do not interpret a passed software gate or an internal explanation certificate as a validated biomedical result.

## Where the work lives

This repository preserves parallel and historical branches rather than one merged research codebase:

| Line | Branch | Most relevant evidence |
| --- | --- | --- |
| Original Proto-sleep / MorphMAE | [`main`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/main) | Legacy README and source-recovery documents; **not** the latest transfer implementation |
| MIST-Evidence reference | [`research/mist-evidence-v1-verified`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/research/mist-evidence-v1-verified) | Authentic training-waveform evidence, provenance, bounded software checks |
| Morph Transport v2 | [`research/mist-morph-transport-v2`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/research/mist-morph-transport-v2) | Experimental comparator; not a confirmed transfer advance |
| MIST-Transfer v3 | [`research/mist-transfer-v3-edf-shhs`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/research/mist-transfer-v3-edf-shhs) | P0–P3B dataset integrity, frozen splits and source-VAL baseline results |
| MIST-COVE | [`research/mist-cove-phase16-expert-event-benchmark-v1`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/research/mist-cove-phase16-expert-event-benchmark-v1) | Exact dependency-audit / witness-circuit program through Phase XVI |
| Original MIST-Sleep / MPI recovery | [`research/mist-sleep-phase-m1-independent-hmc-transfer-v1`](https://github.com/SyedNaveedMahmood/Proto-sleep/tree/research/mist-sleep-phase-m1-independent-hmc-transfer-v1) | Morphology-preservation objective; HMC external-evaluation attempt **blocked** |

Each branch has its own historical gates, inputs and evaluation roles. **Do not merge, combine results, or treat datasets as independent solely because they have different filenames.**

## Scientific objective and why it evolved

The project began as prototype-guided single-channel sleep staging with MorphMAE/ProtoMAE and morphology-oriented representation learning. Early reconstruction and prototype experiments suggested useful structure, but reliable superiority and external transfer were not demonstrated. The focus shifted from simply attaching prototypes to a classifier toward **whether source-traceable EEG morphology and stable correspondence under acquisition variation can improve inductive zero-shot transfer**. This is a **hypothesis**, not an achieved result.

The newer work separated into complementary questions:

1. **MIST-Transfer v3:** Can a trustworthy raw-EDF pipeline and matched baselines support leakage-safe evaluation across subjects/studies?
2. **MIST-Sleep MPI:** Does a morphology-preserving invariance loss improve robustness and actual stage discrimination beyond ordinary augmentation?
3. **MIST-COVE:** What can an exact, dependency-aware explanation *prove* about a trained classifier, and how does that differ from physiological event attribution?
4. **Future exploratory sparse MoE:** Could morphology-conditioned expert routing improve transfer over compute-matched dense and ordinary sparse-MoE controls? **No demonstrated MoE outcome is recorded here.**

## Chronological research history

### Early ProtoMAE, MorphMAE and MorphSpec work (legacy / early branches)

- Recovered historical MorphMAE code/checkpoint compatibility and documented train-only fold-specific pretraining; the earlier model pretrained on full EDF-78 NPZ is **not a clean EDF-20 held-out benchmark**.
- Investigated prototype-guided scoring, morphology-oriented SSL and frozen representations. Some representation measures were promising, but this alone did **not** prove downstream or cross-dataset gains. See [historical recovery](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/main/docs/MORPHMAE_TRANSFER_RECOVERY.md) and [MorphSpec frozen-transfer confirmation](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/main/docs/MORPHSPEC_FROZEN_TRANSFER_CONFIRM.md).
- Developed MIST-Evidence real-waveform exemplars and bounded provenance/evidence checks. Early short CUDA runs were integration smokes, **not** benchmarks or transfer evidence.

### MIST-Transfer v3: P0–P1.5 data audit (October 2026 development)

- Inventoried EDF-20 NPZ (20 people/39 recordings) and EDF-78 NPZ (78/153), plus original Sleep-Cassette (SC) and Sleep-Telemetry (ST). The EDF-20 and EDF-78 NPZs share **20 people and 39 identical signal payloads**; this cannot be called independent external transfer.
- Recovered original recording epoch indices, gaps, units and annotations. Repaired seven ST EDF+ parsing cases using a validated read-only path, yielding **44/44 ST annotation-verified pairs**.
- Identified that prior NPZ retention was label-conditioned; its exact producing script remains unknown. **Trimmed NPZs are barred from inductive zero-shot evaluation.**
- Quarantined every night of SC persons **06, 23, 36 and 74** owing to unresolved demographic/annotation conflicts. Full SC remains out of scope; audited raw SC subset is eligible.

Primary evidence: [P1.5 decision](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-transfer-v3-edf-shhs/reports/data_audit/p1_5/gate_decisions.json).

### P2-SC: canonical raw signal processing

- Verified **74 people / 146 recordings / 397,832 complete 30-second physical epochs**, including **396,378 scored** and **1,454 unscored** positions retained for inference.
- Preserved calibrated Fpz-Cz, 100 Hz, original time indices, stage masks, complete physical grids, waveform provenance, strict split guards and lazy loading.
- Generated waveform arrays and participant-bearing artifacts remain local and are not GitHub research results.

Evidence: [experiment registry](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-transfer-v3-edf-shhs/experiments/registry.yaml).

### P3A–P3B: source-only baseline development and validation

- Frozen **15 SC TRAIN / 4 SC VAL / 55 reserved SC subjects** before training. All 55 extension subjects remained closed to development.
- P3B completed **12 matched runs**: AttnSleep, evidence, summary_only and raw_context, each at seeds **123, 456, 789**; no MCR loss, ST fitting, SHHS or reserved evaluation.
- Reported *mean source-VAL participant Macro-F1* across three seeds (only **four validation people**):

| Model | Source-VAL mean subject Macro-F1 |
| --- | ---: |
| AttnSleep repository adapter | 0.31265 |
| Evidence | 0.71803 |
| Summary only | 0.51216 |
| Raw context | 0.72351 |

These are **development checkpoint-selection metrics**, not independent held-out or external results. The AttnSleep score is specific to the adapted implementation, not proof about the published architecture in general. The small validation group limits inference. P3B logged 433 realized epochs and 2,155,474 optimizer updates across all runs, without numerical/software failures.

Evidence: [P3B report](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-transfer-v3-edf-shhs/reports/p3b/FINAL_COPY_PASTE.txt), [protocol](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-transfer-v3-edf-shhs/docs/TRANSFER_PROTOCOL.md).

### MIST-COVE: exact model-dependency auditing through Phase XVI

- Explored source-traceable evidence, robust explanation and ultimately **exact finite response tables / Boolean witness circuits** under a *specified internal feature-withholding operator*. These certificates verify classifier response dependencies, **not** that a signal interval is a real spindle or a correct clinical explanation.
- Several prospective architecture/robustness improvements **failed** registered gates. In Phase XIII, longer optimization improved developmental classification, but reliability routing failed its incremental benefit test; expert morphological validation was not performed.
- Phase XIV used the authenticated frozen P3B raw_context classifier on **44 ST recordings / 22 people**. This was an exploratory cross-study proxy, not an independent clinical cohort confirmation. ST pooled Macro-F1 was **0.5839**, mean person Macro-F1 **0.5583**, and REM recall **0.2162**; calibration degraded materially. The registered sigma-spectrum incremental event-localization gate **failed**. Internal-vs-physical waveform intervention comparisons often disagreed.
- Phase XV replicated the **8,349 scored ST REM epochs** and examined 360 selected explanation cases without training. On 55 proxy-eligible cases from 20 people, exact-positive interval precision was **0.171**, compared with **0.217** for the equal-duration sigma baseline; person-level difference **−0.0462**, interval **[−0.0797, −0.0167]**. These are *automated event proxies on previously examined ST outcomes*, **not human-scored event validation**.
- Phase XVI implemented rights-gated, generated-only expert-event benchmark software. The DREAMS archive/use permission was **unclear**, no real DREAMS EEG/annotations were acquired, and human event accuracy is **NOT TESTED**. Central-channel DREAMS data are also not interchangeable with the original Fpz-Cz P3B classifier input. Software tests passed; scientific event-validation gates remain blocked.

Evidence: [Phase XIV decision](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-cove-phase16-expert-event-benchmark-v1/docs/mist_cove/PHASE_XIV_SCIENTIFIC_DECISION_V1.md), [Phase XV report](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-cove-phase16-expert-event-benchmark-v1/docs/mist_cove/PHASE_XV_RESEARCH_REPORT_V1.md), [Phase XVI report](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-cove-phase16-expert-event-benchmark-v1/docs/mist_cove/phase16/RESEARCH_REPORT_V1.md).

### Original MIST-Sleep morphology-preservation / MPI recovery (Phase M0–M1)

- Implemented the original **Morphology Preservation Invariance (MPI)** objective, combining within-instance morphology-representation L2 alignment with KL consistency of prototype assignments under bounded views. Its mechanism is related in spirit to the proposed correspondence-reliability direction, but names, architectures and experimental evidence should not be conflated.
- The two-seed source-development test found reduced representation drift/KL, but **the prespecified +0.02 robustness improvement over augmentation-only was not reached**: M2-minus-M0 mean participant Macro-F1 improvement on the bounded view was about **+0.00585**. The gate outcome was **`MPI_FEASIBILITY_NOT_ESTABLISHED`**.
- An independent HMC v1.1 evaluation was prepared using official metadata and **151 EDF headers**, with four model identities authenticated. **No HMC waveforms or scoring outcomes were read and no external predictions were computed** because available persistent storage (~7.06 GB) was insufficient for the ~15.7 GB collection plus safety margin. The external outcome is **`PRIMARY_EVALUATION_BLOCKED`**, *not* negative transfer.

Evidence: [M0 scientific decision](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-sleep-phase-m1-independent-hmc-transfer-v1/docs/mist_sleep/OVERNIGHT_SCIENTIFIC_DECISION_V1.md), [M1 scientific decision](https://github.com/SyedNaveedMahmood/Proto-sleep/blob/research/mist-sleep-phase-m1-independent-hmc-transfer-v1/docs/mist_sleep/PHASE_M1_SCIENTIFIC_DECISION_V1.md).

## What is demonstrated vs. unresolved

**Demonstrated (within stated scope):** auditable SC raw-epoch processing; source-only matched development training; finite exact-model response auditing under explicitly defined operators; authenticated historical checkpoint use in bounded investigations; descriptive source-to-ST weaknesses.

**Not established:** a reproducible improvement in **independent-cohort zero-shot transfer** from morphology preservation / correspondence reliability; a significant benefit of sparse morphology-conditioned MoE; superiority of exact explanation supports over simple sigma for event localization; clinician-verified EEG event explanations; general clinical robustness, independent HMC evaluation or SHHS evaluation.

**Current gates:** 55 SC extension people remain reserved in the MIST-Transfer P3B protocol; ST has been exposed by *separate* MIST-COVE investigations and must not be rebranded as untouched confirmation; HMC external evaluation is blocked by data availability; SHHS data/access remain unverified in the reviewed transfer line; DREAMS expert-event work is blocked by rights/access and input-channel compatibility.

## Next decision points (proposals only)

1. Select **one** primary paper claim: (a) exact dependency-audit method with candid negative physiological comparisons, or (b) prospective morphology-reliability/transfer method with new held-out confirmation. Do not merge their evidence into one claimed proof.
2. For transfer research, secure a **genuinely independent, authorized** target study, audit participant overlap, channels, stage mapping, scoring coverage and data rights, and freeze source-only model/hyperparameters *before* target results.
3. Compare MCR/MPI and a proposed morphology-conditioned sparse MoE against augmentation-only, ordinary latent consistency, dense networks and generic MoE with matched active compute and protected test participants. All are **future experiments**.
4. For event explanations, obtain rights-cleared, channel-compatible, expert-annotated evaluation with independent reviewer agreement and a prespecified equal-budget sigma comparator. Do not substitute generated fixtures or automatic proxies for ground truth.

## Reporting rules

- Use labels `VERIFIED`, `CONDITIONAL`, `BLOCKED`, `UNKNOWN`, or `NOT TESTED` with the measured scope attached.
- Never equate EDF-20 vs EDF-78 with independent cross-dataset transfer.
- Do not use target-test labels or target statistics for normalization, exemplar selection, tuning or model choice in an inductive zero-shot claim.
- Keep private participant EEG, raw annotations, waveform banks, checkpoints, secrets and sensitive per-epoch outputs out of public GitHub.
- The source of truth for any numerical result is the **versioned report on its own branch**, not this cross-branch synopsis. New branch results should cause a dated update of this file.

# MIST-Sleep: Morphology-Preserving Invariance for Compute-Efficient Cross-Dataset EEG Sleep Staging

**Draft status:** method and source-domain feasibility only. No independent external-target result exists. This is not ready for submission as a transfer paper.

## Abstract

Single-channel EEG staging models may be sensitive to acquisition variation. We implemented a compact morphology tokenizer with multi-scale filters, a shared 64-prototype bank and a within-epoch temporal encoder, and tested whether a morphology-preserving invariance (MPI) loss adds value beyond the same architecture trained on the same augmented views. On an already studied 10-fit/five-development participant partition of original Sleep-Cassette TRAIN, ten fixed trajectories compared an adapted AttnSleep reference, stage-only morphology training, SupCon, MPI, and their combination. MPI lowered bounded-view embedding drift and prototype-assignment divergence in both seeds, without prototype collapse. Its two-seed mean development Macro-F1 gain over the matched stage-only control was +0.0037 clean and +0.0058 under bounded nuisance, below the frozen +0.02 nuisance-readiness threshold. A destructive stress view erased performance across arms. These exploratory findings establish implementation feasibility but do not establish robust stage benefit, cross-dataset transfer, clinical morphology or label efficiency.

## 1. Introduction

The original [MIST-Sleep proposal](original_proposal/MIST_SLEEP_ORIGINAL_PROPOSAL_TEXT.md) asks whether explicit waveform-morphology invariance improves transfer under acquisition shift while preserving sleep-stage information. The question matters because a representation can become stable by discarding discriminative content. We isolate MPI from architecture and ordinary augmented training before spending resources on independent external cohorts.

## 2. Scientific motivation and related work

[AttnSleep](https://github.com/emadeldeen24/AttnSleep) provides a strong published single-channel architecture reference, though this branch uses an adapted training contract. [Supervised contrastive learning](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) clusters labeled instances; MPI compares two nuisance views of the **same** instance without consulting its label. Shared prototype assignments resemble broader prototype representation-learning ideas such as [SwAV](https://ai.meta.com/research/publications/unsupervised-learning-of-visual-features-by-contrasting-cluster-assignments/). Thus the methodological claim cannot rest on a parts list. Its potential distinctiveness depends on safe transformation semantics, two independent stability measures, matched causal isolation and genuinely independent transfer validation. No first-in-the-world novelty claim is made.

## 3. Formal assumptions

Write an observed epoch as a function of morphology (m) and acquisition nuisance (n). For a bounded transformation (t\in\mathcal T_{\mathrm{preserve}}), the working assumption is that (t) changes mainly (n), leaving clinically relevant (m) and the stage label approximately intact. This assumption can fail when montage or physiology alters visible waveform morphology. N2/N3 are plausible stronger morphology cases; single-channel N1/REM are weaker. The assumption is a falsifiable premise, not a theorem or clinical guarantee.

## 4. Architecture and preprocessing

Each calibrated 30-second Fpz-Cz epoch at 100 Hz is represented by 29 two-second windows at a one-second hop. Parallel 7-, 31- and 101-sample convolution kernels create 64-dimensional local tokens. One learned positional vector per token and a 64-dimensional GRU aggregate only within the physical epoch. A 64-by-64 shared normalized prototype bank at temperature 0.1 provides soft assignments and a prototype mixture to the five-class head. The parameter count is 41,429. Phase M0 reused the authenticated ten-fit-only SC TRAIN normalization from Phase XIII; all development samples use those fixed statistics.

## 5. Prototype learning

After 50 identical supervised warmup updates, 4,096 embeddings from fit-only tokens initialize cosine k-means. The bank then trains by gradient descent with the same commitment, usage-balance and separation terms in M0–M3. Per-seed k-means input digests match exactly across the four arms. Effective occupancy, dead units, assignment entropy and near-duplicate vectors are measured; no clinical name is inferred from these statistics.

## 6. Objective and ablations

For original (x) and aligned transformed (t(x)), MPI is the sum of L2 distance between unit-normalized pooled representations and directional (D_{\mathrm{KL}}(q(x)\Vert q(t(x)))) over corresponding token assignments, each with frozen coefficient 0.1. Stage cross-entropy averages logits from both views. M0 adds no auxiliary objective; M1 adds label-based SupCon at coefficient 0.05; M2 adds MPI; M3 adds both. A0 is an adapted AttnSleep architecture reference, not an MPI causal control. The primary comparison is M2–M0.

## 7. Transformation policy

The training view combines gain 0.95–1.05, constant offset ±0.05 normalized units, Gaussian noise SD 0.02 and 0.2 Hz wander amplitude 0.03. The token grid remains aligned. Shift and bandpass jitter were excluded from this bounded run because the overnight protocol required verified correspondence and physical safety before use. A stronger destructive view is reserved for failure analysis. A 100-example fit-only audit found finite outputs, no clipping, and bounded RMS and spindle/slow-band power changes; it does not validate clinical label preservation.

## 8. Experiment and statistics

The authenticated original SC TRAIN partition contains ten fit and five historically exposed internal-development people. All 29 recordings and 79,984 physical epochs retained their original coordinates. Two seeds (123, 456) each trained five arms for 1,200 updates at batch 16. The final checkpoint was fixed, not selected on development data. The primary exploratory endpoint was mean participant fixed-five Macro-F1 on the full scored development grid. Paired five-person bootstrap intervals describe each seed's contrast; they cannot substitute for independent confirmation. The original paper-level plan calls for more seeds and external targets, neither available here.

## 9. Source-development feasibility results

At the final checkpoint, M0 clean participant Macro-F1 was 0.6902/0.7268 across seeds, and M2 was 0.6896/0.7349. Under the frozen bounded nuisance, M0 was 0.6905/0.7226 and M2 was 0.6910/0.7338. The two-seed mean M2–M0 differences were +0.0037 clean and +0.0058 bounded. The latter missed the +0.02 overnight readiness threshold. SupCon increased N1 recall but harmed REM in the second seed; adding SupCon to MPI did not yield a consistent gain. The adapted A0 reference was 0.6889/0.6534 clean and cannot establish superiority over published AttnSleep.

## 10. Representation and prototype results

M2 reduced bounded-view embedding L2 drift from 0.0371 to 0.0277 and from 0.0257 to 0.0212 across seeds, relative to M0. Prototype KL fell from 0.00407 to 0.00297 and from 0.00228 to 0.00201. M2 effective prototype counts were 59.2 and 57.5 of 64 with zero dead units, but 49 and 36 near-duplicate pairs. A fit-only linear probe on frozen embeddings scored 0.7884/0.7895 for M2 versus 0.7923/0.7885 for M0 on a disjoint development sample; stage-information gain was not consistent. The two MPI terms produced distinct nonzero gradients, but isolated term-training ablations remain unperformed.

## 11. Compute

The complete campaign used 10 trajectories, 12,000 optimizer updates and 384,000 original/augmented training views in 95.44 observed GPU seconds. Morphology models had 41,429 parameters versus 522,805 for adapted A0. M2 mean trajectory time was 8.54 seconds versus M0's 8.13 under the same in-memory hardware setup. Batch inference and source-I/O costs must be reported separately in any paper-level efficiency comparison.

## 12. Failure and negative transfer analysis

The deliberately destructive stress view reduced all arms to roughly 0.16–0.21 participant Macro-F1; M2 did not protect REM recognition. This is an artificial failure control, not a real clinical negative-transfer result. The recovered proposal explicitly predicts possible failure under age, pathology, medication, fragmentation and montage changes that alter morphology itself. Those risks require future authorized independent cohorts.

## 13. External evaluation plan

The [future transfer protocol](FUTURE_TRANSFER_EVALUATION_PROTOCOL_V1.md) defines a source-trained M2–M0 inductive contrast, subject-independent external test sets, a fixed primary rule, capacity and DA controls, label-efficiency curves, representation analysis and target-level failure reporting. HMC is a potentially informative clinical target but its central derivations differ from SC Fpz-Cz; channel compatibility must be audited before interpreting any result. No external EEG was opened for this draft.

## 14. Limitations

The development cohort has been used in earlier phases; its five participants and two model seeds cannot support confirmatory inference. The original PDF binary is pending even though its full page-by-page text is tracked. The input is single-channel and cannot represent all expert REM evidence. Transformations are numerically bounded but not clinician-validated. The completed runs were uninterrupted, while exact CUDA resume passed only after enforcing deterministic runtime flags in a generated test. Prototype interpretation, domain invariance, label efficiency and external efficacy are untested.

## 15. Ethics, data and reproducibility

Only previously authorized original SC TRAIN recordings were read. SC VAL, RESERVED, ST and all new external cohorts had zero reads. Private waveform cache and model states are ignored by Git; committed results are aggregate. Source, protocol, checkpoints and report digests provide an audit trail. CI uses generated fixtures only.

## 16. Claim hierarchy and next decision

The implementation supports a narrow statement: MPI slightly improves stability of this compact model under one bounded synthetic acquisition-like view without reaching the frozen stage-performance gate. It does not justify a transfer-efficacy statement. Scientific reviewers should decide whether a new independently authorized external test is worth running despite this NO-GO feasibility result. If authorized, the decisive test is unchanged source-trained M2 versus M0 on a prespecified, independent and channel-audited target.

## 17. References

See the [original proposal's references](original_proposal/MIST_SLEEP_ORIGINAL_PROPOSAL_TEXT.md) and the primary-source links above. Dataset candidate facts are sourced in the [future transfer protocol](FUTURE_TRANSFER_EVALUATION_PROTOCOL_V1.md). External citations in the original proposal remain literature context, not evidence that this branch ran their experiments.

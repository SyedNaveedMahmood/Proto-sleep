# next_plan.md - MIST-SSDG: shape-routed experts for single-source sleep-staging generalization

**Status:** PLAN ONLY. Nothing in this file is a completed experiment, a measured result of the proposed method, or a verified SOTA or novelty claim.
**Plan prepared:** 2026-11-18.
**Branch:** `research/mist-ssdg-shape-routed-v1`.
**Base:** `research/mist-transfer-v3-edf-shhs` at `15995b31821ee8db717f3d6edfa41166de65d4a3`.
**Supersedes:** the MCR plan of 2026-10-09, archived unchanged at `docs/archive/next_plan_v3_mcr_2026-10-09.md`.

Labels used below: `VERIFIED` (read from this repository or from the cited source), `INFERENCE` (reasoned from verified facts, untested), `SPECULATION` (a bet), `UNVERIFIED` (must be checked before relying on it).

## 0. One-paragraph summary

Build and freeze everything on open datasets, keep SHHS untouched until the freeze, then use SHHS as the locked final test. The benchmark is single-source domain generalization (train on one dataset, test on the others), where the published number to beat is SleepBand's 67.08 average macro-F1. The model is a standard sequence-to-sequence sleep stager with three new parts: an acquisition-invariant front end, a shape-only token router with additive experts, and an optional two-pass stage-conditional reference. A per-expert, per-class ledger of the transfer gap answers the original question of which morphology matters, whether or not the accuracy claim succeeds.

## 1. Why the plan changed

### 1.1 What this repository already shows

- `VERIFIED` Inputs use one global source-TRAIN microvolt z-score (`source-train-global-uV-zscore-v1`, `src/mist_transfer/source_data.py`). The only gain augmentation is +/-2% (`GainEvidence`, `src/mist_transfer/baselines.py`).
- `VERIFIED` Phase XIV measured ST as +20.73 microvolts stage-balanced centered RMS above SC, and spindle-candidate rates of 32.48/hour (SC) against 121.87/hour (ST) on Fpz-Cz.
- `VERIFIED` Frozen SC to ST recall (SC figures are in-sample): Wake 0.994 to 0.909, N1 0.504 to 0.281, N2 0.921 to 0.906, N3 0.869 to 0.779, REM 0.791 to 0.216.
- `INFERENCE` The stages with clear positive morphology (N2, N3) transfer; the low-amplitude stages (REM, N1) collapse. That pattern fits an uncorrected gain shift. No arm in the reviewed branches was tested with per-recording scaling, so this is untested.
- `VERIFIED` Morphology objectives have not moved F1: MorphMAE initialization -0.0055 against baseline; MPI +0.00585 against a +0.02 gate.
- `VERIFIED` P3B models are epoch-only with 13,556 and 34,197 parameters, trained on 15 subjects and validated on 4. Their 0.72 source-VAL macro-F1 is not comparable to any published table.

### 1.2 Why not a direct MoECLIP port

- MoECLIP adapts a frozen CLIP whose zero-shot generalization is worth protecting. Sleep EEG has no such backbone: the NeuroAtlas benchmark reports EEG foundation models level with supervised sleep stagers only at single-window level, and behind them once sequence context is allowed.
- MoECLIP's own ablation attributes roughly one point of pixel AUROC and two of image AUROC to its orthogonal-subspace and ETF pieces over a plain MoE. They are small effects.
- MoE for EEG is crowded: MoESleepNet (multi-view experts, single-channel sleep), SREA (shared plus routed experts, cross-subject, includes Sleep-EDFx and ISRUC-3), SpecMoE (spectral gating), and SleepBand (single-source generalization with a learnable Morlet filter bank).
- What is kept from MoECLIP: token-level top-k routing, disjoint expert input subspaces, the ETF loss and the load-balance loss.

## 2. Research questions

- **RQ1.** Does shape-only routing with night-referenced amplitude reduce the cross-dataset gap beyond what per-recording normalization alone achieves?
- **RQ2.** For each sleep stage, which expert carries the transfer gap, and is it routing shift (how often a morphology occurs) or response shift (how the same morphology is read)?

RQ2 yields a finding even if RQ1 is null.

## 3. Benchmark and the claim

| Benchmark | Number to beat | Decision |
|---|---|---|
| In-dataset Sleep-EDF-20, single channel | Published claims up to 86.7% accuracy, 81.7 macro-F1 | Skip: saturated, protocol-inconsistent |
| Single-source generalization, 5 datasets, 20 source-to-target pairs | SleepBand 67.08 average macro-F1, 72.25% accuracy; plain training 62.98 macro-F1 | Primary |
| Leave-one-dataset-out | SleepBand 72.61 average macro-F1; plain training 70.35 | Secondary, if time allows |

SleepBand facts (`VERIFIED` from arXiv:2607.04851): one EEG plus one EOG channel per dataset (SleepEDF Fpz-Cz + horizontal EOG; HMC F4-M1 + E1-M2; ISRUC F4-M1 + E1-M2; SHHS1 C4-M1 + ROC-LOC; CinC 2018 C3-M2 + E1-M2); 100 Hz, 0.5-35 Hz bandpass, z-score; sequences of 20 epochs; 1.15M parameters.

Claim hierarchy:

1. Beat 67.08 on the full 20-pair table: SOTA on single-source generalization, with SHHS as a pre-registered locked test.
2. Beat SleepBand re-run by us but not the published number: report both.
3. Either way: the transfer-gap ledger (RQ2).

"Completely novel" and "SOTA" cannot both be promised. The design is a standard skeleton plus new mechanisms placed where the benchmark is weakest.

## 4. Decisions frozen before any run

- **Inputs:** EEG + EOG primary; EEG-only as a secondary table.
- **Datasets:** Sleep-EDF, HMC, ISRUC, CinC 2018, SHHS1.
- **Preprocessing:** 100 Hz, 0.5-35 Hz bandpass, z-score, matching SleepBand.
- **Normalization rule:** recording-local, label-free statistics are permitted and applied identically to every arm and every dataset. This amends `docs/TRANSFER_PROTOCOL.md`, which currently forbids target statistics. The amendment must be committed before the first run. Statistics pooled across target recordings remain forbidden.
- **Selection:** every hyperparameter is chosen on source validation only. At least three seeds per arm.
- **SHHS rule:** no SHHS data in any form before the freeze in phase 7. This includes the third-party "Preprocessed SHHS1 dataset" deposit (doi:10.21979/N9/EAMYFO), which is C4-A1 only at 125 Hz, has no EOG, and whose licence was not read.
- **Reused targets:** ST has been exposed by MIST-COVE and is development data, never confirmation.

## 5. Architecture

### 5.1 Standard skeleton (known parts)

- EEG + EOG input, sequences of 20 epochs, sequence-to-sequence cross-entropy.
- Epoch encoder of roughly 0.3-0.5M parameters and a small transformer over epochs, about 1M parameters in total (`SPECULATION` on exact size; chosen to match SleepBand's scale).

### 5.2 New part 1: acquisition-invariant front end

Treat each recording's acquisition chain as an unknown filter.

- Cut each 30-s epoch into 2-s tokens with 1-s stride (29 tokens).
- **Shape stream:** divide each token by its own RMS; use a sign-symmetric stem (shared weights on x and -x, then pool).
- **Amplitude stream:** log-RMS and five band envelopes per token, each expressed as a robust rank within that recording.
- By construction the model is invariant to per-recording gain, polarity and band-wise gain. This is the sleep analogue of cepstral mean normalization in speech.
- Known weakness: night statistics shift with stage composition. Part 3 targets this.

### 5.3 New part 2: shape-only router with additive experts

- A linear router on shape features selects top-2 of K = 4 experts. It never sees amplitude, so a pure gain change between datasets cannot change routing.
- Each expert reads a disjoint slice of the shape features plus the amplitude ranks, and emits per-token stage evidence and a pooled embedding.
- Epoch logits = sum of token evidence (local term) + sequence-model output (context term).
- Auxiliary losses, each ablatable: ETF on expert outputs (weight 0.01 as in MoECLIP), load balance (squared coefficient of variation of expert load).
- In a stem trained from scratch, the frozen orthogonal split reduces to disjoint channel groups. It carries less meaning than in MoECLIP.

### 5.4 New part 3 (optional): two-pass stage-conditional reference

The first pass yields stage posteriors. The second pass re-references each token against that person's own predicted-N2 or predicted-wake background. No gradient updates at test time. High risk; run only if phase 4 passes and stage composition visibly hurts on HMC.

### 5.5 Transfer-gap ledger (RQ2)

With n_k the routing share of expert k and m_k its mean evidence for the true class, the local margin is the sum over k of n_k * m_k. The source-to-target change splits exactly:

```
delta = sum_k [ dn_k * m_k(S) ]      routing shift
      + sum_k [ n_k(S) * dm_k ]      response shift
      + sum_k [ dn_k * dm_k ]        interaction
```

Illustrative numbers only (not data): n 0.60 to 0.55 and m +1.2 to +0.3 gives -0.555 = -0.060 routing - 0.540 response + 0.045 interaction. The ledger must reconstruct the local logits to FP32 tolerance; a failed reconstruction is a hard stop. Target labels are used only for this post-hoc analysis, never for selection.

### 5.6 Expectations

- `INFERENCE` Part 1 is the component most likely to move cross-dataset numbers.
- `SPECULATION` Part 2 adds accuracy over a dense matched model. Its firmer value is the ledger.
- `SPECULATION` Part 3 helps on clinical cohorts.

## 6. Phases and gates

Each phase is committed separately with its report. A failed gate stops the line; do not retune to rescue it.

**Phase 1 - parallel setup.** Submit the NSRR application for SHHS. Get SleepBand's public code running. Check disk space. Commit the protocol amendment from section 4.

**Phase 2 - open-data pipeline.** Stream-extract EEG + EOG one file at a time (download, extract two channels at 100 Hz, delete the original) for Sleep-EDF, HMC and ISRUC. Add CinC 2018 when storage allows. Estimate: two channels at 100 Hz in float16 is about 11.5 MB per 8-hour night. Reuse the P2 manifest, hashing and split guards.
*Gate:* P2-style integrity checks pass on every dataset; channels, native rates and stage maps recorded from headers.

**Phase 3 - reproduce baselines.** Plain training and SleepBand on the open pairs.
*Gate:* match SleepBand's published numbers on pairs not involving SHHS, where the paper reports per-pair values. If not matched, fix the pipeline before anything else.

**Phase 4 - front end alone.** Add part 1 to the skeleton. Ablations: gain only, band-wise gain, sign symmetry.
*Gate (proposed threshold):* at least +2 average macro-F1 over plain training across the open pairs, three seeds. SleepBand's own margin over plain training is about 4 points; less than 2 is not competitive.

**Phase 5 - router and experts.** Add part 2. Arms: dense model of matched size with the same front end; router that also sees amplitude; full model.
*Gate:* full minus dense matched. If no gain, keep part 2 only as the measurement instrument for the ledger, provided it matches the dense model within seed noise.

**Phase 6 - two-pass reference (optional).** Conditions in 5.4.

**Phase 7 - freeze.** Tag code, configs, checkpoints and the hash of every manifest. Record the tag in `experiments/registry.yaml`.

**Phase 8 - SHHS locked test.** Requires approved NSRR access. Download SHHS1 only, with filtered commands, into the approved local root; never commit tokens or data:

```bash
nsrr download shhs/polysomnography/edfs/shhs1
nsrr download shhs/polysomnography/annotations-events-nsrr/shhs1
```

Extract C4-A1 and the EOG pair used by SleepBand. Run the frozen models on SHHS as a target, then train on SHHS as a source with the frozen hyperparameters. This completes the 20-pair table.

**Phase 9 - leave-one-dataset-out.** Second table, number to beat 72.61 average macro-F1.

**Fallback.** If access has not arrived by the freeze, write up the open-data results and add SHHS in revision. Phases 1-7 do not wait on it.

## 7. Endpoints and reporting

- **Primary:** average macro-F1 over the 20 source-to-target pairs, mean and spread over seeds.
- **Secondary:** accuracy, Cohen's kappa, per-stage F1 (N1 and REM in particular), calibration (ECE, NLL), per-pair table, EEG-only table.
- **RQ2 output:** class by expert ledger for each pair, with routing, response and interaction terms.
- Participant is the sampling unit for intervals; seeds are repeated optimizations, not independent subjects.
- Report every run, including failed gates.

## 8. Open items to verify

- `UNVERIFIED` Whether SleepBand's z-score is per recording, per epoch or global. If per recording, the gain part of the front end is already in their baseline.
- `UNVERIFIED` Which SHHS1 recordings SleepBand uses. The common subset is 329 recordings.
- `UNVERIFIED` Whether SleepBand's paper reports per-pair numbers (needed for the phase 3 gate).
- `UNVERIFIED` CinC 2018 download size and available disk space.
- `UNVERIFIED` Novelty of parts 1-3 combined. A handful of searches found no match; a full-text audit against SleepBand, SREA, MoESleepNet, SpecMoE and SleepDG is required before any novelty claim.
- HMC and ISRUC have no Fpz-Cz, so montage shift is confounded with dataset shift on every Sleep-EDF pair. Report it as such.

## 9. Hard stops

Any train/validation/test identity overlap; target data used for selection, normalization across recordings or architecture choice; SHHS opened before the freeze; ledger reconstruction failure; expert collapse (one expert taking nearly all load); non-finite gradients.

## 10. Sources

- Lu, Z., Hu, Y., & Chen, Y. (2026). SleepBand: Single-source domain generalization for sleep staging via physiologically structured spectral modeling. arXiv. https://arxiv.org/abs/2607.04851
- Park, J. Y., Seo, J., Kang, M., & Park, Y. R. (2026). MoECLIP: Patch-specialized experts for zero-shot anomaly detection. arXiv. https://arxiv.org/abs/2603.03101
- NeuroAtlas: Benchmarking foundation models for clinical EEG and brain-computer interfaces (authors not retrieved). (2026). arXiv. https://arxiv.org/html/2605.14698
- Aligning shared and routed experts for cross-subject EEG generalization (authors not retrieved). (2026). arXiv. https://arxiv.org/abs/2602.01728
- Darankoum, D., Habermacher, C., Volle, J., & Grudinin, S. (2026). SpecMoE: Spectral mixture-of-experts foundation model for cross-species EEG decoding. arXiv. https://arxiv.org/abs/2603.16739
- Wu, X., et al. MoESleepNet: A multi-view mixture-of-experts model for single-channel EEG sleep stage classification. Proceedings of the Annual Meeting of the Cognitive Science Society. https://escholarship.org/content/qt7395649p/qt7395649p.pdf
- Alvarez-Estevez, D., & Rijsman, R. (2022). Haaglanden Medisch Centrum sleep staging database (Version 1.1). PhysioNet. https://doi.org/10.13026/t79q-fr32
- Eldele, E. (2022). Preprocessed SHHS1 dataset (Version 1). DR-NTU (Data). https://doi.org/10.21979/N9/EAMYFO
- Automated sleep staging from single-channel electroencephalogram using hybrid neural network with manual features and attention (authors not retrieved). (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12341640/

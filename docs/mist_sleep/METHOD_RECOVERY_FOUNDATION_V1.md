# MIST-Sleep method recovery: source and scientific contract

The original 13-page proposal is now preserved as a [page-by-page text mirror](original_proposal/MIST_SLEEP_ORIGINAL_PROPOSAL_TEXT.md), integrated from the separately committed source registry after the bounded training had already completed. The verified PDF bytes remain unavailable in this environment. The proposal is authoritative for scientific intent; the Phase M0 prompt controls the narrower TRAIN-only scope and compute ceiling. Numerical settings absent from the proposal are identified as newly frozen engineering defaults in `frozen_protocol.json`. The model and protocol were frozen from the Phase M0 prompt before this source integration, so their agreement with the recovered proposal is assessed explicitly below rather than claimed retroactively.

## Question and estimand

The source-domain feasibility question is whether an explicit within-instance morphology-preserving invariance (MPI) objective improves representations beyond the **same architecture, paired inputs and optimizer exposure** trained with stage loss alone. M2 minus M0 is the primary causal contrast. M1 isolates supervised contrastive learning (SupCon); M3 tests combination. A0 is an adapted AttnSleep architecture reference, not a parameter-matched control and not a published-method replication.

All numerical findings in this branch are exploratory development findings on the already studied original SC TRAIN cohort. No external-target outcome is in scope.

## Source identity and isolation

The starting source commit is `5f68b051bbcd6ab0b9ba8b432b339d7c3ebf7c5d`. The Phase XIII authenticated partition provides 10 fit and five internal-development participants from original SC TRAIN. The 29 source recordings have 79,984 physical 30-second epochs; 55,153 scored fit epochs and 23,872 scored internal-development epochs. Every physical epoch retains its original within-recording position in the private cache. Unscored epochs remain in that cache and are excluded from supervised training and scoring.

All waveforms are calibrated Fpz-Cz at 100 Hz and exactly 3,000 samples. The Phase XIII **10-fit-only** mean and standard deviation normalize both roles. The 15-person historical P3B normalizer is not used for new training. The code authenticates descriptors, waveforms and annotations through the inherited TRAIN-only source access layer. No SC VAL, RESERVED, ST, SHHS, DREAMS or other external waveform is read. Private cache bytes, checkpoints and predictions stay under ignored `mist_transfer_runs/`.

## Reuse matrix

| Component | Decision | Reason |
|---|---|---|
| `phase11b/source.py` and `phase13/provenance.py` | Reuse without changing history | Authenticated original TRAIN read, physical epoch geometry and role guard |
| `phase13/sampling.py` | Reuse without changing history | Authenticated fixed 10/5 participant allocation and fit-only normalization |
| `protosleep/attnsleep.py` | Reuse as adapted A0 | Existing MRCNN, AFR and within-epoch attention; training contract differs from published cross-validation |
| `protosleep/prototypes.py` | Reuse mathematical conventions, implement new 64-by-64 bank | Existing 48-by-30 bank is coupled to AttnSleep AFR and cannot represent the proposed independent morphology architecture |
| `protosleep/losses.py` | Reuse KL/normalization ideas, implement independent MPI/SupCon terms | Existing losses concern masked macro reconstruction, not within-instance transformation consistency |
| `protosleep/mist.py`, `micro.py`, `morphmae_transfer.py` | Historical controls only | Different optimization objectives and/or architecture; importing them into MPI would confound the contrast |

## Implemented method

An epoch becomes 29 overlapping two-second windows at a one-second hop. Parallel local convolutions use 7-, 31- and 101-sample first kernels (0.07, 0.31 and 1.01 seconds). The second 5-sample convolution after stride four makes the **true two-layer receptive fields** 23, 47 and 117 input samples (0.23, 0.47 and 1.17 seconds); adaptive pooling lets each token summarize its full two-second input window. A 64-dimensional token representation receives a learned position vector. A 64-vector normalized prototype bank at temperature 0.1 gives soft assignments. A single-layer 64-dimensional GRU aggregates only tokens from the current epoch. The classifier uses pooled temporal and prototype-mixture representations to emit five logits in Wake/N1/N2/N3/REM order. No adjacent epoch or night label is available to this model.

The prototype bank initializes after 50 identical stage-loss warmup updates in each morphology arm. Fixed-seed cosine k-means uses 4,096 token embeddings from 256 fit-only epochs. The first 50 updates, subsequent sample order, view random stream, model initialization, batch size, architecture and optimizer are identical within each seed across M0–M3. Commitment, usage balance and prototype separation regularization are applied identically to those four arms.

For aligned original and transformed views, MPI adds mean L2 distance of unit-normalized pooled representations and mean directional `KL(q_original || q_transformed)` over corresponding physical tokens. Neither term uses labels. SupCon is a separate label-dependent, self-excluding batch contrastive objective over both views. M0 has neither auxiliary term; M1 has SupCon, M2 MPI, and M3 both.

The frozen training transformation applies amplitude scaling 0.95–1.05, a bounded constant offset (±0.05 normalized units), Gaussian noise SD 0.02 and 0.2 Hz baseline wander amplitude 0.03. These preserve the sample and token grid. Shift, warp, masking and aggressive band deletion are excluded because token correspondence or stage evidence could change. This is a numerical safeguard, **not** a clinical proof of label preservation.

## Reconciliation with the recovered original proposal

The implemented tokenizer, parallel multi-scale filters, 64 shared prototypes at temperature 0.1, source-fit-only k-means, five logits, MPI embedding distance and directional assignment KL, and the stage-only/SupCon/MPI/combined ablations match the proposal's method intent. The proposal permits small temporal shifts and restricted bandpass jitter; this overnight implementation excludes both because the prompt required exact token correspondence and a bounded physical safeguard before results. The proposal envisages an epoch-level temporal context encoder; the overnight prompt expressly restricts the model to within-epoch context, so this implementation aggregates the 29 morphology tokens inside one epoch. The proposal's per-subject or per-recording z-scoring is superseded here by the prompt's fit-only, role-safe normalization rule. The proposal's K=32/128, EMA, embedding-only, assignment-only and stronger-preprocessing ablations, quantitative clinical prototype naming, and external source-target experiments were not authorized inside the ten-fit overnight matrix. They remain planned rather than silently credited to this experiment.

The proposal predicts strongest morphology-driven gains in N2/N3, and treats N1/REM benefit as uncertain for single-channel EEG. The completed source-development outcomes must therefore display N1/REM separately. It also distinguishes acquisition shift, where MPI might help, from physiology shift, where removing morphological differences could cause negative transfer. The destructive stress view is an artificial falsification diagnostic, not a clinical domain test.

## Threats and controls

The primary contrast controls capacity, paired views, source split, sample order and optimizer exposure. A0 differs in parameter count and architecture, so M0–A0 is descriptive. Source fitting excludes all five development participants, including normalization and k-means. A class-balanced fit sampler addresses rare stages without class weights; full development scoring retains the natural class mix and fixed-five denominators. Prototype entropy, effective occupancy, duplicates, representation variance and stage-specific prediction coverage detect collapse. Clean and bounded-nuisance Macro-F1 must be considered together; a lower MPI loss by itself is not a useful result. The same five historically exposed development participants cannot establish external transfer, clinical morphology or independent confirmation.

## Prior evidence that constrains interpretation

Historical P3B four-participant source-VAL AttnSleep mean participant Macro-F1 was 0.312650 across seeds; raw-context averaged 0.723511. The selected AttnSleep seed-789 model predicted no N3 or REM. These results belong to an earlier training setup and do not prove the published AttnSleep architecture is weak. MorphSpec-R1 improved a held-out frozen probe, but its staged supervised comparison did not establish a transfer advantage over a fully trainable AttnSleep reference. Prior negative results and source-development reuse remain visible rather than being overwritten.

## Primary literature

- Eldele et al., [AttnSleep](https://github.com/emadeldeen24/AttnSleep), IEEE TNSRE 2021, DOI 10.1109/TNSRE.2021.3076234.
- Khosla et al., [Supervised Contrastive Learning](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), NeurIPS 2020.
- Caron et al., [SwAV prototype assignments](https://ai.meta.com/research/publications/unsupervised-learning-of-visual-features-by-contrasting-cluster-assignments/), NeurIPS 2020. This is related prototype representation learning, not evidence of novelty for MPI.

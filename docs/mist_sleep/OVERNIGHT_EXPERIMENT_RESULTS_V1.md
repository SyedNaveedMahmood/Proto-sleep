# MIST-Sleep Phase M0: bounded source-development feasibility result

**Decision: `MPI_FEASIBILITY_NOT_ESTABLISHED`.** The ten-run matched experiment finished within budget. MPI (M2) made representations more stable under the declared mild nuisance in both seeds, but its mean participant Macro-F1 gain over the architecture- and augmentation-matched M0 control was only **+0.00585** under that nuisance, below the locked +0.02 exploratory readiness threshold. Clean gain was +0.00370. Neither result is external transfer evidence.

## Design and source boundary

The source was the authenticated original SC TRAIN partition only: 10 fit and five reused internal-development participants, 29 recordings, 79,984 physical 30-second Fpz-Cz epochs at 100 Hz. There were 55,153 scored fit and 23,872 scored development epochs. The original recording positions and unscored epochs remained in a private cache; only scored epochs entered stage training or metrics. Fit-only normalization and fit-only prototype k-means were enforced. No original SC VAL, RESERVED, ST, SHHS, HMC, ISRUC or DREAMS waveform was opened.

The five arms shared a fixed 1,200-update, batch-16 schedule at seeds 123 and 456. All arms saw each original and bounded transformed view in stage cross-entropy. M0–M3 had identical 41,429-parameter morphology models, sampler/view streams, prototype regularization and fit-only 4,096-token k-means input identities within each seed. M1 added SupCon, M2 added MPI, and M3 added both. A0 was a 522,805-parameter **adapted** AttnSleep reference. The final update-1,200 checkpoint was selected by rule, not by development score.

| Arm | Clean participant Macro-F1, seed 123 / 456 | Bounded-nuisance participant Macro-F1, seed 123 / 456 | Clean N1 recall, mean | Clean REM recall, mean |
|---|---:|---:|---:|---:|
| A0 adapted AttnSleep | 0.6889 / 0.6534 | 0.6840 / 0.6520 | 0.6444 | 0.7650 |
| M0 stage + ordinary augmentation | 0.6902 / 0.7268 | 0.6905 / 0.7226 | 0.6623 | 0.6239 |
| M1 M0 + SupCon | 0.6892 / 0.6842 | 0.6867 / 0.6844 | 0.7447 | 0.4850 |
| M2 M0 + MPI | 0.6896 / 0.7349 | 0.6910 / 0.7338 | 0.6444 | 0.6426 |
| M3 M0 + MPI + SupCon | 0.6937 / 0.6845 | 0.6934 / 0.6841 | 0.7251 | 0.4964 |

The primary **M2–M0** participant Macro-F1 differences were −0.00067 and +0.00806 on clean development, and +0.00046 and +0.01123 under bounded nuisance, in seed order. Five-person paired bootstrap 95% intervals for the bounded contrast were [−0.00112, +0.00221] and [+0.00561, +0.01686], respectively. These intervals describe each reused development sample conditional on its model seed; they are not independent confirmation or a ten-person trial. The two-seed mean bounded gain (+0.00585) is below the +0.02 gate. M2–M1 clean mean was +0.02551, but this does not rescue the primary M2–M0 gate. M3–M2 clean mean was −0.02308. SupCon-only and combined arms substantially increased N1 recall while reducing REM recall, especially in seed 456; their Macro-F1 was lower than M2 in that seed.

## Mechanism and failure checks

The bounded view retained the exact physical token grid. A 100-example fit-only transformation audit found finite waveforms, an RMS ratio range of 0.950–1.089, 11–16 Hz power ratios 0.903–1.120 and 0.5–4 Hz ratios 0.899–1.101. There was no time shift, filter or clipping operation; filter ringing and shift realignment are therefore inapplicable to the frozen family. Generated 13 Hz burst and 0.75 Hz slow-component projections changed by factors 1.023 and 1.032 without coordinate shifts. On the same 100 fit examples, the authenticated historical P3B classifier retained 98% prediction agreement with mean absolute class-probability change 0.00441. That classifier is a descriptive response probe, **not** an oracle of clinical stage preservation. Clinical stage preservation remains unverified.

M2 lowered global embedding L2 drift relative to M0 from 0.0371 to 0.0277 in seed 123 and 0.0257 to 0.0212 in seed 456. Prototype-assignment KL fell from 0.00407 to 0.00297 and from 0.00228 to 0.00201. Prediction agreement changed only from 0.9829 to 0.9843 and from 0.9893 to 0.9912. The separate MPI terms each produced nonzero encoder and prototype-bank gradients on a fixed fit-only diagnostic; lower loss was therefore not caused by complete gradient detachment. These improvements still did not establish useful robustness efficacy.

The final M2 bank used an effective 59.2 and 57.5 of 64 prototypes in the two seeds, with zero dead prototypes under the prespecified <0.001-usage definition. It had 49 and 36 near-duplicate prototype pairs, so the bank was not perfectly diverse. Development embedding feature variance was 0.00503 and 0.00478, above the observed constant-representation case. A fit-only 5,000-example linear probe, evaluated on 2,000 disjoint-development examples, scored participant Macro-F1 0.7884 / 0.7895 for M2 versus 0.7923 / 0.7885 for M0. This does not show a consistent stage-information improvement. No prototype was labeled a clinical spindle, K-complex or slow wave.

The intentionally destructive stress view reduced every arm's Macro-F1 to roughly 0.16–0.21. M2 averaged 0.1971, effectively equal to M0's 0.1972, and REM recall was nearly zero. This is an artificial failure control and is not a measured clinical shift or external negative transfer.

## Compute and reproducibility

Ten trajectories consumed exactly 12,000 scientific optimizer updates and 384,000 training view exposures (192,000 original plus 192,000 transformed). Engineering probes consumed 65 updates. The full campaign took 95.44 seconds of observed GPU wall time, with 90.31 seconds summed trajectory time; peak allocated GPU memory was 91.6 MB and peak process RSS about 3.11 GiB. The A0/M0/M1/M2/M3 mean trajectory times were approximately 10.73/8.13/8.67/8.54/9.09 seconds. M2 thus cost about 5% more training time than M0 in this small, in-memory run, while sharing the same inference architecture. An in-memory generated-input batch benchmark measured about 76,300 A0 and 124,300 morphology epochs per second at batch 64; these figures omit real source I/O and are not end-to-end clinical throughput.

The CPU regression suite passed after the new core tests. A generated CUDA exact-resume test initially failed when nondeterministic GPU algorithms were allowed; it passed bitwise with deterministic cuDNN and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The completed ten scientific trajectories were uninterrupted and did **not** run under that enforced deterministic-resume mode. Their final checkpoints are byte-digest linked, but a bitwise interrupted-resume guarantee for the completed runs is unresolved. This is a methodological limitation, not grounds to rerun outcomes for a better result.

## Claim boundary

The result supports technical trainability, fit-only prototype initialization and a small reduction in bounded-view representation drift. It does not satisfy the frozen nuisance Macro-F1 readiness threshold, establish external transfer, prove stage-preserving physiology, validate clinical prototype semantics, demonstrate label efficiency, or establish methodological novelty. The original proposal's K sensitivity, EMA, single-term MPI fits, capacity-matched AttnSleep, DA and foundation-model comparisons remain untested. A further external study requires separate data authorization and scientific review of the present NO-GO outcome.

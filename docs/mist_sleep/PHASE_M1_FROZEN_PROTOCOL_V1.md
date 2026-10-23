# Phase M1 frozen independent HMC protocol

The machine-readable registration is `reports/mist_sleep/phase_m1_hmc_v1/frozen_protocol.json`.
It is committed before any HMC waveform header, waveform value or scored annotation is opened.
The user explicitly authorized this independent falsification study despite the unchanged Phase M0
decision `MPI_FEASIBILITY_NOT_ESTABLISHED`. No new training or target adaptation is authorized.

Only M0/M2, seeds 123/456, final step 1,200 may run. Their four actual checkpoint byte
identities and 41,429-parameter states have been authenticated with the inherited restricted
loader under PyTorch 2.10 CPU. The source mean 0.2903136955214651 uV and standard deviation
26.31764815403481 uV come from the authenticated ten-person source-fit receipt.

The primary channel is C4/M1; F4/M1 is a separately labeled secondary sensitivity.
Calibration uses original EDF physical scaling. Fixed 256-to-100 Hz polyphase resampling
uses an explicit symmetric 1,281-tap Kaiser-beta-5 FIR (cutoff 1/64 at the upsampled rate),
float64 arithmetic and constant-zero padding only at physical recording ends. Segment the
continuous result into complete 3,000-sample epochs at t=0,30,60,... seconds. Preserve polarity,
source normalization and every complete physical epoch. No target moment fitting, clipping,
additional bandpass, label-based trimming or synthetic montage reconstruction is permitted.

The primary estimand equally weights recording-level fixed-five Macro-F1, with zero F1 for
absent classes, and averages the paired M2-minus-M0 effects over the two fixed seeds.
Resample complete paired recording clusters 10,000 times with NumPy seed 20261023 and report
the percentile 95% interval. Both seeds stay together within each bootstrap unit. Anonymous
recording IDs do not establish participant independence. Only independently verified linkage
available before scoring may introduce participant clusters; the equal-recording estimand stays
unchanged. No participant mapping is currently available in the aggregate-only metadata.

Success requires gain >=0.02, positive effects in both seeds, lower CI >0, authenticated and
reproduced source behavior, correct preprocessing, sealed predictions before labels, and no
adaptation or post-outcome selection. Report any >=95% dominant-class prediction as
`SEVERE_CROSS_MONTAGE_FAILURE` alongside the original endpoint. Secondary representation
stability cannot rescue a failed primary result.

Blind prediction must include both channels, all four models and all eligible records before
the public hash receipt is committed. Label access claims are exclusive and persisted before
reading annotations; failed/interrupted scoring remains exposed. Generated CSV fixtures use
`onset_seconds,duration_seconds,annotation`; real HMC text schema is not verified and any
unsupported format must block agreement checking rather than be guessed. Unknown stage
texts, overlaps, fractional epoch shifts and contradictory scoring geometries fail closed.

This tests a combined clinical, acquisition and montage shift from SC Fpz-Cz to HMC C4/M1.
Resampling does not correct montage. HMC providers clipped to lights-off/on, unlike SC's
complete physical time grid. A single-target result cannot isolate particular shift causes,
establish clinical suitability, label efficiency or broad multiple-target generalization.

The current acquisition gate fails: approximately 7.1 GB persistent space is available for
a 15.7 GB collection, before the 2 GB safety reserve. No bulk acquisition is launched.
The final scientific outcome will remain blocked unless all gates can actually be satisfied.

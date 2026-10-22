# Phase M0 analysis lock before nuisance evaluation

The five-arm, two-seed scientific training has completed, and clean-development summaries have been inspected. The training intervention, arms, seeds, epochs, checkpoints and final-state selection were sealed before any scientific result in `frozen_protocol.json`. This analysis lock is **after clean results and before nuisance results**; it is not claimed to be an independent preregistration.

Each final 1,200-update state will score every scored epoch of the five-person internal-development partition under (1) unchanged input, (2) one deterministic draw of the exact training `paired_view` policy with seed 2026, and (3) the separately implemented `stress_view` with seed 2027. The stress view is deliberately destructive and is not a clinical label-preserving test. Every arm receives the same waveform views at the same physical coordinates. No target data or other external cohort is used.

The descriptive endpoints are fixed-five participant Macro-F1, pooled Macro-F1, N1 and REM recall, prediction agreement, confidence change and, for M0–M3, global embedding cosine/L2 drift, prototype KL, token assignment switching, effective prototype occupancy and stage-centroid separability. Primary nuisance comparison remains M2–M0. Both individual seeds and their two-seed mean will be shown. A participant bootstrap over the five development people describes sampling uncertainty; it does not repair repeated developmental use.

The existing implementation and the model checkpoints are immutable for this analysis. Any post-lock correction will be labeled as such and will not be used to select a favorable outcome.

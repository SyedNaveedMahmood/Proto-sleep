# Phase VIII representation and interference analysis v1

Frozen P0 diagnostic-development results; not locked confirmatory outcomes. Means below aggregate six optimization/task cases over matched 100-example draws. No human sampling inference.

| Condition, strength1 | Conventional F1 | Privileged local linear probe F1 | Trimmed unchanged-head F1 | Oracle unchanged-head F1 | Diagnostic candidate drift |
|---|---:|---:|---:|---:|---:|
| high_frequency | 0.938668 | 0.780270 | 0.484239 | 0.761438 | 2.125995 |
| low_frequency | 0.779572 | 0.713493 | 0.592279 | 0.716639 | 2.836432 |
| transient | 0.568348 | 0.697197 | 0.607862 | 0.682117 | 2.854176 |
| impulse | 0.850385 | 0.780817 | 0.616141 | 0.737372 | 1.439030 |
| broadband | 0.626572 | 0.747850 | 0.678498 | 0.697349 | 2.882636 |
| sustained | 0.914034 | 0.749128 | 0.393049 | 0.733804 | 8.467738 |
| multiple | 0.802762 | 0.687716 | 0.306983 | 0.704973 | 6.778521 |
| correlated | 0.837309 | 0.704840 | 0.312944 | 0.706831 | 6.935972 |
| overlapping | 0.798636 | 0.569096 | 0.321015 | 0.671531 | 9.180307 |
| disjoint | 0.959840 | 0.824268 | 0.510705 | 0.776002 | 0.818822 |
| boundary | 0.956446 | 0.803836 | 0.518055 | 0.760892 | 1.035547 |
| inside_candidate | 0.941370 | 0.783305 | 0.461281 | 0.768530 | 2.202391 |
| crossing_candidate | 0.943195 | 0.790898 | 0.467175 | 0.765552 | 2.229247 |
| weak_diagnostic | 0.607241 | 0.425250 | 0.185190 | 0.569352 | 12.885155 |
| competing | 0.802762 | 0.687716 | 0.306983 | 0.704973 | 6.778521 |

Abrupt transients and broadband bursts produce severe changes; diagnostic-overlapping versus disjoint nuisance also differs. Diagnostics and nuisance can share a patch; diagnostic-containing drift is not a uniquely diagnostic contribution. The fixed linear local probe uses only privileged TRAIN regions and less exposure/capacity than Phase VII nonlinear probes. Its weak scores here do not contradict Phase VII’s differently specified preservation finding. No universal diagnostic-information-preservation claim passes from these results.

Feature-to-logit and two-order path arithmetic reconcile below 1e-5 (largest observed residual 7.62939453125e-06). Signed per-feature mean deltas are retained in safe aggregate diagnostics. Two substitution orders report interactions; neither provides unique encoder/pooling/head waveform causality. Mean/max/combined/trimmed/bounded/oracle contrasts use the same frozen features and unchanged weights; poor alternative-pooling performance can reflect head-distribution mismatch. T1 is subsequently trained under a matched exposure protocol to test that limitation.

Candidate norms, per-set feature drift, large absolute activation fractions, within/between class separation and eventless feature norms are recorded. Large absolute values are amplitude sensitivity indicators, not proof of a bounded nonlinear unit saturating. Native conventional event scores are zero; candidate deletion rankings are model-internal. Spectral sensitivity here compares generated frequency interventions, not a complete transfer-function characterization.

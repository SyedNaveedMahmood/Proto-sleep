# Phase XIV: paired ST condition descriptions

The authenticated official subject spreadsheet identifies the placebo and temazepam nights for all 22 ST people. Condition was assigned from those explicit columns, never inferred from filenames. All reported contrasts are temazepam minus placebo within the same person; 1,000 bootstrap draws resample people and retain the two nights together. No model or detector tuning used conditions.

| Paired participant mean contrast | Estimate | Bootstrap 95% interval |
|---|---:|---:|
| Classifier accuracy | +0.0426 | [+0.0224, +0.0639] |
| Classifier macro-F1 | +0.0252 | [−0.0028, +0.0512] |
| N1 recall | +0.0450 | [+0.0060, +0.0777] |
| Mean confidence | +0.0085 | [−0.0018, +0.0187] |
| Fpz-Cz spindle-like candidates per eligible hour | +68.54 | [+56.72, +82.73] |
| Fpz-Cz slow-wave-like candidates per eligible hour | −24.60 | [−42.45, −6.53] |
| EEG centered RMS (µV) | −4.40 | [−6.51, −2.65] |

The paired panel includes prevalence, class F1, spectral, EOG, EMG and artifact contrasts in `st_condition_aggregates.json`. The 219-case witness panel pairs first eligible stage examples across nights where both exist; missing strata stay missing. Its diagram complexity, compactness, local/global dependency and proxy-overlap contrasts are in `st_condition_witness_aggregates.json`. Pairing improves control of stable person differences, but recording order, pharmacology, night effects, stage composition and detector threshold sensitivity remain confounded. These are many descriptive endpoints without a multiplicity adjusted causal analysis. **No drug-effect claim is established.**

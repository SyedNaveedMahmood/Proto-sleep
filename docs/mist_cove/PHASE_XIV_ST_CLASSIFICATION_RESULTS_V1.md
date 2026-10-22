# Phase XIV: frozen ST classification

The prospectively sealed evaluation covers 44 recordings from 22 Sleep-Telemetry people, in two nights per person. The independent official checksum list, original condition spreadsheet, EDF headers, annotation geometry, model byte identity, frozen normalization and analysis code were checked before ST waveform evaluation. All 45,446 physical epochs were inferred in order. The 2,755 unscored epochs remain in the physical grid and do not enter supervised metrics. No ST fit, normalization update, threshold choice or checkpoint selection occurred.

| Outcome | SC TRAIN, previously seen study | ST frozen cross-study |
|---|---:|---:|
| Scored epochs | 79,025 | 42,691 |
| Accuracy | 0.9477 | 0.6987 |
| Pooled macro-F1 | 0.8246 | 0.5839 |
| Mean participant macro-F1 | 0.8160 | 0.5583 |
| N1 recall | 0.5044 | 0.2809 |
| N2 recall | 0.9207 | 0.9058 |
| N3 recall | 0.8690 | 0.7790 |
| REM recall | 0.7914 | 0.2162 |
| ECE, 15 bins | 0.0030 | 0.1339 |

The preregistered ST transfer gate **PASSES** its minimum: participant macro-F1 at least 0.50, N1 recall at least 0.20, every class F1 at least 0.10, five predicted classes and no class above 98% of predictions. This minimum gate does not negate the severe REM recognition loss. On ST, REM is often called Wake, N1 or N2. N2 recall remains close to the SC value while N1, N3 and especially REM deteriorate. The ST participant mean F1 bootstrap 95% interval is [0.5328, 0.5842], with people as the resampling unit.

The frozen model is less calibrated on ST: negative log likelihood 0.9149 versus SC 0.1376; Brier 0.4338 versus 0.0739; ECE 0.1339 versus 0.0030. Wrong predictions with confidence at least the fixed high-confidence threshold are 3,209/42,691 ST scored epochs versus 433/79,025 SC scored epochs. The six SC-trained context comparator states were also evaluated at their fixed final steps; their SC promotion gate had already failed, and ST outcomes cannot promote one. Full matrices, per-class F1, transitions, all six comparator metrics and error-calibration summaries are in `st_classification_metrics.json`.

This is a cross-study evaluation with acquisition, medication, selection and stage-prevalence differences. SC TRAIN is in-sample development. These two studies do not establish independent clinical-population generalization.

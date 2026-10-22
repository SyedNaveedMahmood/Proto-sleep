# Phase XIV aggregate dashboard

The complete [scientific decision](../../../docs/mist_cove/PHASE_XIV_SCIENTIFIC_DECISION_V1.md) answers the 30 preregistered questions. All charts below use group aggregates. The separate waveform atlas is private and labeled automated proxies only.

| Gate | Observed result | Decision |
|---|---|---|
| Source integrity | 73/73 PSG/hypnogram pairs, 146/146 official file hashes | PASS |
| SC development proxy versus matched random | 9 eligible cases, 4 people; required at least 20/5 | NOT ESTIMABLE |
| ST proxy versus matched random | +0.1168, 95% participant interval [0.0466, 0.2121], 25 cases/12 people | PASS |
| ST proxy versus sigma ranking | −0.0647 [−0.1312, −0.0124] | FAIL |
| ST original classifier transfer minimum | Mean person macro-F1 0.5583; N1 recall 0.2809 | PASS, with marked REM/calibration loss |
| SC context C1 versus C0 | Transition accuracy declines in all 3 seeds | FAIL |
| Exact finite circuits | 900 SC + 1,095 ST 15-atom tables; 60 context 17-variable tables | PASS |
| Human expert event validity / clinical deletion | No approved reviewers | NOT TESTED |

SC TRAIN includes 79,984 physical and 79,025 scored epochs. ST includes 45,446 physical and 42,691 scored. Frozen ST accuracy is 0.6987, pooled macro-F1 0.5839, N1 recall 0.2809 and REM recall 0.2162. SC in-sample accuracy is 0.9477 and pooled macro-F1 0.8246. ST ECE15 rises to 0.1339 from 0.0030. The six context fits total 18,000 updates; zero ST training updates occurred.

| Aggregate figure | What it shows |
|---|---|
| [Channels and signal scale](figures/01_channels_and_signal_scale.png) | Verified native channels and stage-balanced signal amplitudes |
| [Event rates](figures/02_spindle_and_slow_wave_rates.png) | Person-aware spindle/slow-wave candidate rates |
| [Cross-channel agreement](figures/03_cross_channel_and_detector_sensitivity.png) | Fpz-Cz/Pz-Oz concordance and detector sensitivity |
| [EOG, EMG and quality](figures/04_EOG_EMG_and_quality.png) | Signal and artifact proxy rates; differing EMG acquisition |
| [Matched interval controls](figures/05_proxy_overlap_and_matched_controls.png) | Exact support versus random, amplitude, spectral and gradient controls |
| [Structure and stability](figures/06_structure_and_explanation_stability.png) | Compactness and prediction versus clause validity |
| [Stage confusion](figures/07_SC_ST_stage_confusions.png) | Complete scored-grid SC/ST classification errors |
| [Calibration and prevalence](figures/08_calibration_and_prevalence.png) | Frozen confidence shift and scored-stage composition |
| [Conditions and neutral state](figures/09_paired_conditions_and_neutral_sensitivity.png) | Descriptive within-person ST conditions and operator sensitivity |
| [Context and compute](figures/10_context_and_compute.png) | All three fixed optimization seeds and fit cost |
| [Graph complexity](figures/11_graph_complexity_and_clause_coverage.png) | BDD size and displayed clause coverage |
| [Class errors and transitions](figures/12_stage_errors_and_context_transitions.png) | N1/REM/N2/N3 loss and failed context transition gate |

Exact denominators and participant bootstrap intervals are in the JSON outputs. These automated proxy plots are not clinical event truth. The latest-tree privacy scan and historical path-exposure disclosure are recorded in `publication_manifest.json` and the scientific decision.

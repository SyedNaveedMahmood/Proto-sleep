# Phase XV research dashboard

**Status:** `EXPLORATORY_REUSED_ST`; 0 neural fits, 0 optimizer updates, 0 ST adaptations, 0 new SC VAL/RESERVED reads, 0 external EEG transfers. Human event validation: **NOT TESTED**. Seven [aggregate figures](figures/) and the [complete research report](../../../docs/mist_cove/PHASE_XV_RESEARCH_REPORT_V1.md) give the context behind these figures.

| Question | Denominator and result | Decision |
|---|---|---|
| Original ST REM failure | 1,805/8,349 correct; 2,322→W, 1,570→N1, 2,648→N2, 4→N3; 22 people/44 nights | Serious REM transfer failure; no new model fit |
| Distinct REM-error correlates | REM→N2 frontal sigma +0.018 and transition proximity +0.165 vs correct REM by person; REM→W EOG excursions +5.61 | Exploratory associations, not causes or label adjudication |
| Confidence reliability | Pooled ECE15 0.133863; REM ECE15 0.453; 3,209 high-confidence wrong scored epochs | Fixed simple warning rules have limited sensitivity |
| Exact circuit correctness | 360 cases × 32,768 base masks and 360 neutral tables; full-input max error 0; authenticated state unchanged | `ENGINEERING_PASS` for declared internal operator |
| Proxy eligibility | E1 219 total, 63 with Pz-Oz candidate, 55 with nondegenerate positive support (20 people); E3 50 eligible/240 | Missingness and conditioning explicit |
| Exact vs sigma | E2 mean interval precision 0.171 vs 0.217; participant exact−sigma −0.0462, bootstrap95 [−0.0797,−0.0167] | Prior registered sigma-increment **FAIL unchanged**; no new confirmatory gate |
| Alternative comparators | Exact−RMS +0.107, exact−random +0.138 participant mean on E2 | Does not overcome sigma comparison |
| Wrong predictions | 18/55 E2 eligible cases had wrong stage prediction | Faithfulness can explain a wrong decision |
| Neutral/operator dependence | 190/219 E1 tables changed with neutral .01; inherited compatible bridge 500/3,285 requests | Interpret clauses only under named operator |
| Prediction vs clause stability | Mild HF: 93/100 prediction-stable, 70/100 same-clause-valid, 3 substitutions | Prediction invariance is weaker than explanation invariance |
| Resource ceiling | 33,426,540/38,000,000 charged cached-head states; measured principal stages ≈340 s CPU | Budget respected; cached-head states are not neural training exposures |

The original Phase XIV ST exact-minus-random gate passed and exact-minus-sigma gate failed. Phase XV does not promote a failed scientific gate. The SC developmental proxy primary remains `NOT_ESTIMABLE` (9 eligible cases/4 people), Phase XIV SC context gate remains `FAIL`, and expert event validity remains `NOT_TESTED`.

The recommended next decision is an independently annotated, blinded event study with a prespecified equal-duration sigma baseline, or a separately authorized classifier-development protocol aimed specifically at REM transfer. [Phase XVI is a proposal only](../../../docs/mist_cove/PHASE_XVI_PROSPECTIVE_STUDY_PROPOSAL_V1.md).

# Transfer protocol and gates

A: EDF20 vs participant-disjoint EDF78 is same-SC-cohort generalization.
All overlapping people and nights are excluded from the extension, regardless
of filename or byte equality. It is never independent cross-dataset transfer.

B: SC -> ST is an exploratory cross-study proxy, confounded by population,
setting and placebo/temazepam. Treatment metadata must be verified. Freeze the
policy before scores are viewed; any ST use for selection makes ST development.

C: SHHS is optional until authorized, paired usable data exist. Only a frozen,
participant-disjoint external cohort can support the external transfer claim.
No target-test labels or unlabeled target data for fitting, checkpoint selection,
normalization, anchors, cleaning, augmentations or design. Transductive access
would require a separately declared protocol. No evaluation-label sleep trimming.

The P1 inventory examines annotations for integrity, not model selection. It
creates eligibility sets, not train/validation/test assignments. Explicit split
manifests must reject subject, recording and signal overlap before data opening;
reserved test data have no development-opening path. Unknown indices permit
only independent epochs. Scientific errors block downstream runs. An observed
EDF20/78 intersection is expected inventory evidence; assigning that intersection
to different evaluation roles is the hard-stop leakage violation.

Stop after P1. P2 may only start after review of audit blockers; no P3 training,
architecture redesign, morphology diagnoses, novelty or SOTA claims are made.

The authorized P1.5 follow-up characterizes blockers without overriding the
historical P1 FAIL. Its revised gate applies to the 74-person/146-recording raw
SC subset, excluding all nights of people 06, 23, 36 and 74. These exclusions
must be declared before experiments and their effects on cohort composition
reported. Within-cohort experiments may use the eligible EDF-20 membership
(19 people/37 recordings) and a disjoint extension (55 people/109 recordings);
their observations remain within the SC cohort. No split has been assigned.

The measured NPZ masks match a label-conditioned selection rule, but the
upstream executable and causal selection policy remain UNKNOWN. Existing
trimmed NPZs cannot provide the zero-shot endpoint. Rebuild inference inputs
from every complete physical raw epoch before accessing stage labels; retain
unscored time positions and attach metric masks afterwards. Use source-TRAIN
normalization and source-only model selection. This is a specification for P2,
not an executed preprocessing pipeline.

ST annotations are verified, including seven strictly validated read-only
Recordingfield compatibility cases. ST preprocessing remains CONDITIONAL on
separate fixed-clock policy review and P2 decoder/sample checks. Keep treatment
and participant-night grouping; freeze development versus evaluation roles
before target scores. SC/ST person linkage is UNKNOWN, so this remains an
exploratory cross-study proxy. SHHS access/usability is UNKNOWN and P2 BLOCKED.
See `reports/data_audit/p1_5/gate_decisions.json` for scope-specific evidence.
Stop after P1.5 for review; no scope is authorized to run P2 in this session.

The subsequent authorized P2-SC phase implements only the reviewed raw SC
subset and passes its data gate: 74 subjects/146 recordings, all complete
physical epochs retained with scoring masks. Earlier stop statements describe
their respective historical sessions. P3 source-only baseline development is
data-eligible after review; freeze explicit subject-grouped source roles and
source-TRAIN-only fitting before training. No final test split was chosen in
P2. Stop after P2-SC; ST/SHHS decisions remain separate and unchanged.

## Frozen P3 source-only SC reference protocol (P3A)

The subsequent P3A authorization permits development integration only on the
verified raw SC subset. Before viewing any model scores, seed 123 and NumPy
default_rng partition the sorted 19 eligible historical EDF-20 people into
15 TRAIN (29 recordings) and 4 VAL (8 recordings). VAL is SC:02, SC:08, SC:11,
SC:19. TRAIN is SC:00,01,03,04,05,07,09,10,12,13,14,15,16,17,18.
Reserve all 55 disjoint extension people/109 recordings for eventual SC
subject-generalization evaluation. Keeping historical people for development
and the larger extension untouched makes the exact roles prespecified, without
stratifying on reserved labels. The strata differ in recruitment composition;
this is within-SC generalization, never independent EDF-20/78 external transfer.

Frozen split fingerprint:
`3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84`.
Frozen file SHA256:
`1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5`.
See `experiments/splits/p3a_sc_source_v1.json` and its separate receipt. The
freeze reads only scalar identity prefixes from trusted P2 descriptors, stopping
before annotation sections; development loading opens TRAIN/VAL descriptors
only. It rejects duplicate participants across roles, nights, recordings, paths,
raw-file, exact-signal and normalized-shape source hashes before data access.
No development test-opening API exists. Roles cannot change under resume.

The matched reference comparison is epoch-only (radius zero, no CRF), with
repository AttnSleep, evidence, summary_only and unrestricted local raw_context
adapters sharing P2 input geometry/masks and TRAIN-only scaling. The proven
MIST-Evidence trainer, waveform medoids and atomic checkpoint/RNG machinery are
reused. No MCR or new research losses are added. Primary selection is the mean
of participant fixed-five Macro-F1 after pooling each participant's nights;
earliest best wins ties. Pooled Macro-F1 is descriptive. Undefined class F1 is
zero; absent-class recall/kappa and balanced-accuracy conventions are explicit.
VAL must not fit normalization, anchors, augmentation or calibration. All
reserved test inputs and annotations remain inaccessible to these loaders.

`configs/p3a_sc_smoke.json` fixes one bounded development epoch for all four
controls, with geometry-only windows, FP32 and deterministic CUDA. Smoke scores
cannot choose model designs or tune the proposed matched budget.
`configs/p3b_sc_matched_proposed.json` records the 60-epoch/12-patience, shared
AdamW budget and proposed seeds before scores, for review only. Full matched
training, reserved evaluation, ST preprocessing, SHHS access and MCR remain
outside P3A authorization. AMP is disabled until separately validated. The
optional conservative source gain adapter is not part of the core four-way
CUDA smoke. See `docs/P3A_RUNNER.md` for streaming, resume and provenance rules.
Stop after P3A for review; the SC gate does not approve ST or SHHS.

## P3B authorized full-data source reference training

The subsequent P3B authorization preserves the exact P3A split, SC eligibility,
channel, stage mapping, physical grid, scoring masks, radius zero/no-CRF and
fixed-five metrics. Development remains 15 TRAIN/4 VAL people only; all 55
reserved participants, ST and SHHS remain inaccessible. Do not regenerate the
split or inspect reserved descriptors in this phase.

The reviewed proposed budget is now frozen in `configs/p3b_sc_matched_v1.json`:
AttnSleep/evidence/summary_only/raw_context, seeds 123/456/789, AdamW lr 0.0003,
weight decay 0.0001, maximum 60 epochs, patience 12, core/encoding batch 16,
gradient clip 5, deterministic FP32. Full physical TRAIN/VAL counts are
79,984/21,796; scored loss/metric counts are 79,025/21,789. All controls share
the same maximum budget and stopping/selection policy; report realized epochs
and updates rather than claiming equal realized compute. Preserve negative
model results/class collapse without recipe changes.

Reuse verified full-TRAIN normalization. Before full scores, freeze a real
TRAIN waveform-medoid bank per seed using the existing recipe, with the same
bank across the seed's four controls. No augmentation, additional losses or
architecture changes are authorized. The existing P3A CLI stays smoke-only;
`scripts/run_transfer_p3b.py` is a separate explicitly authorized full-data path.

Gate A requires synthetic/regression success, exact interrupted CUDA resume,
full-grid identity/mask checks, and one full TRAIN epoch plus VAL pass per model
at seed 123. Initial epochs pause with complete optimizer/RNG state under the
unchanged 60-epoch configuration, without completion markers. The prespecified
practicality ceiling is a measured maximum-budget estimate of 24 GPU hours.
Gate B may resume only after correctness/capacity PASS, running all 12 model/seed
combinations sequentially. Freeze/commit code before benchmarking; any later
correctness fix requires a new versioned identity. All terminal output and
checkpoint/metric provenance must be retained. Never claim completion without
validated output markers. See `docs/P3B_RUNNER.md` for execution/resource rules.

Stop after P3B. This phase supplies SC development reference models, without
reserved SC evaluation or genuine external-transfer evidence. Four VAL people
limit inference about performance variability; seeds are not new participants.


## Completed P3B source-only reference runs

All 12 model/seed runs have validated completion markers. The frozen implementation is 661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23; run fingerprint a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b. Source roles, split, preprocessing, five-stage mapping and selection policy were unchanged. No reserved SC waveforms/annotations, ST or SHHS were accessed; no MCR or augmentation was trained.

| Model | Participant Macro-F1 mean ± seed SD | Pooled Macro-F1 mean ± seed SD |
|---|---:|---:|
| attnsleep | 0.312650 ± 0.007300 | 0.335615 ± 0.010289 |
| evidence | 0.718031 ± 0.016188 | 0.738142 ± 0.015252 |
| summary_only | 0.512157 ± 0.000532 | 0.515835 ± 0.000254 |
| raw_context | 0.723511 ± 0.005192 | 0.740560 ± 0.004202 |

These are checkpoint-selection source-VAL scores from four participants, pooling nights within participant. Seeds are repeated optimization/anchor variation, not independent people. Maximum budgets matched, while realized early-stop epochs/updates differed. AttnSleep was weak; its selected seed-789 model predicted no N3 or REM. Summary-only mean N1 F1 was 0.006557. These negative results were retained without changing the recipe. This is the frozen repository-baseline policy and does not establish published AttnSleep accuracy, clinical morphology, novelty, SOTA, reserved performance or independent external transfer. EDF-20/78 overlap.

Gate A passed with a maximum-budget estimate of 11.297 GPU hours against the prespecified 24-hour ceiling. Gate B completed in 5:56:13; all runs used deterministic FP32 with AMP/TF32 off. The 433 realized epochs involved 2,155,474 optimizer updates. Focused CPU tests: 40 passed/4 CUDA skipped; CPU regression: 189 passed/5 CUDA skipped; all four required CUDA exact-resume fixtures passed separately. Completed-run resume verified unchanged outputs, and changed learning rate under resume was rejected before data access.

Full per-seed metrics, stage F1/confusions, calibration, participant variability, parameter counts and resource measurements are in [the P3B aggregate report](../reports/p3b/FINAL_COPY_PASTE.txt) and [machine-readable results](../reports/p3b/results.json). Checkpoints, predictions, banks and full console logs remain local/ignored. Stop after P3B for review; additional MCR, reserved-cohort or external-study work requires separate authorization and applicable data gates.

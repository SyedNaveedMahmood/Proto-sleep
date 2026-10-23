# MIST-Sleep Phase M1 dashboard

**Scientific decision: `PRIMARY_EVALUATION_BLOCKED`. Research recommendation: BLOCKED.**

The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`.

- Four source checkpoints authenticated. Original full-input forward behavior reproduced on generated inputs.
- Official v1.1 manifest: 151 recordings, 453 source files; 151 actual headers audited.
- C4/M1 and F4/M1 electrode pairs present in all headers at 256 Hz/uV.
- Persistent disk: 7.06 GB free, 17.7 GB estimated need with reserve. No bulk acquisition.
- Target waveforms, labels, predictions and optimizer updates: zero. Eligible cohort and all clinical metrics: unknown/null.
- CPU regression: 1246 passed, 13 skipped; M1 generated GPU inference-only correctness: 1 passed.
- Forty work packages have individual status, implementation, tests, evidence, finding, limitation, resource and blocker records in progress.json.
- Four figures show actual metadata/capacity and explicitly generated signal/cost validation. Clinical result figures were not fabricated.

See [scientific decision](../../../docs/mist_sleep/PHASE_M1_SCIENTIFIC_DECISION_V1.md), [source audit](../../../docs/mist_sleep/PHASE_M1_HMC_SOURCE_AND_LICENSE_V1.md), [registered protocol](../../../docs/mist_sleep/PHASE_M1_FROZEN_PROTOCOL_V1.md) and [verification](verification.json).

Initial freeze: 933bcba. Header-only label syntax refinement: d7947ef; primary electrode pair unchanged. Original strict-slash audit preserved. Whole EDF checksums and real HMC text schema remain unverified. No external result, clinical deployment, label efficiency or multiple-target generalization claim is supported. Stop after M1; do not merge automatically.

Draft delivery: [PR #20](https://github.com/SyedNaveedMahmood/Proto-sleep/pull/20). Exact final-head CI is verified through its GitHub checks after the last publication commit; local receipts cannot contain their own final commit SHA.

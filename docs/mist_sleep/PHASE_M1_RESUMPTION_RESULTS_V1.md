# MIST-Sleep M1 resumption v1: stopped on source transport failure

**Scientific decision: `PRIMARY_EVALUATION_BLOCKED`. Storage gate: PASS.**

The configured private HMC directory is on the root SSD. Its resumption preflight
found **26,015,137,792 free bytes**, exceeding the
unchanged 15,700,000,000-byte estimate plus 2,000,000,000-byte reserve. Permissions
remain 0700. All four exact source checkpoints, source-fit normalization and frozen
protocol/readiness identities passed before acquisition.

Acquisition from the official HMC v1.1 mirror retained **8 complete recording
triplets / 24 files**, totaling **908,992,759 bytes**.
Every retained complete file passes its original official SHA256, including an
independent post-stop checksum audit. One **83,886,080-byte
partial** remains unauthenticated and unusable for evaluation. No raw file was
deleted. Available persistent space after stopping is
**24,992,694,272 bytes**; storage remains sufficient.
Whole-cohort integrity, real scoring-schema compatibility and scoring alignment
remain unverified. Eight source triplets do not establish a scored eligible cohort.

Two intentionally interrupted slow streams retained their verified files and
partials before any integrity failure. The transport was changed to curl, then
bounded 16 MB ranges with at most four prefetched responses. The original source
allowlist, full-file SHA256, byte ceiling and disk-reserve checks remained active.
All scientific inference/scoring source files and protocol choices are byte
identical to the original M1 implementation. Small opaque transport probes are
retained privately and accounted for.

The bounded transport subsequently raised `OSError` with the exact diagnostic
`official curl source range truncated or transport failed`. The failed stage lasted
142.81 seconds and used 0.596
Python process CPU seconds. A 120-second per-range wall limit was active, but curl's
exit code and detailed stderr were not retained. The evidence therefore does not
distinguish a timeout from remote truncation/another nonzero curl exit. It does
establish a failed response-body/transport gate. **No automatic retry followed.**

The persisted stop receipt prevents additional automatic acquisition, inference
or scoring for this attempt. No target waveform value was decoded, no model
prediction was produced or sealed, and no original annotation was semantically
opened. Thus M0/M2 external Macro-F1, both seed effects, the paired 95% interval,
stage/calibration/representation analyses and secondary-channel outcomes are all
**unmeasured**. The +0.02 criterion was not evaluated. This provides neither a
positive nor a negative efficacy result for MPI.

The original `phase_m1_hmc_v1` blocked reports and publication hashes remain
unchanged; this version records the newly passed storage gate and subsequent
transport blocker. M0 remains `MPI_FEASIBILITY_NOT_ESTABLISHED`. No scientific
checkpoint, channel, signal conversion, normalization, label rule, metric,
bootstrap or decision threshold was changed. Scientific training, target tuning,
source/target normalizer fitting, RESERVED/VAL waveform reads and other-dataset
access remain zero. No research data was automatically deleted.

Validation: **1255 CPU tests passed / 13
skipped**, including nine generated resumption/transport tests. The generated
tests check historical preservation, stop persistence, source/version allowlists,
range geometry, opaque byte ordering and prefetch budget enforcement. Actual HMC
data is excluded from CI. Exact final-commit CI is verified remotely after the
publication commit and recorded in [draft PR #21](https://github.com/SyedNaveedMahmood/Proto-sleep/pull/21).

Stop after this M1 resumption. Do not merge, repair/retry a failed real-data stage,
start another dataset or begin another research phase automatically.

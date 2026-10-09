# Data integrity contract (P0/P1/P1.5/P2-SC)

The future output contract is 30 s, one EEG, 100 Hz, 3000 samples and
Wake/N1/N2/N3/REM = 0/1/2/3/4. Raw input properties must be measured.
Sleep-EDF R&K 3 and 4 merge into N3; Movement and unknown are excluded with
original positions retained. Annotation gaps are retained, never compressed.

Canonical identity includes study, participant, night and full PSG recording ID.
SC4ssN (observed E/F/G recording variants) and ST7ssN (J) are distinct naming schemes and study namespaces. Filename
identity is checked against paired raw data and public provenance inputs; it
alone cannot certify absence of signal duplication. Reject ambiguous mappings.
Do not unpickle NPZ object headers. Units of normalized NPZ remain unknown until
raw alignment and the conversion are verified. Label-code semantics cannot be
established from integer values alone. A verified sample is not a full-epoch
provenance reconstruction; unknown original indices forbid temporal models.

Default paths are user-reported. CLI/env overrides are explicit and recorded.
SHHS data are not opened without separately confirmed authorization; access
and metadata usability remain optional blockers to an external-cohort claim.

An optional NPZ MANIFEST.TXT may describe a partial download. Missing listed files
or changed listed sizes are integrity errors. Unlisted NPZ files are explicitly
unknown distribution provenance and block acceptance pending review; their
waveforms may still be checked read-only against checksum-verified raw sources.

## P1.5 scope and annotation compatibility

P1 remains an immutable historical FAIL. P1.5 assigns evidence and eligibility
per scope; none of these decisions authorize preprocessing or model training in
this session. A complete run/report marker does not certify every dataset scope.

The annotation-only compatibility path is read-only and narrowly supports one
EDF+C annotation signal, one zero-duration record and complete TAL byte
consumption. It checks declared payload size, separators, numeric onsets and
unsigned durations, stage vocabulary, 30-second grid, non-overlap and source
pairing. Every decoded TAL must agree exactly with MNE 1.10.2; compliant controls
also agree with pyEDFlib 0.1.42. Unknown deviations lead to quarantine.

Seven ST annotations use a preceding-day date in the EDF+ Recordingfield while
the fixed date/time matches their PSG. Only that one-day ST discrepancy may be
accepted, with identical PSG/hypnogram patient and recording fields, canonical
night identity and fixed clock. Preserve the conflicting dates; use the shared
fixed clock plus TAL onset without any 24-hour adjustment or file repair.

A scored TAL covering one physically incomplete final epoch may be recorded as
an explicit diagnostic overhang. Do not create an inference/scoring epoch from
it. Longer scored overhangs remain invalid. This geometric policy does not
consult stage labels when selecting inference inputs.

SC pair demographic conflicts or scored annotation overhangs longer than the
single incomplete physical tail quarantine the affected night and every other
night of its person from the smallest proposed scope. Preserve both header
values and spreadsheet evidence; do not automatically overwrite or adjudicate
identity from age/sex/stage labels. Equipment/program fields of a derived
hypnogram may differ when patient field, canonical ID, clock and core recording
identification agree; record that provenance difference separately.

A partial optional MANIFEST.TXT is not evidence of corruption when all listed
sizes match and every extra NPZ has independently verified raw-source waveform
provenance. Preserve it verbatim as a partial distribution index. Its intended
scope/authoring history remains UNKNOWN; never manufacture missing entries.

Recovered NPZ inclusion masks may exactly match a label-conditioned candidate
rule without proving that code was executed. Compare candidate rules as an
audit diagnostic; retain UNKNOWN upstream lineage. Existing trimmed NPZs are
BLOCKED as zero-shot inference/evaluation inputs. Define new raw evaluation
input epochs from complete signal duration before consulting annotations;
attach labels/scoring masks afterwards. Keep unscored positions, gaps, clocks,
record boundaries, units, montage and source/annotation hashes explicit.

The P1.5 conservative SC scope excludes people 06, 23, 36 and 74 (all nights):
three PSG/hypnogram demographic conflicts and SC4362's scored Wake ending at
83,100 seconds despite a 67,980-second PSG. Causes remain UNKNOWN. The eligible
raw scope contains 74 people/146 recordings; the subject-disjoint EDF-78
extension contains 55 people/109 recordings. The original 20-person/39-recording
EDF-20/78 overlap is unchanged. Full-cohort acceptance remains BLOCKED.

## Authorized P2-SC canonical manifest and lazy data access

The follow-up authorizes only the exact reviewed 74-person/146-recording SC
subset. `load_sc_scope` hash-binds P1.5 gates/resolutions and the historical P1
proof. Excluded people cannot enter manifests. ST/SHHS are outside this phase.

`mist-transfer-sc-epochs-v1` stores recording descriptors joined by canonical
recording ID. Each descriptor includes source/annotation paths and SHA256,
study/person/night/variant, native clock (timezone UNKNOWN), duration, Fpz-Cz
montage, calibration bounds, acquisition prefilter, native/output rates and
units. Dense original-index/start-sample/start-second ranges preserve every
complete physical epoch. The grid is constructed from signal samples before
annotation access. Exact file byte extent, sample counts, contiguous EDF format,
100 Hz/uV and audited calibration/clock are validated; unsupported discontinuous
EDF, truncation or metadata changes stop the run, with the recording identified.

Original annotation descriptions/codes and TAL intervals remain explicit. R&K
3/4 map to N3. Movement/unknown/unannotated positions have y=-1 and a false
scoring mask, and stay in the inference timeline. A missing annotation is None,
not an invented stage. The manifest is normalized metadata: epoch i joins its
recording descriptor, starts at sample 3000*i and second 30*i, and inherits that
recording's source/calibration/annotation provenance. There is no cross-record
context join or inference-time label-based selection. Incomplete signal tails
are recorded in samples/seconds and excluded by geometry only.

`iter_recording(record, chunk_epochs=16, include_labels=False)` reads CPU float32
microvolts as [N,1,3000]. Batches expose original indices, physical start times
and the recording provenance join; include_labels=True additionally exposes y,
scoring_mask, original codes/descriptions and exclusion reasons. Inference does
not access annotation files or consult labels. Downstream supervised loss and
metrics MUST use scoring_mask; never compress unscored time into contiguous
context. No default Pz-Oz input, added filter, resampling, waveform export or
fitted normalization exists in P2.

The default batch provenance join excludes annotation intervals, stage counts
and scoring metadata. Include_labels=True is the explicit path to target fields;
annotations cannot arrive inadvertently through inference batch metadata.

`assign_subject_roles` accepts an explicit complete person-to-role map and
reuses the audit split guards, adding duplicate identities/nights/paths/source
checks. All nights stay together. `iter_development_split` opens train/val only
and rejects reserved test access before reading EEG. P2 assigns no final split
and fits no statistics; future normalization must use source TRAIN only.

The CLI requires a passing ten-person E/F/G and gap-case pilot before the
complete build. Pilot validation streams all physical epochs, checks all labels
with independent pyEDFlib and compares bounded waveform positions with MNE
volts converted to uV and an EDF digital-calibration oracle. Tolerance is fixed
before data: rtol=2*float32 eps, atol=2e-5 uV. Atomically completed manifests
verify every descriptor digest; resume binds code/environment/config/sources.
Outputs in `mist_transfer_runs/p2_sc/` stay out of GitHub.

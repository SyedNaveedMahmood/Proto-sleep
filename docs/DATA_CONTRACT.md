# Data integrity contract (P0/P1)

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

SC pair demographic conflicts quarantine the affected night and every other
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

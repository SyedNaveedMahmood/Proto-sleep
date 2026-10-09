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

# MIST-Transfer v3 provenance

Authoritative scope: root `next_plan.md`, P0/P1 only, 2026-10-09.

Verified v1 parent: `676963bea58b749d90662d133217fe72c4156981`.
Initial research branch HEAD: `3f614e7` (master plan added above that parent).
Development uses a separate worktree on `research/mist-transfer-v3-edf-shhs`.
Main and the v2 comparator remain untouched. No running job was interrupted.

Reuse inspected: v1 file/config SHA256, atomic JSON, selective train/validation
access, gap handling, actual training waveform-anchor provenance, additive
explanation numerical checks, and safe checkpoint completion/resume tests.
P0/P1 directly import proven digests and atomic JSON. No v1 code is changed.
The EDF-20-specific manifest and synthetic contiguous-index fallback are not
used in the new audit. No model or preprocessing pipeline is implemented.

All console logs are `.txt`; local waveform caches remain Git-ignored. Audit
outputs contain public Sleep-EDF inventory/counts/digests, never waveform arrays,
pickled patient headers, credentials, or SHHS records. SHHS defaults disabled.
Full installed package versions, git/source/config hashes, seed, output path,
split IDs and access flags accompany every run. `COMPLETE.json` means an audit
finished writing validated outputs, not that its scientific gate passed.

P0 committed as `97c9bddf814b0d86f85ef231710e297647f332ae`.
P1 implementation committed as `3f770c6c65c107eb171425142eefce91dd107bf0`.
The final audit ran against that exact committed implementation. Its provenance
records all 594 inventory input hashes and full package versions. Scientific gate
FAIL is preserved: seven checksum-valid ST annotation files cannot be read by
pyEDFlib 0.1.42; EDF78's optional distribution manifest lists only 50/153 files.
The complete console history includes failed preflights and corrected parser/
checker defects; none are presented as passing data runs.

Supplied raw path uses `Sleep_edf_fulldataset`; actual accessible path uses
`SLeep_edf_fulldataset`. The three explicit overrides are recorded in audit
provenance. Default user paths remain unchanged in the example config.

Official filename/label/study reference:
https://physionet.org/content/sleep-edfx/1.0.0/ . SC variants E/F/G were verified
against local published checksums and participant/night spreadsheets. No study
metadata or waveform snippets were invented. No clinical event labels assigned.

Final regression: 63 passed. Only existing v1 tests run tiny CPU synthetic
optimization to verify checkpoint behavior; no recording-data model was trained.
Run completion markers remain in ignored local runs. Published `audit.json` and
`AUDIT_FINAL_COPY_PASTE.txt` are exact audit outputs. `FINAL_COPY_PASTE.txt` adds
session validation/findings, so it is not misrepresented as an unchanged run
artifact with the original completion digest.

## P1.5 (2026-10-09; authorized follow-up scope)

Historical input/report commit: `f86ac39863cf8c0c40faff105db824206b69b497`.
The original failed P1 report and console logs remain unchanged. P1.5 reports
live exclusively under `reports/data_audit/p1_5/`. Expensive waveform proofs are
reused only after input hashes match the 594 recorded P1 hashes; no raw waveform
realignment or NPZ-array loading is required. A local trusted stat/hash receipt
supports subsequent reuse and a cheap postflight guards input stability.

Annotation validation independently compares a bounded full-byte TAL decoder,
MNE 1.10.2 and pyEDFlib 0.1.42 controls. No reader writes original data. The EDF+
Recordingfield/fixed-clock exception is documented alongside the original
headers and decoded timing evidence. Header demographic disagreements are
quarantined conservatively at person level, as is the 15,120-second scored Wake
overhang in SC4362. This is a source annotation/signal extent conflict with
UNKNOWN cause, distinct from a demographic disagreement. No epoch duration, stage or date is
repaired to obtain a passing result.

Adjacent archive evidence uses its central directory and CRC-checked tiny
MANIFEST.TXT members only. NPZ payloads are not extracted or re-audited from the
archive; the directory's cached raw-waveform proofs remain the content evidence.
Local `processor.ipynb` and DeepSleepNet `prepare_physionet.py` are hash-bound
candidate references, never executed. A Pz-Oz notebook cannot establish Fpz-Cz
NPZ execution lineage; candidate-rule agreement is reported as compatibility.

EDF+ normative reference (header and TAL grammar):
https://www.edfplus.info/specs/edfplus.html . These rules permit zero record
duration for annotation-only files, prescribe timestamped annotation lists, and
separate the fixed clock from the EDF+ descriptive recording identification.

Published P1.5 audit implementation: `2d789c688225b8f3b3beab59a38d4020c3f56586`
(initial compatibility resolver: `d4c7775919042d5dec19d626968f5085ad93109a`).
Release validation: 23 focused P1.5 tests and 92 relevant regression tests pass.
The read-only audit intentionally exits 2 for four unresolved SC recording
conflicts; person-level quarantine removes seven nights. All 47 historical P1
report/log files were checked byte-for-byte against `f86ac39` before publication.
No P2, recording-data training, GPU job or SHHS access occurred.

All P1.5 audit JSON/CSV files and `AUDIT_FINAL_COPY_PASTE.txt` are byte-exact run
outputs. `AUDIT_COMPLETE.json` preserves the original completion digests, with
its `FINAL_COPY_PASTE.txt` entry mapped to `AUDIT_FINAL_COPY_PASTE.txt`.
The expanded `FINAL_COPY_PASTE.txt` and `REVISED_AUDIT.txt` are session reports;
they are not claimed as unchanged source outputs. `validation.json` records the
artifact map, exact test commands and historical byte checks. Audit and test
terminal output is preserved in `reports/data_audit/p1_5/session_logs/*.txt`;
commit/push verification output remains in the ignored local run directory to
avoid changing a committed report merely to record its own publication SHA.

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

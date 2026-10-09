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

## P2-SC authorized follow-up (2026-10-09)

Reviewed parent: `19351edfd79807ba1716189bd3cc9671f2c97499`. The user authorized
SC-only P2 in the attached request. Earlier P1/P1.5 stop decisions remain
historical; ST preprocessing, SHHS access and P3/model training remain excluded.

Reuse: v1 SHA256/fingerprint/atomic JSON; transfer atomic CSV/text and AuditRun
transactions; canonical identities, calibration, annotation grids and split
guards; P1.5 strict paired annotation decoder and unchanged-input hash receipts.
The EDF-20 NPZ loader/20-fold order is unsuitable for the expanded raw SC grid
and is not used. Cached P1 waveform alignments are never repeated. Only selected
SC EDF/hypnogram sources are checked; no NPZ/ST/SHHS data discovery or opening.
If a trusted local stat receipt no longer matches, recompute that source digest
and stop on a changed hash. Every pilot/full run records source hashes, code,
environment, seed, scope proof, elapsed seconds and process peak RSS.

Reproduction (from this worktree; redirect full output to .txt with pipefail):

```
.venv/bin/python scripts/build_transfer_manifest.py --help
.venv/bin/python scripts/build_transfer_manifest.py --mode pilot --seed 123 --dry-run --output-dir mist_transfer_runs/p2_sc/v1/pilot
.venv/bin/python scripts/build_transfer_manifest.py --mode pilot --seed 123 --output-dir mist_transfer_runs/p2_sc/v1/pilot
.venv/bin/python scripts/build_transfer_manifest.py --mode complete --seed 123 --pilot-dir mist_transfer_runs/p2_sc/v1/pilot --output-dir mist_transfer_runs/p2_sc/v1/canonical
```

Use --resume with the same code/environment/inputs after interruption or to
verify completed output without waveform reloading. Import
`mist_transfer.manifest.load_complete` to verify/load normalized descriptors,
then `mist_transfer.preprocessing.iter_recording` for lazy inference chunks.
Canonical manifests, annotation metadata, waveform checks and terminal logs
stay local under ignored `mist_transfer_runs/p2_sc/`; no participant arrays or
private outputs are committed. Tests generate only synthetic EDF+ fixtures.

Validated implementation: `9f4ba06c71f0e577ebb58e2a8eba57f88605010e`
(initial implementation: `dc53edab7ef308a6fc99a5af42c6a35715afe60b`).
Canonical v1 covers exactly 74 subjects/146 recordings/397,832 physical epochs:
396,378 scored plus 119 Movement and 1,335 unknown positions. No extra subject
was excluded; no real input invariant failed. All 96 historical P1/P1.5 report
files remain byte-identical to the reviewed parent.

Focused tests: 25 passed. Full CPU regression: 149 passed, one CUDA-only test
skipped with GPU disabled. The ten-person E/F/G and gap-case pilot streamed
27,678 physical epochs and compared 66 bounded waveform epochs against MNE
and digital calibration. Maximum discrepancy was 7.616352832e-6 uV, within the
prespecified float32 tolerance. Full build verified 292 unchanged SC file
proofs without rehashing or repeating NPZ waveform alignment. Completed pilot
and full manifests resumed with all output digests verified and no EEG reload.

Final pilot external wall time: 4.14 s; peak RSS 600,532 KiB. Full metadata build:
2.31 s; peak RSS 562,084 KiB (includes library/import and JSON overhead). Largest
float32 waveform batch: 192,000 bytes. No signal arrays were exported. Earlier
implementation outputs remain local as historical checks; `p2_sc/v1/` is final.
Detailed measurements, exact key commands and full terminal output are in
`mist_transfer_runs/p2_sc/{validation_summary.json,COMMANDS.txt,session_logs/}`.

AuditRun resume binds the recorded Git SHA as well as the implementation hash,
environment/config/sources. To resume these completed runs after later commits,
use a worktree at the validated implementation with the same environment/output
identity, or create a new output directory. `load_complete` verifies and loads
the existing manifests under the current code without rerunning preprocessing.
No split, normalization fit, P3 baseline development or training was performed.

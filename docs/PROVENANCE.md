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

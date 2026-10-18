# Repository coding-agent instructions

This repository contains completed and frozen MIST-Transfer P0/P1/P1.5/P2-SC/P3A/P3B research artifacts. Do not rewrite their history.

## For MIST-COVE work
Before editing code, **read in order**:
1. `docs/mist_cove/README.md`
2. `docs/mist_cove/CODING_AGENT_MASTER_PROMPT.md`
3. `docs/mist_cove/RESEARCH_HYPOTHESES.md`
4. `docs/mist_cove/EXPERIMENT_PROTOCOL.md`
5. `docs/mist_cove/INTERFACES_AND_PROVENANCE.md`
6. `docs/mist_cove/IMPLEMENTATION_ROADMAP.md`
7. `docs/TRANSFER_PROTOCOL.md`, `docs/DATA_CONTRACT.md`, `docs/P3B_RUNNER.md`

The v0 package is **design-only**: its validation CLI is not a training or evaluation authorization. The old P3B runner/configuration, frozen split, and existing historical results remain immutable. Any subsequent coding request may authorize code/synthetic tests; real EEG training, reserved-participant evaluation, Sleep-Telemetry preparation, and SHHS access require their own explicit phase authorizations and protocol gates.

Non-negotiable:
- Do not access any of the 55 RESERVED SC subjects' EEG, annotations, descriptors or derived statistics in development, including by convenient globbing, exploratory scripts, debugging or caching. Do not assume reserved identities are accessible just because the split file lists their role.
- Use only `load_development(frozen, 'train')` / `load_development(frozen, 'val')` for data-bearing source development after separate authorization; reject any other role *before* opening a file.
- Never fit normalizers, retrieve waveform exemplars, or select architectures from VAL/test; only TRAIN may fit statistics and waveform banks.
- Never use legacy trimmed NPZ for zero-shot inference. Preserve every complete physical epoch and original position; mask unscored only for supervised loss/metrics.
- Do not treat Sleep-EDF-20 versus Sleep-EDF-78 as independent external cohorts.
- Do not claim external validation, historical novelty, SOTA, or improved transfer without permitted independent experiments.
- Keep all EEG, checkpoints, fitted banks, predictions, subject-level exports and run logs local/ignored under `mist_transfer_runs/`. Commit only safe source code, design configs, synthetic test results and aggregate evidence.
- Ask for explicit phase authorization before real-data or reserved/external work; when authorization is absent, stop after non-data-bearing contract checks.
- Never silently broaden permissions because an automated task or a document mentions a future phase.

For unrelated work, follow applicable historical research contracts; these new instructions do not authorize changes to existing frozen baselines.

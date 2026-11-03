# Verification record

## Executed in this environment

Date: 9 September 2026.

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, matplotlib 3.10.8, pytest 9.0.2. The source uses Python 3.10-compatible syntax and the parent package dependency contract, but the user's Python 3.10/CUDA environment has not been executed here.

Command:

```text
PYTHONPATH=src pytest tests/evidence -q
21 passed, 1 skipped in 11.99s
```

The skipped test requires CUDA. This is the new isolated suite, not a claim that all legacy repository tests were re-executed.

Also executed:

- Two-epoch synthetic CLI training, best-checkpoint reload, validation, and HTML/JSON/CSV explanation export.
- Full-default-dimension evidence, summary-only, and raw-context forward/backward passes on artificial [4,1,3000] tensors.
- Python compilation of the new package and entry script.
- `bash -n scripts/run_mist_evidence_dev.sh`.
- Visual inspection of a generated synthetic query/anchor overlay.

The default main model has 47,491 trainable parameters and 113 named evidence features. The summary-only control has 5,395 trainable parameters; the raw-context control has 66,760. These are implementation counts, not matched-capacity performance results. The smoke uses a reduced model with 8,095 trainable parameters.

## Tested properties

1. CRF partition, Viterbi path, and marginals agree with brute-force enumeration.
2. CRF right padding does not alter valid sequence scores; hole masks are rejected.
3. Additive evidence terms plus bias reproduce emissions, including boundaries.
4. A feature-level deletion produces the predicted change in emission scores.
5. Halo-block and full-record temporal emissions agree for core positions.
6. Distant samples outside a local window cannot change that window's match score.
7. Encoding is batch-composition invariant under the local normalization scheme.
8. Anchor buffers remain unchanged after an optimizer update.
9. Scores are bounded; matching the identical anchor window yields near-unit similarity.
10. Mandatory observed-component weights satisfy the documented lower bound.
11. Flat signals and low-precision CRF inputs produce finite tested outputs/gradients.
12. Test split access is rejected; a nonexistent reserved test file is never opened.
13. Subject crossing, duplicate paths, malformed labels, and wrong sample rates are rejected.
14. Known temporal gaps split sequences and never join separate recordings.
15. Missing original indices require an explicit assumption or independent-epoch mode.
16. Selected anchors exactly equal their recorded training-source snippets.
17. The main model and raw control share initialized local encoder states for a fixed seed.
18. Epoch-boundary interrupted/resumed training matches uninterrupted final tensors on CPU.
19. Best-so-far checkpoints are not mistaken for completed runs.
20. Completed checkpoint re-evaluation and signature mismatch rejection work.
21. Reference JSON defaults agree with typed model/training defaults.

## Not tested or established

Actual Sleep-EDF files and the user's RTX 4080 SUPER were not available. No CUDA success, real-EEG Macro-F1, clinical event validity, external transfer, or SOTA result is claimed. Existing experiment logs are historical evidence only. The development runner intentionally does not score a reserved test split.

The main PC must pass the CUDA smoke and the data preflight. In particular, original epoch continuity cannot be reconstructed from labels alone.

# Verification record

## Executed in the editing environment

- Python 3.13.5; PyTorch 2.10.0+cpu; NumPy 2.3.5; SciPy 1.17.0; scikit-learn 1.8.0; Matplotlib 3.10.8; pytest 9.0.2.
- New regression suite: **38 passed, 1 skipped**. The skipped test is the CUDA forward/backward test because no CUDA device is available here. The final run, including integration-test isolation of process-wide deterministic settings, completed in 10.50 seconds.
- Full synthetic command: `python -u scripts/run_mist_evidence.py smoke --output-dir <new-folder> --device cpu --threads 2` completed local train/validation, atomic checkpointing, CRF training, JSON/CSV/HTML export and input perturbation checks.
- Resume test simulated interruption immediately after an atomic epoch commit. Resumed training matched the uninterrupted final model tensor-for-tensor on this CPU stack.
- Explicit adaptive mean bins matched PyTorch adaptive-pool outputs and input gradients. The encoder avoids relying on atomic CUDA adaptive-pool backward in deterministic mode.
- `compileall` passed for the new package and entry script.
- CRF partition functions, marginal probabilities, label-sequence NLL and explanation identities were checked against exhaustive enumeration for short chains. Zero-transition CRF output was checked against independent softmax on a long chain.
- Loading train/validation succeeds while a deliberately invalid locked-test archive stays unread.

## Not executed here

The original repository's entire legacy test suite, the main PC's Python 3.10/CUDA stack, real Sleep-EDF recordings, the optional existing AttnSleep reference training arm, independent clinical event annotations and external-cohort evaluation were not executed in this environment. Existing repository code was read through the GitHub connector; ordinary network cloning was unavailable. The tests exercise the newly added package, not an invented full-repository verification.

The old modules are not rewritten by this branch. Run `python -m pytest -q` and the CUDA smoke on the main PC before launching the controlled experiments. Passing these checks is evidence of execution, not a guarantee of GPU portability or state-of-the-art performance. PyTorch documents that exact reproducibility across releases/platforms is not guaranteed.

## Data privacy

Only source, configurations, plans and tests belong in GitHub. No EEG exemplar waveforms, patient-derived checkpoints, private subject reports, or fabricated real-data scores are included in this commit. Generated outputs are ignored under `mist_evidence_runs/`.

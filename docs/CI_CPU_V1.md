# Full CPU regression environment v1

The Phase III Actions failure in run [39187604489](https://github.com/SyedNaveedMahmood/Proto-sleep/actions/runs/39187604489) occurred during collection: ten modules could not import `pyedflib` or `mne`. They were listed only in a separate audit requirements file, while CI installed `.[dev]`, whose sole extra was pytest. Generated EDF compatibility tests need these readers even though they never open a real cohort.

`.[dev]` now declares the full test dependencies. `.[edf]` separately exposes optional EDF runtime readers; neither changes the base model dependencies or authorizes data access. `configs/ci_cpu_constraints_v1.txt` fixes the scientific package versions for this CPU test job. It does not edit any frozen experiment or training environment.

Actions installs Python 3.12.9 and the official PyTorch 2.5.1 CPU wheel before resolving the constrained editable development install. The [official previous-version instructions](https://pytorch.org/get-started/previous-versions/) document the CPU wheel index. The workflow verifies `torch.version.cuda is None`, rejects installed NVIDIA runtime packages, checks dependency consistency and runs the entire test suite. CUDA-only tests retain their existing hardware skips; no tests are deleted, broadly excluded or marked `continue-on-error`.

Feature branches run through `pull_request`; direct push checks cover `main`. A new commit cancels a superseded check on the same PR. This avoids duplicate push/PR runs without disabling tests or changing account email preferences. The job has read-only repository permissions and two CPU threads for reproducible bounded execution.

Reproduce in a fresh Python 3.12.9 virtual environment:

```bash
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c configs/ci_cpu_constraints_v1.txt -e '.[dev]'
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 python -m pytest tests -ra
```

The actual remote workflow outcome, run URL and tested commit will be recorded with the Phase IV results. A local pass is not a substitute for that gate. Historical Phase I–III and P3B scientific sources, configurations and reports remain unchanged.

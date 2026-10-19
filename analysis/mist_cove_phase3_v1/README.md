# Phase III operational and postfit analysis sources

These files preserve the versioned operations used around the frozen Phase III trainer. They live outside the implementation inventory in `mist_cove.phase2.provenance.implementation_snapshot()` so adding analysis and reports does not invalidate the sealed training identity. The scientific trainer, prepared configuration, historical baselines and historical results are unchanged.

The operational files are archival execution sources, intended to be materialized in the ignored directory `mist_transfer_runs/mist_cove/phase3_source_v1/`. Their relative paths deliberately bind that location and the existing private authorization/provenance receipts. Do not execute the launch script directly from this directory, or overwrite an existing private execution file. No user approval, real EEG, participant identity, prediction file, fitted artifact, checkpoint or private trace is included here.

The following operations require the actual, separately recorded human Gate B authorization and the original matching private Gate A artifacts. Copying an archival script supplies neither authorization nor missing provenance.

| File | Purpose |
|---|---|
| `launch_gate_b_v1.py` | Verify actual Gate B authorization, original proposal, source/environment proofs and six epoch-one seals; release the verifier CUDA context before starting the child. |
| `monitor_gate_b_v1.py` | Supervise the unchanged runner, enforce the 24-hour gate and reject changed frozen code or competing GPU jobs. It stops only this experiment's child. |
| `monitor_best_predictions_v1.py` | CPU-only, seal-checked selected-best prediction collapse diagnostics. It does not read EEG or inspect every non-best epoch. |
| `post_pipeline_v1.py` | Wait for successful completion, verify frozen analysis source hashes, then run the private analyses once. Failed gates stop without automatic retry or publication. |
| `collect_phase3_results_v1.py` | Reverify all 18 checkpoint identities/seals, original full prediction grids, masks, selection, optimizer updates, resources and unchanged sources; project only aggregate results. |
| `waveform_analysis_v1.py` | Apply the frozen sample/donor/group plan to trained analytic candidates. Save real traces, plots and source ledgers only in ignored outputs. |
| `individual_attribution_summary_v1.py` | Separately summarize original additive scores and full-model individual effects, preserving their different meanings. |
| `render_aggregate_figures_v1.py` | Render figures and CSV from the verified public aggregate alone. No real EEG or checkpoint access. |
| `supplemental_audit_summary_v1.py` | Postfit descriptive confidence and joint-versus-individual summaries from the already frozen private audit, with no new interventions. Runs directly from the repository root. |

From the repository root, use the reconciled Phase III Python environment with `PYTHONPATH=src`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Once materialized and after successful training, the private analysis commands are:

```bash
mist_transfer_runs/mist_cove/phase3_source_v1/environment/bin/python mist_transfer_runs/mist_cove/phase3_source_v1/collect_phase3_results_v1.py
mist_transfer_runs/mist_cove/phase3_source_v1/environment/bin/python mist_transfer_runs/mist_cove/phase3_source_v1/waveform_analysis_v1.py
mist_transfer_runs/mist_cove/phase3_source_v1/environment/bin/python mist_transfer_runs/mist_cove/phase3_source_v1/individual_attribution_summary_v1.py
```

These commands require the original, hash-matching completed private artifacts and refuse to overwrite their final outputs. Inspect existing outputs rather than rerunning successful operations. A new experiment needs a separate identity, output root, protocol and authorization.

The detailed waveform plan is copied byte-for-byte to `configs/mist_cove_phase3_waveform_analysis_v1.json`. The parent protocol preceded training. The detailed plan and exact private sample list were frozen before any Gate B run completed, after intermediate scores already existed. Supplemental perturbation/group diagnostics are descriptive, not new confirmatory clinical tests. No spectral group is asserted to be a verified clinical event or independently identified redundant physiological structure.

The generated-input tests in `tests/mist_cove/test_phase3_waveform_analysis.py` and `tests/mist_cove/test_phase3_postfit_integrity.py` exercise the published analysis sources without calling a real source loader. Exact source hashes and quantitative outcomes belong in the final aggregate receipts and report. Four source-VAL participants cannot establish external transfer, and optimization seeds are not independent participants.

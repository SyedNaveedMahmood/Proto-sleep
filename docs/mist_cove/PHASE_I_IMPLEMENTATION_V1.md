# MIST-COVE Phase I v1 — synthetic implementation

The 2026-10-18 user execution request authorizes code and synthetic training/tests only. No real EEG, development descriptors, reserved subjects, ST, or SHHS are opened. Phase S/R/T remain blocked pending separate authorization and their protocol gates. The implementation branch starts at verified design commit `ac842955fdee1b911bb0ba8a82aed39abef4bbb3`; the original clean checkout was `research/mist-transfer-v3-edf-shhs` at `15995b3`. No merge or reset was used.

## Smallest candidate and falsifiers

Inspected existing interfaces: `AttnSleepAdapter.encode([N,1,3000])` already returns `[N,5]` logits and `emissions` is identity; `EvidenceModel.encode` returns `[N,F]` features, then `emissions([B,T,F])` returns `[B,T,5]`. Both expose `predict(logits)` and masked `loss`, with radius zero/no CRF in the frozen comparison. `evaluate_source` encodes every physical VAL position as `[N,1,3000]`, adds the sequence batch dimension for emissions, and masks only loss/metrics. `SourceRecording` preserves recording/subject/role, physical indices, labels and scoring masks; `LazyEpochArray` streams TRAIN-normalized FP32. `load_development` rejects every role except train/val before descriptor access. `training_bank` and `validate_bank` bind medoids and original crop positions to TRAIN-only provenance. `full_experiment` additionally guards implementation identity, frozen recipe, grid counts, sealed outputs and exact resume. None is modified or invoked on real data. Only the pure `confusion_metrics` implementation is reused by the new harness; old synthetic regression tests exercise their existing adapters and guards.

`src/mist_cove/` is a standalone epoch-only module. It accepts finite FP32 `[B,1,3000]`, restores calibrated microvolts with supplied frozen mean/std, and never fits normalization. The default identity normalization is for generated microvolt signals only. There is deliberately no real-data runner or real trace adapter in Phase I.

The analytic local encoder computes Hann spectral fractions in five bands (0.5–4, 4–8, 8–12, 12–16, 16–30 Hz) and physical RMS over 200-sample windows, stride 100. A six-input linear head followed by tanh produces signed five-stage evidence. Event selection ranks physical band RMS, resolves ties by sample time, and suppresses overlap before retaining up to four intervals. Flat windows are invalid; no event is fabricated for constant input. This is a compact analytic encoder, not a learned waveform encoder or a claim that band membership diagnoses clinical stages.

Availability is fixed spectral concentration above 0.35 multiplied by `RMS/(RMS+5uV)`. It is shared across stages, constrained to [0,1], and cannot learn a class bypass. This conservative first mechanism is intentionally weaker than stage-pair observability: it is neither calibrated probability nor an information-loss oracle. An isolated narrowband artifact can fool it. Its noise-scale constant is a synthetic assumption, not a TRAIN-fitted estimate.

The background head is linear in five whole-epoch spectral fractions, log physical RMS, and physical mean. Both its seven feature contributions and bias are exposed. The full model has 75 trainable parameters (background 40, event 35). Event magnitude is bounded by `8/K`; summation remains sensitive to repeated structures. Nonoverlap prevents sample duplication, not correlated-evidence overconfidence.

`Evidence` exposes intervals, validity, ranking scores, signed scores, availability, weighted contributions, background terms, and physical-input SHA256. Logits are computed directly as background plus summed event terms. `SyntheticTrace` binds the exact generated physical bytes to recording/night, original epoch index, scoring mask, and absolute half-open sample/time intervals. The ledger rejects mismatched hashes and every real-data role. Original gaps and unscored epochs remain intact; `masked_loss` excludes only supervised positions. Future real provenance must use verified source proofs and local ignored trace storage; synthetic hashes are not replacements for source-file provenance.

`audit_interval` recomputes the full prediction, background, and event selection after replacement. Linear bridge and explicit disjoint same-epoch background donors have a 20-sample endpoint-preserving crossfade and energy/edge/slope guards. It reports the claimed contribution, actual original-minus-replacement logit change, background/event deltas, and reselection. These guards reject obvious invalid replacements but cannot certify physiological or distributional plausibility.

The versioned config is `configs/mist_cove_synthetic_v1.json`. Criteria were set before the first held-out synthetic run. The initial config hash was `f8286a8e9218555fd5725a3cfe27e11b11d58a45894377875e1af68fa3559ac7`. Initial testing exposed three incorrect sign assertions on intentionally zero-score controls and a q-only control that still retained background. Correcting the control and adding the distinct background-plus-fixed-event-score comparison changed the config hash to `66a6d51ebd121ffdd36ce3b72a1550c91a0282bd6df57fd463e7582f1a85cee8`; **thresholds, candidate, optimizer recipe, seeds, and held-out construction were unchanged**. Initial receipt remains local at `mist_transfer_runs/mist_cove/synthetic_v1/results.json`. Exploratory random-control and redundancy probes added afterward are explicitly separate from predeclared acceptance checks.

## Experimental protocol

Synthetic classes 0–4 are artificial frequency codes, not clinically scored sleep. Each has two distinct carrier frequencies in its band, randomized phase, amplitude, physical location, and label-independent noise/off-band nuisance. Training includes either or both witnesses; evaluation uses fresh generator seeds and includes clean, reversible calibrated gain, removal of A/B/both, partial attenuation, and a matched-L2 off-band corruption that preserves A. There is no source/target cohort in these fixtures.

Seeds 123/456/789 each fit 150 generated training examples for 180 full-batch AdamW updates at lr 0.03, then evaluate 150 new examples per condition. This **synthetic** recipe is not the frozen future source-training recipe. The six comparisons are full, background-only, q=1 (score-only), true q-only (background zero, fixed event bias), background plus fixed event score, and time-first fixed-patch selection. All instantiate 75 parameters; active parameter counts differ. q=1 and fixed-patch retain the full candidate's head capacity and equal update budget. No capacity-parity claim is made for background-only or q-only. Historical raw_context/evidence checkpoints are not loaded.

H1 is falsified here if availability does not improve selective risk over both entropy and global RMS by at least 0.01 at 95/90/80% coverage. Full coverage and class retention are always reported. This easy synthetic task may leave no room for such superiority; passing ordinary signal-loss detection does not establish H1's distinguishing prediction.

H2 tests selected positive and negative contributions, multiple donors, energy/length-nearest non-witness intervals, and separate randomized matched nuisance controls. Primary aggregate selected-minus-control logit effect must exceed 0.05 with mean donor disagreement at most 0.2. Claims are compared with true effects, including their MAE/correlation; equality is not assumed. Similar-looking nonselected repeated waveforms expose the limits of single-witness necessity under reselection.

H3 removes each alternative independently and both together, measures contribution shares, requires at least 0.9 single-loss accuracy, a shift toward the surviving witness, and erased confidence at most 0.5. Model collapse, q-only bypass, prediction counts, corruption confidence, class-specific margins, fixed-five F1/confusion/recall, balanced accuracy, kappa, NLL, Brier, ECE15, and risk–coverage are recorded without silently repairing negative results. Seeds are repetitions of constructed experiments, not participant uncertainty estimates.

## Reproduction and receipts

Run from the repository root with its existing virtual environment:

```bash
python scripts/validate_mist_cove_design.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests -ra
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/run_mist_cove_synthetic.py \
  --output mist_transfer_runs/mist_cove/synthetic_v1_corrected/results.json
```

The runner refuses existing output paths. Use a fresh output path on repeat runs. It saves config bytes/canonical hashes, implementation file hashes and base commit, Python/PyTorch identity, actual updates, CPU wall time/RSS, all conditions/controls and failed scientific checks. Only aggregate synthetic results may be committed. Fitted models are not saved. GPU memory, throughput, and exact CUDA resume are not measured in this phase; the 16 GB RTX 4080 SUPER remains a future constraint.

Final results and exact test counts are recorded in [PHASE_I_RESULTS_V1.md](PHASE_I_RESULTS_V1.md) and the linked synthetic aggregate receipt. The default shell lacks pytest; `.venv/bin/python -m pytest` uses the installed repository environment.

## Next separately authorized gate

Do not infer a source-training green light from software correctness. H1 failure requires a recorded decision on whether to test this deliberately restricted candidate on source data or preregister another synthetic mechanism. A source-only proposal must first freeze a new candidate/config and explicit authorization; verify unchanged split SHA256/fingerprint and TRAIN/VAL proofs without opening RESERVED descriptors; reuse verified TRAIN-only normalization; implement and synthetic-test a separate radius-zero adapter/runner with full physical-grid validation, scoring masks, finite-state checks, atomic provenance, and exact interrupted CUDA resume. It must preserve the P3B budget: seeds 123/456/789, AdamW 3e-4/1e-4, max 60 epochs, patience 12, batch/core 16, clip 5, deterministic FP32, AMP/TF32 off. Earliest-best participant-mean fixed-five source-VAL F1 selects the checkpoint. Preflight must measure compute/VRAM on the actual 4080 SUPER and retain matched q-free/background and raw_context controls. No real-data runner has been validated here, and no source experiment may start automatically. RESERVED, ST, and independent external evaluations each remain separate later gates.

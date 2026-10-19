# Phase V synthetic identifiability results v1

**Decision: do not launch another real EEG experiment.** The historical benchmark does not isolate temporal phase, and the new benchmark fails its amplitude-restriction criterion even though conventional learning succeeds. The confirmatory sequence stopped at V3; no new additive-head fit was executed.

## Frozen execution and scope

Base: Phase IV `7fd29d3bb6d36c4b7c27f09273ad155922aa4de1`; branch `research/mist-cove-phase5-identifiability-v1`. Protocol/config/criteria were committed at `6405148` before confirmatory fits, then implementation/tests were frozen at `03687bc`. Full hashes are in `experiments/mist_cove/phase5_preregistration_v1.json`, `phase5_execution_manifest_v1.json` and the [execution receipt](../../reports/mist_cove/phase5_v1/execution_receipt_v1.json).

All inputs were generated tensors, reconstructed historical synthetic signals or mock interfaces. No real TRAIN/VAL EEG, data-bearing source descriptor, fitted real artifact, RESERVED input/annotation/descriptor, Sleep-Telemetry, SHHS or legacy NPZ was opened. Historical source code, configs, splits, scientific reports, checkpoints and metrics remain unchanged. No other GPU process was interrupted, no credential was used/saved, and no CI dependency or workflow change was needed.

Three seeds, two historical tasks and one newly versioned harmonic task yielded **114 completed fits / 34,200 optimizer updates / 547,200 example exposures**. These are new audit fits, not reruns or revisions of the 54 Phase IV fits. Historical controls get 1,000 balanced generated TRAIN examples and 2,000 independent evaluation examples per condition, with 300 batch 16 updates. Conventional controls retain the exact 3,593-parameter encoder/head. Restricted probes get the identical minibatch exposure, TRAIN-only feature standardization and a fixed 32-unit hidden layer; they have different, explicitly reported capacities. No evaluation-based selection or post-result threshold/generator/model change occurred.

## Gates V0/V1: historical reconstruction and leakage

**PASS V0:** original hashes and exact source replay verified. **FAIL historical phase isolation:** isolated-template magnitudes match to 2.17e-7 relative range, but zero-padding to 3,000 exposes 0.7363 relative class range. Diagnostic-only magnitude accuracy is 1.0000/0.999333 across historical tasks. This alone rejects the exclusive-phase interpretation without needing nuisance interference. See the [generator audit](PHASE_V_GENERATOR_AUDIT_V1.md) for source-level RNG, cross-term, cropping, boundary and transformation measurements.

New clean audit means across seeds:

| Input/decoder | Substitutable Macro-F1 | Complementary Macro-F1 |
|---|---:|---:|
| Original waveform, conventional M1 | 0.656616 | 0.337727 |
| Historical independent patch-zero M1 | 0.959905 | 0.747332 |
| Coherent epoch-zero M1 | 0.112118 | 0.105395 |
| Diagnostic-only embedded magnitude | 1.000000 | 0.999333 |
| Complete epoch magnitude | 0.414277 | 0.301651 |
| Original patch magnitude | 0.361378 | 0.268249 |

This table uses a new sample/exposure identity; it does not replace Phase IV's 0.4794/0.2781 original-waveform results. Patch-zero exceeds both other waveform controls by ≥0.10 in every seed/task. Complete-signal magnitude probes decode substantial information on the substitutable task, but do not reach the neural patch-zero result. Magnitude information is sufficient for the diagnostic-only task, while the exact optimization/accessibility explanation in noisy superposition remains unresolved. Historical M1_phase also retains signed patch means; the magnitude-only probes retain DC magnitude rather than its sign. Their information maps and capacities are therefore not identical, further preventing attribution of the entire gap to reconstruction alone.

All historical position/nuisance/amplitude/envelope/permuted probes meet the specified clean accuracy equivalence criterion: each 99% Wilson interval lies within [0.15, 0.25]. The chance reference uses independent balanced random tie-breaking and uniform 0.2 probabilities; it does not confuse constant-class Macro-F1=0.0667 with balanced chance. This is finite-probe evidence, not proof of universally absent nuisance information. The complementary complete epoch/patch magnitude results are intermediate under the stronger leakage rule, rather than a PASS for equivalence. Near-chance probes can still be overconfident: substitutable nuisance-magnitude confidence averages 0.5308 with accuracy 0.2003.

## V2/V3: new harmonic relative-phase code

**PASS V2 correctness:** one continuous 3,000-sample waveform, exact paired nuisance preservation, original fourteen windows, finite geometry, noiseless full/patch magnitude class range 1.94e-8, and analytic relative-phase oracle accuracy 1.0 in all six seed/clean-or-competitive checks. The artificial class is the relative phase of 2-Hz and 4-Hz components under randomized global phase, amplitude, independent 13-Hz nuisance and Gaussian noise. It is neither a sleep-stage observation nor a local clinical event. Every candidate contains the diagnostic code.

**PASS conventional learnability component:** original M1 clean and competitive Macro-F1 both equal 1.0 in every seed. Competitive means a continuous independent 40-µV 13-Hz nuisance here, not arbitrary artifacts. Diagnostic-class swaps change every paired prediction; replacing nuisance preserves every paired prediction, across 100 independent paired nuisance realizations per seed. Bootstrap intervals for those observed fractions are [1, 1]; they do not prove future-perfect performance. Coherent and independent patch phase removal reduce clean accuracy to 0.1980/0.2038 means, consistent with removing the intended code in those defined representations.

**FAIL compound V3 benchmark validity in all three seeds:** amplitude summaries recover class information. Their clean accuracies are 0.6085, 0.5855, 0.6080, with 99% intervals [0.5801, 0.6362], [0.5569, 0.6135], [0.5796, 0.6357]. The near-chance requirement fails decisively. This is not independent nuisance amplitude leaking labels: intended relative phase changes waveform extrema/pointwise amplitude structure. Equal RMS and magnitude did not make the declared amplitude summaries non-informative. That shortcut can explain part, though not all, of five-class performance. No benchmark revision or retraining was attempted after seeing it.

Other new restricted clean/competitive probes satisfy their interval criterion in every seed. This does not exclude untested spectral decoders or amplitude interactions. Their large parameter counts and high confidence at chance also show finite-sample overfitting; accuracy equivalence is not calibration or a universal absence proof.

| New-task method | Parameters | Clean F1 | Competitive F1 | Clean accuracy |
|---|---:|---:|---:|---:|
| Original conventional | 3,593 | 1.000000 | 1.000000 | 1.000000 |
| Independent patch-zero conventional | 3,593 | 0.104216 | 0.066667 | 0.203833 |
| Coherent epoch-zero conventional | 3,593 | 0.096725 | 0.066667 | 0.198000 |
| Epoch magnitude probe | 48,229 | 0.196693 | 0.193006 | 0.197000 |
| Patch magnitude probe | 90,245 | 0.203490 | 0.137552 | 0.203667 |
| Amplitude-summary probe | 3,077 | 0.587279 | 0.066667 | 0.600667 |
| Position probe | 869 | 0.138269 | 0.138269 | 0.195000 |
| Nuisance magnitude probe | 48,229 | 0.199239 | 0.193797 | 0.199667 |
| Nuisance latent probe | 709 | 0.165389 | 0.078570 | 0.198167 |

## Interventions, confidence and stopping

**V4 NOT TESTED:** V3 failure prevented all fixed-encoder additive/conventional/event-disabled head fits, with zero V4 optimizer updates. The code path is guarded and tested. Exact additive/counterfactual correctness tests do not constitute evidence-organization success. A global code would also make focal localization superiority non-identifiable when every candidate is genuinely diagnostic; no local truth was invented to rescue the comparison.

Conventional M1's erased accuracy is 0.1970 with confidence 0.2577. Its new-task clean/competitive/erased/artifact NLL is 0.000846/0.037064/1.625530/0.488762; Brier is 0.000002/0.002958/0.806783/0.244137; ECE is 0.000845/0.036063/0.060688/0.074801. Independent abrupt 20-sample ±80 artifacts reduce mean accuracy to 0.8355 while confidence remains 0.9020. Accurate harmonic classification is not generic artifact robustness. Amplitude-probe competitive accuracy is 0.20 with confidence 0.9811, another retained false-confidence failure.

Original-model internal feature-deletion rankings receive individual/group oracle, bridge and independent-nuisance interventions on the first synthetic example of each class. Signed background/full/fixed/reselection effects remain separate. Mean donor spread is 1.3907/1.1917/1.1184 logits on the new task, exceeding 0.25 in every seed. Tiny-noise ranking Jaccard distances are 0.08/0/0.08. Global diagnostic support prevents a meaningful focal localization advantage, and conventional models have no native additive event scores. Their zero counterfactual reconciliation residual is bookkeeping, not evidence necessity. Alternative donors remain unvalidated physiologically. The complete vectors/diagnostics, per-stage recall/confusions, calibration, seed variation, pair bootstrap intervals and failed clauses are retained in the lossless [per-run JSONL](../../reports/mist_cove/phase5_v1/per_run_v1.jsonl), [CSV](../../reports/mist_cove/phase5_v1/metrics_v1.csv), [summary](../../reports/mist_cove/phase5_v1/summary_v1.json) and [signal audits](../../reports/mist_cove/phase5_v1/signal_audits_v1.json).

## Engineering and measured resources

- Full CPU suite: **582 passed, 0 failed/errors, 11 CUDA skips**, including 78 new tests; one pre-existing warning. Generated geometry/RNG/independence, exact replay, original intervals, role rejection before file access, finite five-output backward, counterfactual accounting and atomic sealed checkpoint recovery pass.
- Separate existing CUDA regression: **11 passed, 0 failed/skipped, 582 deselected**, 4.16 s. This and the CPU run cover all 593 collected cases, but are not called a single 593-pass invocation.
- New bounded CUDA: **five batch 16 smoke checks and three exact model/optimizer/Python/NumPy/CPU/CUDA RNG resume probes PASS**. Actual RTX 4080 SUPER, driver 580.95.05, imported torch 2.5.1 / CUDA 12.1 / cuDNN 90100, NumPy import/metadata 1.26.4. Peak smoke allocator 69.246 MiB allocated / 90 MiB reserved; maximum additive residual 2.98e-8. Checker budget wall 1.163 s excludes Python import and preliminary device query. Cold-start/order/cache effects prevent comparing these timings as model-speed benchmarks; allocator peaks exclude context/driver memory and are not full EEG feasibility.
- Confirmatory CPU study: **113.132s** wall with two threads, peak process RSS **1,751,016 KiB (1.670 GiB)**; well below the frozen 1,800 s ceiling. Imported Python 3.12.9, torch 2.5.1+cpu, NumPy 1.26.4; deterministic FP32/TF32 off. Per-fit times/capacity/updates are recorded, not asserted equal actual compute. No confirmatory training used the GPU. The CUDA restricted-probe smoke uses 201 features, not the largest 2,814-feature CPU probe; its allocation is not a measurement of every restricted classifier.

Correctness and CUDA measurements are in separate public receipts. Actual GitHub Actions must pass on the final PR head; the verified workflow URL is recorded in the draft PR rather than inserting a self-referential commit into this report. Existing CI/dependencies remain unchanged.

## Final scientific answers and next decision

1. **Why did phase removal help?** Local magnitude information and a zero-padding leakage mechanism are demonstrated; inconsistent patch reconstruction and a matched neural advantage are measured. The exclusive share due to reconstruction/optimization remains unresolved. It did not demonstrate temporal-morphology necessity.
2. **Did the benchmark isolate morphology?** Historical benchmark: **FAIL**. New task: **NOT ESTABLISHED** under the declared restricted-decoder requirement, because amplitude summaries predict labels.
3. **Can conventional learning recover intended information?** **PASS for the declared harmonic learnability and paired-nuisance controls** in all seeds; **FAIL for the compound shortcut-isolation claim**. Perfect generated accuracy is not a clinical biomarker.
4. **Does organized additive evidence improve a fixed representation?** **NOT TESTED in Phase V**, because the prerequisite gate failed. Historical negative conclusions stand.
5. **Is another EEG experiment justified?** **No.** A new independently versioned diagnostic-information design must address amplitude-summary predictability and a nontrivial localization estimand before further evidence-organization claims. Do not tune this confirmatory generator into success. Real source, reserved and external work also lack authorization.

## Reproduction

Use the existing constrained CPU environment described in `docs/CI_CPU_V1.md`, with `PYTHON` pointing to that interpreter. Execution manifests bind source/config bytes; the runner refuses reuse of an existing output directory and stops on integrity/resource failure.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" -m pytest tests -ra
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" scripts/run_mist_cove_phase5.py \
  --output mist_transfer_runs/mist_cove/phase5_v1/NEW_UNUSED_OUTPUT
# CUDA_PYTHON must be a CUDA-enabled interpreter; GPU must be idle.
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$CUDA_PYTHON" scripts/check_mist_cove_phase5_cuda.py \
  --output mist_transfer_runs/mist_cove/phase5_v1/NEW_UNUSED_CUDA_OUTPUT
```

`analysis/mist_cove_phase5_v1/collect.py` verifies the completed private aggregate receipt and frozen identities before lossless projection. It opens no EEG or checkpoint and refuses public-result overwrite. All weights, noise/latent traces, checkpoint states and detailed logs remain local/ignored. Stop here; no new real experiment follows.

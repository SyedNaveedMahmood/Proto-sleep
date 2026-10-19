# Phase VI localized diagnostic evidence v1 — results

**Decision: retain the localized benchmark and negative robustness result; do not launch real EEG training.** The construction and specified unrelated-feature controls pass. Conventional M1 learns the clean local codes, but fails the collective competitive-performance gate. All confirmatory matched-head comparisons therefore remain NOT TESTED, with zero VI-4 updates.

## Identity and scope

Base Phase V: `a7c32bfff85a6f991404443ad216adab04f4df4c`; branch `research/mist-cove-phase6-local-evidence-v1`. Construct/config preregistration commit `b36f5ca`; scientific implementation/test freeze `438cffc7807bff3881f74a9a0a3e187cf88ac158`. Source/test inventory SHA256 `9bca5bc8b1a372c75ff8472ae7e06882bbd181b57a968f7ba18b69b3492c4e09`; config file SHA256 `c9006a1697c417c4bb0663c68bc572b0b7597b7908b454836b2a6ce40a48d75a`. Full byte inventories and commands are recorded in the preregistration/execution manifests and [execution receipt](../../reports/mist_cove/phase6_v1/execution_receipt_v1.json).

Only generated inputs, mock interfaces and public historical aggregates were used. No actual TRAIN/VAL EEG, annotation or development descriptor, fitted real normalization/bank, RESERVED input/descriptor, ST, SHHS or legacy NPZ was opened. The immutable source split, historical results/checkpoints/code/configs and repaired CI are unchanged. No other GPU job was interrupted and no credential was requested, used or stored. Previous real-data approvals did not authorize this phase to launch source training.

## Construct and generator validity

**PASS VI-0/VI-1 within the declared scope.** Diagnostic components occupy a nominal 200/3000 samples for single witnesses and at most 400/3000 for the two primary witnesses. Full continuous-grid construction precedes patch extraction. Each complete event has a containing 400-sample candidate; overlapping candidates can see fragments. Class, placement, diagnostic parameters, background, noise, nuisance, artifacts and redundant/complementary mode have separate streams. Exact class swaps retain baseline/geometry, and component deletion recomposes the stored class-free baseline exactly. No label or latent metadata enters the blind waveform classifier.

All four unrelated-feature probes (background+noise, nuisance+noise, position and artifact-only) have clean/competitive 99% accuracy intervals inside [0.15,0.25] in every task/seed. Separately fitted removed/background/nuisance M1 controls also meet this criterion. This is bounded evidence for the specified decoders and exposure, not proof of absence for every possible nuisance decoder. Tasks deliberately share matched baseline realizations and labels at the same seed, so repeated negative-control results are not independent replications. Seeds are generated optimization repeats, never human participants.

Task A uses one tapered frequency code. Task B mixes two redundant complete witnesses with two typed complementary factors. Factor type differs from complete-witness type, while duration, positioning and amplitude distributions are shared across labels. A complete remaining witness still carries its code. An anonymous factor is compatible with two labels under unspecified-member removal. **Important limitation:** the frozen partial condition always removes the first factor; conditioning on that known intervention rule reveals that the remaining code is the successor. Thus the protocol's broad ambiguity language must not be read as proving unidentifiability for that specific partial dataset. This was not an acceptance endpoint, and neither generator nor outcomes were revised after execution.

Local spectral information is intentional Class B information, not Class C leakage. Oracle-region spectral probes reach mean clean/competitive F1 about 0.9998/0.9577 (single) and 0.9993/0.9591 (multiple). Full-epoch magnitude probes have clean F1 1.0 in every seed. Oracle-region amplitude probes remain weaker (clean F1 about 0.163/0.184); this does not establish universal amplitude non-informativeness. Oracle proxies are given interval information unavailable to blind M1, and have different capacities. These tasks test localized diagnostic frequency/waveform information, not exclusive phase dependence or clinical morphology.

## Conventional learnability and failed stopping gate

The original encoder remains 3,408 parameters; M1 remains 3,593. Five waveform controls per task/seed train on full, oracle-only, removed, background+noise and nuisance+noise inputs. Restricted probes use fixed 32-unit hidden layers and TRAIN-only standardization. Every confirmatory fit has 300 batch-16 AdamW updates at lr 0.001 / wd 0.0001 / clip 5; final-step evaluation, no checkpoint selection, no tuning or budget extension. Three seeds, two tasks, four leakage probes, five waveform controls and three proxy probes give **72 completed fits / 21,600 optimizer updates / 345,600 example exposures**.

| Task | Seed | Full M1 clean F1 | Full M1 competitive F1 | Oracle-only clean/competitive F1 | VI-3 |
|---|---:|---:|---:|---:|---|
| Single | 123 | 1.000000 | 0.435111 | 1.000000 / 1.000000 | FAIL |
| Single | 456 | 0.980504 | 0.742143 | 1.000000 / 1.000000 | PASS |
| Single | 789 | 0.999500 | 0.615255 | 1.000000 / 1.000000 | FAIL |
| Multiple | 123 | 0.935341 | 0.338518 | 0.999000 / 0.999000 | FAIL |
| Multiple | 456 | 0.984511 | 0.695932 | 0.998500 / 0.998500 | FAIL |
| Multiple | 789 | 0.997002 | 0.541102 | 0.999500 / 0.999500 | FAIL |

All clean F1 thresholds, clean diagnostic-substitution criteria, ordinary nuisance-pair agreement and learned null-control criteria pass. Appropriate diagnostic changes are 1.00/0.97/1.00 for single and 0.91/0.96/0.98 for multiple; nuisance agreement is 1.00/0.97/1.00 and 0.87/0.96/0.98. Each uses 100 independently generated paired nuisance realizations with stored 500-replicate bootstrap intervals. These clean pairs do not test invariance to the stronger competitive nuisance distribution.

**FAIL collective VI-3:** five of six comparisons fail the frozen competitive F1≥0.70 criterion. In particular 0.695932 fails; it was not rounded into a PASS. No hard seed or unsuccessful condition was removed. Mean clean/competitive full-M1 F1 is 0.993335/0.597503 for single and 0.972285/0.525184 for multiple. Oracle-only performance demonstrates available local information under these synthetic inputs, but does not uniquely assign the full-waveform failure to encoder capacity, pooling, optimization or signal interference. Phase III clinical failures remain unchanged and cannot be reinterpreted using these artificial codes.

**VI-4 NOT TESTED:** the collective failure prevents all M1/M2/M3/M2_rms frozen-encoder head fits. Four model types appear only in generated engineering/debug smoke and resume tests; these are not confirmatory evidence-organization comparisons. No M2 gain, fixed-encoder comparison result or native learned witness faithfulness is claimed.

## Localization, oracle effects and difficult cases

Conventional explanations rank internal leave-one-candidate feature deletions, clearly distinct from native additive scores and full-waveform replacements. Four-window budgets are matched to random and RMS rankings. At IoU≥0.25, clean top-K hit rates average 0.9423/0.9665 (single/multiple), compared with random 0.4432/0.7095 and RMS 0.5515/0.7565. High top-K hit is a lenient any-hit measure: single selection precision is only 0.3855, multiple precision 0.6553. It is not equivalent to identifying all necessary evidence.

| Task/condition | Internal top-K hit | Event coverage | Selection precision | Mean best IoU | False nuisance fraction |
|---|---:|---:|---:|---:|---:|
| Single clean | 0.942333 | 0.942333 | 0.385458 | 0.193210 | 0.265583 |
| Single competitive | 0.877333 | 0.877333 | 0.326042 | 0.158771 | 0.516250 |
| Multiple clean | 0.966500 | 0.866417 | 0.655292 | 0.314576 | 0.097833 |
| Multiple competitive | 0.950167 | 0.752250 | 0.508500 | 0.247543 | 0.316708 |

Good any-hit localization can coexist with wrong classifications and irrelevant selected windows. Under competition, mean confidence is 0.8813/0.8128 despite accuracy 0.5838/0.5488; incorrect confidence≥0.9 occurs in 20.32%/11.92% of examples. After full diagnostic removal, accuracy is 0.200/0.192 with confidence 0.8031/0.7159. Two single-task seeds retain confidence above 0.90 on removed examples. This is a retained corruption-confidence failure, not principled abstention. Fully eventless inputs have no selected intervals and null localization, not fabricated oracle events.

Other full-M1 stress Macro-F1 means (single/multiple): nuisance substitution 0.9927/0.9735; position shifts 0.9920/0.9703; repeated distractors 0.9826/0.9423; correlated distractors 0.9840/0.9468; weak diagnostics 0.7447/0.7135; strong artifact 0.7513/0.5847. Multiple-only redundant/complementary/adjacent/overlap F1 is 0.9852/0.9559/0.9353/0.9138. Partial F1 0.9440/0.6798 is descriptive under the particular known removal rule, with the ambiguity limitation above. No stress threshold was introduced after scores.

Generator-declared groups receive exact component deletion and individual member removal, separately from background/fixed-path/reselection bookkeeping. Across 15 selected redundant-group examples, mean joint absolute true-class effect is 8.4795 logits versus 2.3069 for the largest individual effect; joint exceeds largest individual in 14/15. Complementary means are 9.0282 versus 6.6523, exceeding in 12/15. These are descriptive, first-per-class model examples, not independently sampled clinical evidence or a universal redundancy criterion. Conventional internal-ranking tiny-noise set Jaccard distance is zero in these sampled audits; stable ranking does not establish native additive faithfulness.

M1's native event contributions are zero: its entire pooled decision is called background in the additive interface. Exact full-effect decomposition therefore reconciles trivially through background changes (max residual 0). Untrained M2 engineering checks verify signed accounting, but cannot establish learned individual necessity. The length/RMS-matched non-diagnostic control is zeroed, whereas exact oracle deletion removes only the diagnostic component and retains background. That contrast uses different intervention types and cannot isolate a unique causal mechanism; this limitation is explicit in every audit record. Internal score deletion is never labeled physiological.

## Donor gate and limitations

**FAIL numerical donor availability; donor robustness NOT TESTED.** Across 105 nonempty audited examples, zero of 210 bridge/independent-baseline replacements passed all fixed guards. Energy/spectral diagnostics are computed on concatenated targeted union segments, with separate boundary-jump checks. No guard was weakened or refitted on evaluation draws; the historical 3-µV guard is unchanged. Rejected donor effects remain visible but accepted-donor spread is null, never fabricated agreement. These failures do not invalidate exact generator intervention effects, and even accepted numerical donors would not certify physiological erasure.

The model may exploit legitimate local spectra and other allowed proxies. No blinded clinical-event annotation, real EEG, independent population or transfer endpoint was tested. Strong artifact distribution differs from TRAIN; failure does not identify a unique architectural cause. Source data remain sealed for any new experiment. Partial-factor identifiability and the unmatched oracle/control erasure types require explicit refinement before stronger follow-up claims, not retrospective alteration of this experiment.

## Engineering and actual resources

- Full CPU suite: **673 passed / 0 failed/errors / 11 CUDA skips**, including 91 new cases; one existing warning. Tests cover independent streams, exact causal pairs/recomposition, original intervals, no latent metadata in Class C features, empty truth, five outputs/backward, group/reselection accounting, role-before-filesystem rejection and exact model/optimizer/RNG recovery. Existing mock source masks/grid protections remain covered.
- Separate existing CUDA suite: **11 passed / 0 failed/skipped / 673 deselected**, 4.20 s. CPU and CUDA invocations cover all 684 cases; they are not reported as one 684-pass invocation.
- Four bounded generated CUDA batch 16 forward/backward/interval/additive checks and four exact model/optimizer/Python/NumPy/CPU/CUDA RNG recovery probes: **PASS**. Actual RTX 4080 SUPER, driver 580.95.05, Python 3.12.9, torch 2.5.1 / CUDA 12.1 / cuDNN 90100, NumPy import/metadata 1.26.4. Peak smoke allocation 69.064 MiB, reserved 90 MiB; maximum additive residual 1.1921e-7. Checker budget wall 1.107 s excludes interpreter import/initial device query. Cold-start/order/cache effects and context/driver allocations limit interpretation; these are not real-source feasibility measurements.
- Confirmatory synthetic CPU wall **124.557 s**, two threads; peak process RSS **3,261,728 KiB (3.111 GiB)**, below the 1,800 s ceiling. Environment Python 3.12.9, torch 2.5.1+cpu, NumPy 1.26.4, deterministic FP32/TF32 off. Per-fit actual updates, exposure, parameters and time remain in the receipts. No confirmatory training ran on GPU.
- Before the implementation freeze, generated tests caught five failing cases caused by a manifest-key assumption and a positional argsort argument. Both were fixed before confirmatory evaluation; no scientific condition, model, seed or threshold changed. All negative scientific runs remain retained.

See [correctness receipt](../../reports/mist_cove/phase6_v1/correctness_receipt_v1.json), [CUDA receipt](../../reports/mist_cove/phase6_v1/cuda_receipt_v1.json), [lossless per-run records](../../reports/mist_cove/phase6_v1/per_run_v1.jsonl), [metric CSV](../../reports/mist_cove/phase6_v1/metrics_v1.csv) and [aggregate summary](../../reports/mist_cove/phase6_v1/summary_v1.json). These include class-wise recall/confusions, NLL/Brier/ECE, localization controls, signed individual/group vectors, all gate clauses and unsupported donors. The final-head green Actions URL is recorded in the draft PR to avoid a self-referential report commit. No CI workflow/dependency change was needed.

## Final answers

1. **Localized benchmark? PASS within the construction and tested leakage controls.** Diagnostic support is local and counterfactually traceable. No exclusive temporal-phase or clinical validity claim. The partial-condition interpretation is limited as described.
2. **Conventional learning? Clean learning and ordinary paired use PASS; the full demanding VI-3 contract FAILS.** Strong nuisance competitiveness is not reliable across seeds/tasks.
3. **Additive incremental usefulness? NOT TESTED.** The prerequisite failed; no confirmatory matched-head comparison ran.
4. **Witness faithfulness? Oracle group dependence and conventional internal rankings are measured; native additive necessity and accepted-donor robustness remain NOT TESTED.** Redundant witnesses need not be individually necessary.
5. **New real EEG study justified? NO.** First resolve the prespecified synthetic robustness failure and estimator/partial-intervention limitations in a separately versioned protocol. No new architecture, optimization budget or real-data campaign is automatically authorized.

## Reproduction and stopping point

Use the existing constrained CPU and reconciled CUDA environments; no historical environment was modified. Set PYTHON/CUDA_PYTHON to those appropriate interpreters and choose unused private outputs.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" -m pytest tests -ra
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PYTHON" scripts/run_mist_cove_phase6.py --output mist_transfer_runs/mist_cove/phase6_v1/NEW_UNUSED_OUTPUT
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$CUDA_PYTHON" scripts/check_mist_cove_phase6_cuda.py --output mist_transfer_runs/mist_cove/phase6_v1/NEW_UNUSED_CUDA_OUTPUT
```

The CLI verifies frozen source/config and historical hashes, bounds wall time, writes atomic failure/completion receipts and refuses output overwrite. The read-only collector projects the generated aggregate receipt and refuses public overwrite. All learned weights, complete latent/noise traces, optimizer/RNG states and detailed logs stay ignored locally. Stop before any real EEG or restricted/external evaluation.

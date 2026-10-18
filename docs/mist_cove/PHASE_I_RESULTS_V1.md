# MIST-COVE Phase I v1 results — synthetic only

Implementation and synthetic correctness are complete. **H1’s distinguishing superiority criterion failed in all three seeds.** This experiment does not justify conditional availability over simpler selectors, clinical usefulness, or improved transfer. H2 aggregate direction/donor checks and H3 constructed substitution checks passed, with substantial attribution limitations described below.

- Date: 2026-10-18 (Asia/Dhaka). Authorization: user Phase I execution request; synthetic code/training/tests only.
- Design ancestor: `ac842955fdee1b911bb0ba8a82aed39abef4bbb3`.
- Tested implementation: `533df355dc07715b042de92896b2d9c21cd6f653`; later report commit changes no model code.
- Branch: `research/mist-cove-implementation-v1`; PR base: `research/mist-cove-design-v0`.
- Config file SHA256: `66a6d51ebd121ffdd36ce3b72a1550c91a0282bd6df57fd463e7582f1a85cee8`.
- Config canonical SHA256: `0c143fb82bf16a2066c8e0b318c403f66f89f5e4411f5099652ac89e201d8517`.
- Frozen split fingerprint remains `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84`; split files, frozen P3B code/config/results are unchanged. No real split or descriptor loader was executed.
- [Complete aggregate synthetic receipt](../../reports/mist_cove/synthetic_v1.json) includes hashes for every implementation file, per-seed conditions/controls, metrics, actual updates, resource measurements and all failed scientific checks.
- [Implementation, baseline-interface inspection and reproducible commands](PHASE_I_IMPLEMENTATION_V1.md).

## Comparisons

Each cell lists seeds 123 / 456 / 789. Each run used 150 generated training examples, 180 full-batch optimizer updates, and 150 fresh generated evaluation examples per condition. There were 18 runs / 3,240 updates. All generated class codes, nuisance patterns and amplitude ranges are artificial.

| Model | Clean accuracy | A removed accuracy | B removed accuracy | Both removed mean confidence | Active parameters |
|---|---|---|---|---|---|
| full | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.240 / 0.258 / 0.258 | 75 |
| background_only | 0.913 / 0.953 / 0.953 | 0.813 / 0.787 / 0.860 | 0.773 / 0.820 / 0.787 | 0.274 / 0.315 / 0.298 | 40 |
| q_one | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.321 / 0.355 / 0.321 | 75 |
| q_only | 0.200 / 0.200 / 0.200 | 0.200 / 0.200 / 0.200 | 0.200 / 0.200 / 0.200 | 0.200 / 0.200 / 0.200 | 5 |
| fixed_event_score | 0.927 / 0.940 / 0.953 | 0.807 / 0.807 / 0.827 | 0.767 / 0.793 / 0.780 | 0.336 / 0.357 / 0.483 | 45 |
| fixed_patch | 0.980 / 0.993 / 0.980 | 0.893 / 0.907 / 0.913 | 0.860 / 0.920 / 0.867 | 0.259 / 0.277 / 0.256 | 75 |

All models instantiate 75 parameters; background-only, q-only, and fixed-event-score use fewer active parameters. q=1 and fixed-patch are controls with the full head capacity and identical update counts. q-only is at chance (0.200); its prediction collapse is retained in the receipt. The unchanged historical raw_context/evidence checkpoints were not loaded or evaluated.

## H1: not supported by the distinguishing comparison

The candidate classified clean and single-witness-loss examples perfectly, and mean confidence fell to 0.240–0.258 when both witnesses disappeared. q=1 achieved the same classification accuracy. Thus, these fixtures show evidence use but do not establish that conditional gating is necessary.

Availability failed the predeclared requirement to reduce risk by at least 0.01 against both entropy and global RMS at every 95/90/80% coverage point. The mixed pool contains clean, both-erased, and equal-L2 nuisance-corrupted examples. The full receipt includes full coverage, all class retentions, confusion, fixed-five F1, per-distinction margins, observed/fixed-five balanced accuracy, kappa, NLL, Brier and ECE15.

| Seed | Selector | Risk at 100% / 95% / 90% / 80% coverage |
|---|---|---|
| 123 | availability | 0.2844 / 0.2570 / 0.2222 / 0.1417 |
| 123 | negative_entropy | 0.2844 / 0.2547 / 0.2148 / 0.1444 |
| 123 | global_rms | 0.2844 / 0.2523 / 0.2173 / 0.1389 |
| 456 | availability | 0.2556 / 0.2266 / 0.1877 / 0.1250 |
| 456 | negative_entropy | 0.2556 / 0.2243 / 0.1975 / 0.1306 |
| 456 | global_rms | 0.2556 / 0.2336 / 0.2049 / 0.1278 |
| 789 | availability | 0.2622 / 0.2453 / 0.2173 / 0.1500 |
| 789 | negative_entropy | 0.2622 / 0.2383 / 0.2025 / 0.1333 |
| 789 | global_rms | 0.2622 / 0.2430 / 0.2198 / 0.1389 |

The largest calibrated-gain logit difference was 0.000001431. This verifies a known measurement inverse; it does not demonstrate robustness to unknown calibration errors. Matched non-erasing corruption reduced mean candidate confidence slightly (about 0.993 to 0.989), rather than increasing it.

## H2: direction checks pass; contribution magnitude is not causal necessity

Primary audits cover 15 examples per seed under each of two donors (30 comparisons per seed). Selected-minus-control effects exceeded the 0.05 criterion. Mean donor disagreement was 0.009–0.020, below the 0.2 criterion. Controls were 200 samples long, with about 1.6–1.9% mean energy mismatch. Separate exploratory probes randomized among multiple matched nuisance intervals and tested similar-looking genuinely nonselected redundant intervals.

| Seed | Selected true-stage delta | Control delta | Claimed contribution | Claim/delta MAE | Claim/delta correlation |
|---|---|---|---|---|---|
| 123 | 1.624 | -0.307 | 1.305 | 0.325 | 0.085 |
| 456 | 1.571 | -0.246 | 1.240 | 0.332 | 0.411 |
| 789 | 1.743 | -0.376 | 1.230 | 0.518 | 0.122 |

Exact additive reconciliation is a bookkeeping property. In the repeated-witness probe, claimed contributions of 1.212–1.289 produced selected-replacement deltas of −0.020 to 0.036 because selection substituted another interval. A similar nonselected interval sometimes had a larger effect than the selected interval. Claim/effect correlations are weak, and background changes contribute to total effects. Consequently, the broad claim that advertised magnitudes measure single-window causal necessity is **not established**. Positive/negative direction tests and donor guards pass; faithful physiology and distribution-valid replacements remain untested.

## H3: constructed substitution works, but does not establish a gating advantage

With A removed, B contributed essentially 100% of positive true-stage event support; with B removed, A contributed essentially 100%. Single-loss accuracy was 1.000 in every candidate run. Both lost reduced confidence below the predeclared 0.5 ceiling. q=1 also handled these fixtures perfectly, so substitutability alone does not establish conditional-availability value. Distinct generated carriers in the same spectral band are much simpler than real alternate physiological structures.

## Correctness, resources and failures

- Metadata design validator: PASS; 9 experiment definitions / 4 hypotheses, no data access or training.
- Required design contract suite: **11 passed, 0 failed, 0 skipped**.
- New MIST-COVE synthetic correctness suite: **53 passed, 0 failed, 0 skipped**.
- Full CPU regression suite, including new tests: **253 passed, 0 failed, 5 skipped, 1 existing warning** (5.24 s final implementation run). Skips: four historical CUDA exact-resume fixtures and one historical CUDA numeric fixture. Warning: existing PyTorch nested-tensor configuration. New model GPU checks and GPU resource measurements are NOT TESTED.
- Initially the default-shell pytest command was unavailable; the repository virtual environment resolved it. Initial new tests had 3 failures / 49 passes from sign assertions applied to intentionally zero-score ablations; assertions were corrected. An inspected q-only implementation bug retained background; it was corrected before the reported experiment. Initial and corrected config hashes and the unchanged criteria are documented in the implementation note. No software failures remain.
- Corrected experiment: **43.68 s CPU wall**, **568,548 KiB process peak RSS** (includes Python/PyTorch and setup; not GPU memory). Environment: Python 3.12.9, PyTorch 2.5.1, NumPy 1.26.4, pytest 8.4.2; deterministic FP32 CPU, 2 threads, CUDA hidden.
- Every model had 180 realized optimizer steps; no early stopping or hyperparameter sweep. Only full-batch synthetic training occurred. No weights, waveform arrays or participant records are committed.
- No real source TRAIN/VAL EEG, RESERVED SC data/annotations/descriptors/statistics, ST, or SHHS were accessed. Regression EDF/NPZ fixtures are generated under temporary test directories. No real-data normalization was fitted, no frozen split changed, and no historical checkpoint or metric was altered.

## Remaining gates and exact next steps

I1 software correctness is PASS. I2 experimental infrastructure is complete, with H1 FAIL and limited H2/H3 synthetic findings; it is not an all-hypotheses acceptance receipt. Source/external scientific outcomes are NOT TESTED. S0/R0/T0/T1 remain BLOCKED pending separately explicit authorization and their required proofs.

1. Record a scientific decision on the failed H1 distinction and attribution limitations: retain this restricted candidate for a source feasibility comparison or preregister a revised synthetic mechanism. Do not present the present results as an availability benefit.
2. For a separately authorized source-only experiment, freeze a new source config and runner identity. Preserve the exact TRAIN/VAL subject roles and verify split digests, source hashes, montage, calibration, physical indices and scoring masks before opening permitted files; use only `load_development(frozen, "train"/"val")`. Reject RESERVED/ST/SHHS before file access.
3. Reuse verified TRAIN-only normalization. No exemplar retrieval is needed; if added, fit/retrieve from TRAIN only. Implement the standalone radius-zero adapter/runner, full-grid VAL inference, participant-mean fixed-five F1 earliest-best selection, atomic provenance and exact interrupted CUDA resume; verify these contracts synthetically first.
4. Use the frozen future source budget: seeds 123/456/789; AdamW lr 3e-4, wd 1e-4; 60 max epochs, patience 12; core/encode batch 16, clip 5; deterministic FP32 with AMP/TF32 off. Prespecify candidate, q-free/background and unchanged raw_context/evidence comparisons and disclose capacity/realized-compute differences.
5. Run an authorized dry preflight, then measure feasibility/VRAM on the actual RTX 4080 SUPER 16 GB and stop on provenance, numerical, resume or resource failures. Commit/freeze the source implementation before any real training. Source-only completion still does not unlock RESERVED, ST, or an independent external cohort.

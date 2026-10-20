# Phase XI-B prospective readiness protocol v1

Status: **NOT READY TO LAUNCH**. This is a proposal only. XI-A has not authenticated or loaded the original trained checkpoint. Its metadata and generated-fixture results do not authorize trained-artifact access or real EEG.

The candidate remains original P3B `raw_context`, seed 123, source commit `661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23`. Phase III and any prospective reproduction retain separate experiment identities. First follow the receipt-only step in [the restricted artifact plan](PHASE_XIA_RESTRICTED_ARTIFACT_PLAN_V1.md). Then obtain explicit permission for the selected checkpoint bytes and a separately reviewed restricted loader. The original best payload contains a fitted TRAIN waveform bank; approval must cover that embedded content even if inference does not use its values. No fallback to unrestricted pickle is permitted.

## Required entry conditions

Independently authenticate the original root and selected completion receipts; recover the selected serialized-byte digest; verify the actual candidate against it before parsing. Establish exact source/configuration, split, normalization, TRAIN bank, environment and creation/completion lineage. Where an authenticated metadata reference remains missing, report it and stop the dependent operation. Preserve byte/state/configuration digests as distinct identities. Actual restricted-load compatibility is untested, not assumed from the installed PyTorch version.

Construct the exact original architecture from authenticated source and configuration without fitting, resume or parameter edits. Validate strict state keys, tensor shapes/dtypes/finiteness, buffers, the original named-state digest and source feature order. Freeze evaluation mode and use deterministic FP32 with recorded software versions and fixed generated-input streams. Hash state before and after testing. No input-dependent fitting, bank reconstruction, source crop selection or normalization updates are allowed. Fitted-value inspection requires explicit permission and must not be published.

## Generated-input tests only

Use two prospectively fixed generated 3,000-sample mock inputs; label them generated and make no physiological inference. Check the existing Phase X adapter's 169 windows, lengths 100/200/400, stride 50, 209 feature pathways and seventeen whole-epoch features against the original constructor. Verify feature ordering explicitly.

Compare the original forward path with full availability on the same authenticated state. Record exact bitwise equality separately from the proposed maximum absolute FP32 tolerance of `1e-6`; a tolerance result must not be described as bitwise identity. The final authorization should freeze these criteria before testing. Check hidden-sample noninterference by changing withheld samples only and comparing all exposed features and outputs, including global pathways. Record unchanged state hashes. A mismatch fails the affected identity claim; do not alter weights or acceptance criteria to force success.

For the bounded fifteen-atom interface, generate the complete 32,768-mask response table for each of the two fixed inputs and its fixed generated nuisance view: at most 131,072 adapter/head queries, with neutral/full masks included in the tables. Cache only verified identical feature computations; report feature and full-network work separately. Verify every finite Boolean response against its fixed-order circuit, with no monotonicity assumption. Propose a 120-second CPU ceiling, measured before the confirmatory table run; if the resource pilot exceeds it, stop and request a revised budget rather than truncate and call the result exhaustive. This is a future budget, not consumed XI-A compute.

The neutral state is the defined internal availability operator. It is not an observed physiological baseline, and prior-only predictions remain possible. Generated-input correctness does not establish clinical interpretability, actual EEG localization, deletion compatibility or robustness. Preserve wrong predictions and noncompact circuits without changing the fixed examples.

Optional CUDA correctness requires separate authorization and an idle device; never interrupt another process. Restrict it to the same two generated inputs and a fixed small mask subset, with CPU/CUDA agreement reported separately from bitwise identity. No GPU activity is proposed for XI-A.

## Separate future decisions

| Stage | Scope | Authorization needed |
| --- | --- | --- |
| Receipt-only follow-up | Two bounded hash-pinned original completion JSONs | Separate read-only receipt permission |
| XI-B | Authenticated checkpoint loading and generated-input adapter tests | Artifact, embedded fitted-bank content and reviewed loader/test permission |
| XI-C | Two-example actual TRAIN-only resource pilot | New source-data authorization and frozen narrow protocol |
| XI-D | Up to 150 original TRAIN epochs | Another review after XI-C, with resource and scientific gates |

No stage grants the next. XI-B has zero neural fits/optimizer updates and excludes real EEG, annotations, descriptors, RESERVED participants, external cohorts, legacy NPZ, adaptation and source evaluation. Failure to recover a trusted historical byte reference blocks P3B recovery. A separately authorized prospective reproduction could test a different trained model, but cannot be called the original P3B checkpoint.

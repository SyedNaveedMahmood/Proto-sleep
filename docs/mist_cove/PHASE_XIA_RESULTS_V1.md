# Phase XI-A results v1 — historical checkpoint authentication

**Historical P3B authentication remains BLOCKED.** All safe metadata, security engineering and planning work is complete. Actual checkpoint bytes, fitted artifacts, private historical directories and real EEG were not opened. The investigation recovered a useful original completion-root reference that was not established in Phase X; it does not itself authenticate the selected checkpoint.

Base: `0ce6d4235661c6bda57392040954abb506abb7b8`, tree `2e1f172907840aa4b941e3fcf3e3a1f3919f1ecb`. The ledger is `reports/mist_cove/phase11a_v1/progress.json`. Historical results, source and the frozen Phase IX plan remain unchanged. No Phase XI-B/C/D work was launched.

## Q1. Is the original P3B checkpoint independently authenticated?

No. The selected original `raw_context`, seed 123, has metadata-level evidence, but no actual serialized-byte comparison or restricted load has occurred. A published named model-state digest cannot authenticate arbitrary serialized content. Synthetic fixture success cannot promote the historical gate.

## Q2. Which original references can be verified?

The audit verified 61 pinned historical record observations and searched 78 historical blobs across 64 explicitly approved source/publication paths. Git commit/blob framing and ancestry were checked. The original publication commit is `15995b31821ee8db717f3d6edfa41166de65d4a3`, tree `328f2848ab3ca169b90b006a6100226332774f40`; the GitHub API agreed with local content. It is unsigned. The user-pinned Git history is the declared content trust root; neither timestamps nor two files in one publication are independent signing/time attestations.

| Typed reference | Independently preserved value or finding | Evidence limit |
| --- | --- | --- |
| Original source | `661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23` | Recoverable Git content; not trained weights |
| P3B run | `a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b` | Published selected run only |
| Selected named model state | `1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d` | Not a checkpoint-byte hash |
| Split-file reference (directly preserved in Phase III publication; shared split fingerprint matches P3B) | `1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5` | Actual split file not opened |
| Split fingerprint | `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84` | Original field is split identity, not a verified normalization artifact |
| Original root COMPLETE bytes | `905ba78cf8326594a431ce7a71ce9624c35814d2e76c94f57abda8afbe7c7e7d` | Root/child JSON contents uninspected |
| 42-file implementation manifest | `2e306bf630547b66d8c273d1999d9fe2772766a404316691b65db7d23c6046b3` | Reconstructed exactly from original Git objects |
| Nine-file reused-source manifest | `9d7e52495bc985d64436957f16c5cc0705c9b25a7560f543ed9d252ef21a038e` | Reconstructed exactly from original Git objects |
| Original correctness receipt reference | `06015bbca8214121e92f7eef998e6ef4e226f3c93346861de496926fdfc5c4b8` | Published reference; private receipt uninspected |

The root hash appears in original `reports/p3b/FINAL_COPY_PASTE.txt` and `validation_summary.json`. Original `full_experiment.run_full` code binds per-run VALIDATED_COMPLETE markers and their output hashes, including selected `best.pt`. This establishes a possible transitive historical chain. It is inaccurate to conclude that no original byte evidence exists. The selected direct checkpoint-byte value has not been materialized in the approved corpus. The search does not prove absence elsewhere.

The graph has fourteen nodes and thirteen explicit edges; structural validation passes. Level 1 means local self-consistency; Level 2 means independently pinned historical reference content; Level 3 would require the actual authorized byte comparison. There are zero historical Level 3 edges. The graph and lifecycle report describe missing links and falsifiers individually.

## Q3. Does the Phase III seal authenticate P3B?

No. Phase III implementation `a601e8e004b9e0c2942e83d4517fb8a47856de72` and publication `4095c805cfd48fbb1e9ef938dbc46da28ed1dfb1` describe a separate experiment. Its selected seed-123 COMPLETE_SEAL reference is `a7747559ecc838307c342e09f226e536356ce6231f1b64ca58128fbd344b169b`. Matching architecture, seed, reported metrics or duration does not establish weight or byte identity. No candidate substitution occurred. The full comparison table is in the historical-record audit.

## Q4. Are source and scientific configuration recoverable?

Original source and tracked scientific configuration are recoverable and their published implementation manifests reproduce exactly. The frozen code establishes construction, TRAIN fit roles, training, selection, saving, resume, completion and aggregate publication contracts. That is distinct from proof that a private artifact was created under those exact conditions. Detailed original environment, creation, TRAIN normalization and bank linkage remain partly uninspected. The published fingerprint beginning `3e7a` must not be relabeled a scaler digest.

## Q5. Is actual restricted loading safe now?

Not established. Original `load_trusted` uses `torch.load(..., weights_only=False)`; its payload can contain a fitted TRAIN waveform bank. Historical checkpoint-byte verification must precede parsing, and a separately reviewed loader must consume immutable authenticated bytes with strict schema, configuration, named-state and buffer checks. Actual `weights_only` compatibility has not been tested. No unsafe fallback or real-artifact CLI unlock exists. The security review and restricted artifact plan specify allowlists, no-follow regular-file handling, independent receipt trust, size limits and race-safe snapshots. These supported interfaces are not an OS sandbox against arbitrary hostile Python.

## Q6. Can the Phase X adapter be trusted without trained weights?

Phase X supports mock architecture/full-input identity, hidden-sample noninterference and exact finite circuits for its defined operator. It does not establish behavior on the actual trained P3B state. XI-A adds provenance tooling, not trained-model explanation evidence. No clinical, robustness, physiological-neutrality or source-transfer claim follows. Existing Phase IX/X negative findings remain unchanged.

## Q7. Is a TRAIN-only EEG pilot authorized?

No. There was zero real EEG, annotation, data-bearing descriptor, RESERVED/external cohort, real bank/scaler or real checkpoint access; no real inference, fitting, training or adaptation. XI-B, XI-C and XI-D remain independent future decisions.

## Q8. What is the smallest next approval?

Read only the original root completion JSON pinned to `905ba78c…` and the selected P3B seed-123/raw_context VALIDATED_COMPLETE JSON pinned by that root. Limit both to 2 MiB, read-only hash-before-parse, strict schema and selected logical digest extraction. Exact private locations must be supplied separately, without directory exploration. This may recover the original selected byte reference. It does not permit checkpoint, bank, scaler or EEG access, and does not authenticate the candidate. Later byte hashing and restricted loading need further approval. See [the concrete procedure](PHASE_XIA_RESTRICTED_ARTIFACT_PLAN_V1.md).

## Independent gates and generated security evidence

| Gate | Status | Meaning |
| --- | --- | --- |
| XI-A0 repository | PASS | Git/source manifests verified |
| XI-A1 run | PASS | Published selected-run identity only |
| XI-A2 split | PASS | Published split reference only |
| XI-A3 training lineage | INCONCLUSIVE | Detailed original creation/fit/environment links missing |
| XI-A4 historical byte reference | INCONCLUSIVE | Root anchor exists; selected direct byte value unmaterialized |
| XI-A5 actual artifact bytes | BLOCKED | Outside XI-A authorization and entrypoint |
| XI-A6 restricted loading | NOT TESTED | No actual artifact loaded |
| XI-A7 trained adapter | NOT TESTED | Future XI-B |

All twenty required generated scenarios returned expected decisions: one valid chain PASS, fourteen intentional FAIL and five BLOCKED. Duplicate and contradictory records were both rejected. Tampering, rewritten local sidecars, state-versus-byte confusion, wrong seed/variant/source/normalization/bank, missing references, unrelated markers, symlinks, incomplete writes and TOCTOU changes fail closed. Authenticated unsafe pickle fixture bytes remain BLOCKED at parsing; the execution tripwire was absent. No unsafe deserializer was called. The fixture lab creates its own fresh namespace and offers no caller-supplied artifact path.

## Resources and regression validation

The scoped metadata audit took 1.066441875998862 seconds, below its 60-second ceiling. The twenty-case generated verification run took 0.004288673997507431 seconds, below its 10-second ceiling. Its interpreter process high-water RSS was 378,364 KiB; this is not incremental verifier memory. Forty-nine added tests cover strict provenance, generated security cases, Git identities, gate enforcement and safe publication. The final complete CPU suite passed 1,022 tests with 11 CUDA skips and zero failures in 19.73150334499951 seconds (pytest: 18.83 seconds). An initial full run before five additional reference-drift tests passed 1,017 tests in 20.0940092880046 seconds; total measured full-suite wall time was 39.82551263300411 seconds. Final child-process high-water RSS was 896,720 KiB. Python 3.12.9, torch 2.5.1, NumPy 2.0.1, pytest 8.4.2, MNE 1.10.2 and pyEDFlib 0.1.42 were used locally. Exact pinned CI dependencies are checked independently by the unchanged workflow. Log/JUnit digests and measurements are in `correctness_receipt.json`; final commit and CI links are recorded in the draft PR closeout. No GPU was queried or used. There were zero new scientific neural fits or optimizer updates; any established generated training/resume unit tests are regression work, not a research campaign.

## Decision and remaining uncertainty

Preserve the useful original root reference and pursue the two-receipt permission before requesting checkpoint bytes. Do not switch to Phase III because its seal is easier to inspect. If the original chain cannot be recovered, historical authentication stays blocked; a prospective reproduction would be a separately authorized new model identity. Finish XI-A publication and stop for review. This package demonstrates verifier behavior and recoverable historical references, not original-weight authenticity or clinical explanation validity.

## Publication closeout

All twelve authorized work packages are complete; completing a provenance investigation does not mean the historical artifact gate passes. Draft PR [#13](https://github.com/SyedNaveedMahmood/Proto-sleep/pull/13) targets the Phase X branch. The unchanged CPU workflow passed [run 39448706371](https://github.com/SyedNaveedMahmood/Proto-sleep/actions/runs/39448706371) on `33624493a9cb053179f7d6867e76733b58980ba5`, with 1,022 passed and 11 CUDA skips. Its job took 83 seconds including setup. The final closeout changes only reports, documentation and their publication hashes; the final commit and its own exact-head CI result are recorded externally in the PR after push, avoiding a self-referential commit digest. No later phase is launched.

# Phase X source provenance review v1 — authentication BLOCKED

This review reads committed source definitions and published safe aggregate Git blobs only. No checkpoint, bank, normalization value, real descriptor, annotation, EEG directory or recording was opened, hashed or deserialized. Current checksums of committed reports prove only report integrity.

Prospective first candidate: **original P3B raw_context seed123**, selected to fix identity before any new real scores. Its source implementation commit is `661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23`; run fingerprint `a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b`; published best model-state digest `1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d`. This state digest is not a serialized checkpoint byte digest. A newly hashed file cannot become independent historical authentication.

| Requirement | Published/code evidence | Unresolved before artifact access |
|---|---|---|
| Variant/seed | P3B raw_context123 selected; epoch-local no CRF required | prove actual checkpoint cfg/feature order after pre-deserialization authentication |
| Frozen split | fingerprint3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84; published file SHA1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5 | bind authenticated original run to exact frozen role/source proofs |
| Source code/config | frozen implementation661dbb3; committed p3b config specifies deterministic FP32, radius0, TRAIN fitting | original run provenance/signature/implementation inventory, no current-HEAD bypass |
| Environment | code/Phase III aggregate document Python3.12.9, torch2.5.1, NumPy1.26.4 and deterministic/TF32 settings | Phase III is distinct; original P3B environment and compatible runtime evidence must be verified independently |
| Normalization | fixed all-physical TRAIN affine and amplitude statistics described in source contract | original authenticated normalization digest, TRAIN roles and exact lineage; no values inspected |
| Waveform bank | TRAIN-only medoids per seed, common within-seed bank; raw path uses dimensions not waveform values | original bank digest, cfg, shape and TRAIN lineage; unused values do not waive provenance |
| Creation/boundary records | frozen trainer writes provenance, epoch seals, complete/validated markers | authenticate original records against independent historical references before trusting their claims |
| Checkpoint bytes | original published best_state_digest available | independently pinned original serialized best.pt digest/validated-marker chain not established by this metadata audit |
| Associated state | P3B best state digest recorded above; 34,197 total /33,957 trainable parameters | authenticated bytes first, then safely verify actual named tensor/state identity and exact construction |

There is useful separate evidence: committed Phase III aggregate rows publish COMPLETE_SEAL digests for all three raw_context fits. The collector source explicitly verifies seals before loading and publishes the marker digest. A separately authorized phase could authenticate those marker bytes against the published SHA, then their sealed output digests before deserialization. Those are separately identified Phase III fits; equal source-VAL metrics do not establish equality to original P3B weight bytes. Switching to that route requires explicit candidate/lineage review, not silent substitution.

Future verification procedure, **not executed here**:

1. Obtain separate narrowly scoped artifact-access authorization, exact candidate identity, trusted historical references, permitted roots and privacy plan. No EEG access follows automatically.
2. Verify the original signed/sealed provenance and immutable implementation/config/environment/split identifiers against independently preserved references. A self-consistent newly generated manifest is insufficient.
3. Authenticate normalization/bank/creation and completion markers by their independently pinned historical digests. Require TRAIN-only lineage; do not inspect private descriptors or values without their separate permission.
4. Before deserializing a checkpoint, match its bytes to a digest from the authenticated historical chain. A model-state digest alone cannot protect pre-load pickle execution. Reject missing evidence; never call historical unrestricted load_trusted on unauthenticated bytes.
5. Under the separately reviewed loader, verify named tensor identities against the historical model-state digest, complete config/features/buffers and required environment. Preserve model eval/FP32 semantics. Do not fit or retrain anything.
6. Repeat mock dependency/identity/noninterference gates for the exact authorized construction. Any unknown global or temporal path blocks explanation claims. Only a further authorized actual-input resource pilot can precede a frozen TRAIN audit launch.

Gate X-D is a completed metadata review, with actual checkpoint authentication **BLOCKED**. Missing evidence and uninspected artifacts are not evidence of corruption. The current API has no checkpoint loader or real-data release switch.

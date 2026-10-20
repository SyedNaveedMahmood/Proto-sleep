# Phase XI-A historical record audit v1

## Search boundary and trust

The scoped audit examined78 historical Git blobs across64 explicit approved code/report paths, including original P3B reports/config/runtime and Phase III public receipts/collector. It did not scan arbitrary filesystem roots or open any ignored historical experiment directory. Full corpus identities and61 pinned record observations are in `evidence_inventory.json`. Negative search results apply only to this corpus, not every possible private backup.

Commit objects and blob objects were recomputed using Git SHA1 object framing. Original commits661dbb3/15995b3 and Phase III a601e8e/4095c805 were verified ancestors of the user-pinned Phase X base. GitHub's publication commit/tree matches local15995b3 (tree328f2848ab3ca169b90b006a6100226332774f40); its API reports unsigned. This confirms content/ancestry under the pinned Git trust root, not an external signer or trusted timestamp. Commit author dates are historical claims. Two files in one commit are not two independent experiments or archival witnesses.

## Stronger P3B evidence recovered

Original publication15995b31821ee8db717f3d6edfa41166de65d4a3 contains:

| Typed reference | Value / source | Current meaning |
|---|---|---|
| Root completion-marker-byte SHA256 |905ba78cf8326594a431ce7a71ce9624c35814d2e76c94f57abda8afbe7c7e7d; FINAL_COPY_PASTE line6 and validation_summary.root_completion_sha256 |Level2 reference to original P3B root marker, not checkpoint bytes |
| Run fingerprint |a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b; results/summary |Original P3B experiment reference; actual candidate payload not checked |
| Selected best model-state digest |1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d; raw_context123 row |Tensor-state reference; cannot protect pre-load pickle parsing |
| Source implementation hash |2e306bf630547b66d8c273d1999d9fe2772766a404316691b65db7d23c6046b3 |42 original tracked source/scripts file hashes reconstructed exactly from661dbb3; matches publication |
| Reused implementation hash |9d7e52495bc985d64436957f16c5cc0705c9b25a7560f543ed9d252ef21a038e |Nine reused source-file hashes reconstructed exactly; matches publication |
| Correctness receipt SHA256 |06015bbca8214121e92f7eef998e6ef4e226f3c93346861de496926fdfc5c4b8 |Published original receipt reference; actual private receipt not opened |
| Split fingerprint |3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84 |Original config/aggregate call this split_fingerprint; not an authenticated normalization fingerprint |
| Split-file SHA256 |1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5 |Published frozen split-file reference; actual file not opened here |

This corrects an incompleteness in the Phase X metadata review without changing its historical report: there is a published original P3B **root completion hash**, even though no selected direct checkpoint-byte value was materialized in that review. The frozen root writer explicitly includes selected per-run VALIDATED_COMPLETE marker hashes and their outputs, including best checkpoint bytes. A two-JSON receipt-only investigation could therefore recover an original byte reference if the actual root/selected markers match their anchors. Their contents and present existence are not inspected in XI-A.

The root marker itself is not in tracked publications; its SHA is. No direct selected best.pt serialized-byte SHA was found in the scoped approved published records. Gate A summary contains resource/model counts, not an independently published selected epoch28 checkpoint digest. Missing/uninspected evidence is not corruption. No present-day candidate checksum is calculated or promoted to historical evidence.

## Separate Phase III records

Publication4095c805cfd48fbb1e9ef938dbc46da28ed1dfb1 publishes COMPLETE_SEAL references for raw_context123/456/789. For123 the seal SHA is a7747559ecc838307c342e09f226e536356ce6231f1b64ca58128fbd344b169b. The collector code verifies the seal before loading and publishes its file digest. That is another Level2 marker reference with a potential original output-map chain. It does not authenticate original P3B. No seal contents or model bytes are opened, and no lower-friction substitute is selected.

Current conclusion: source/config Git identities are recovered; original root-marker reference is historically anchored; detailed TRAIN-fit/run/creation links and direct selected byte reference remain uninspected/inconclusive; actual selected artifact authentication is BLOCKED. Smallest useful next permission is bounded read-only inspection of the pinned P3B root completion JSON and exact selected child completion JSON, without checkpoint/bank/normalization/EEG bytes.

---

# Phase XI-A P3B versus Phase III lineage comparison v1

This table is part of the historical record audit, not a checkpoint substitution decision. Source definitions and already published aggregates are inspected; no actual artifacts are opened.

| Property | Original P3B raw_context123 | Phase III raw_context123 |
|---|---|---|
| Source implementation |661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23 |a601e8e004b9e0c2942e83d4517fb8a47856de72; Phase II sealed wrapper; distinct launch/code inventory |
| Scientific protocol |P3B four baselines ×three seeds |Phase III four analytic models plus two historical-definition baselines ×three seeds |
| Run identity |a9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b |Distinct sealed runtime identity binds Phase III source/proposal/code/environment; private value not inspected |
| Construction |make_baseline(raw_context) from TRAIN bank |build_adapter(raw_context) delegates same historical definition; actual buffers uninspected |
| Dataset role |15 TRAIN/4 development VAL; no RESERVED |Same frozen role contract; full physical positions; no RESERVED |
| Initialization |seed_all123 before construction; seed123 TRAIN bank reused from original fit contract |same seed/definition intended; matching initialization/weight bytes not established by source inspection |
| Recipe/selection |AdamW3e-4/wd1e-4; max60/patience12; earliest best mean participant F1 improvement>1e-8 |same declared recipe; sealed wrapper/provenance differs |
| Duration |40 epochs; selected epoch28;199120 updates |40 epochs; selected epoch28;199120 updates |
| Selected state digest |1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d published |Not directly published in selected safe row; actual COMPLETE/state uninspected |
| Serialized selected byte SHA |Not materialized; published original root map may bind it |Not materialized; published COMPLETE_SEAL may bind it |
| Marker lineage |Root COMPLETE905ba78c…c7e7d → selected VALIDATED_COMPLETE/output map |Selected COMPLETE_SEAL a7747559…b169b → outputs, including best/last/completion/epoch seal |
| Environment |Original code/precision recoverable; full runtime environment uninspected |Published Phase III environment Python3.12.9/torch2.5.1/NumPy1.26.4; not proof of original P3B environment |
| Independent reference |Original Git publication15995b3 |Separate Git publication4095c805 |
| Current completeness |Level2 original root/state references; no actual bytes or detailed lineage verified |Level2 selected seal reference; no actual seal/weight bytes verified |

The published selected scores, best epochs and training durations match. That does not prove byte or weight identity, replay equivalence, bank identity or interchangeability. The Phase III collector explicitly compares scores and seals its own experiment; it does not publish a cross-experiment tensor-byte equality proof. Both chains could be useful, but neither is Level3 now. Prefer investigating the selected P3B root chain first. Choosing Phase III requires explicit scientific approval and its own identity; no substitution occurs in XI-A.

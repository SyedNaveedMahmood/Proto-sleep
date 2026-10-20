# Phase XI-A serialization and scientific-integrity review v1

## Historical observations

The frozen original runtime uses torch.save and torch.load(weights_only=False). The best payload contains named state, model config, fitted bank, signature, best epoch and metrics. Last also contains optimizer/history/Python/NumPy/torch RNG. Merely opening a best checkpoint may expose real TRAIN waveform-bank values, even though raw_control ignores their values during inference. No such file is opened here.

PyTorch2.5.1 exposes a weights_only argument; the original loader explicitly disables it. A restricted loader's compatibility with the actual best bank/metadata payload is NOT TESTED. The last checkpoint's non-tensor RNG objects make blanket unrestricted loading especially inappropriate. Restricted loading is not historical authentication, and a valid checksum is not proof arbitrary pickle execution is safe. Never fall back to weights_only=False or broad allowlisting because a payload fails.

Original resume and postfit code checks signatures/seals and state digests, but some state/schema checks occur after unrestricted deserialization. Locally rewritten sidecars are not independent proof. Output maps must be reached from an independently pinned original root; current filenames, tensor shapes and matching metrics are insufficient. The original tensor_digest hashes names and tensor bytes without explicit shape/dtype metadata; future schema checks must independently validate those. Bank/amplitude buffers, feature order, normalization and pointwise head configuration can change explanations even without changing nominal model name.

Atomic temporary-file+fsync+replace reduces incomplete writes but does not authenticate an author, prevent adversarial replacement, freeze an open path or prove original chronology. P3B's writer lock and Phase III's per-run lock address cooperating writers, not attackers. Original output-path joins and ordinary opens follow symlinks; P3B verify_outputs does not itself enforce a confined cryptographic content store. Phase III's basename check narrows traversal but ordinary file opens can still follow symlinks. Private provenance may include absolute paths/source-linked metadata; public XI-A output must not copy them.

## Threat model and required controls

| Threat | Failure example | XI-A generated-fixture control / future requirement |
|---|---|---|
| Accidental identity corruption |wrong seed, variant, source commit, split, normalization or bank |typed exact experiment identity checked at every linked receipt; missing evidence blocks |
| Wrong experiment substitution |Phase III seal offered for P3B |explicit experiment namespace; no architecture/metric equivalence shortcut |
| State versus serialization confusion |same state digest, swapped outer payload/optimizer |independent original serialized-byte SHA mandatory before parsing |
| Rewritten local records |modified bytes and newly matching sidecar |in-memory independently retained fixture anchor pins original receipts; unanchored present-day records cannot mint trust |
| Forged historical assertion |today's checksum labeled historical |trust origin never accepted from untrusted JSON flags/timestamps; historical Git publication is a separate pinned reference |
| Missing/incomplete writes |truncated candidate or absent creation/completion |bounded sizes, complete required schemas/digest chain, explicit BLOCKED/FAIL reasons |
| Traversal/symlink/special-file redirection |absolute or../names, outside symlink, FIFO |fresh owned fixture directory capability, fixed basenames, directory-relative no-follow/nonblocking opens, regular-file checks; no caller source path |
| TOCTOU |candidate changes after hash before load |verified immutable snapshot; pre-load descriptor/path identity and digest recheck; parser consumes the verified bytes, never reopened pathname |
| Unsafe deserialization |pickle reduce executes side effects |new library has no pickle/torch loader; safe bounded generated JSON envelope only; executable fixture rejected without execution |
| Contradictory/duplicate records |duplicate JSON keys, mismatched run/root children |strict duplicate-key/schema/type checks and deterministic graph validation |
| False gate promotion |fixture PASS or metadata PASS offered as real auth |fixture-only domain and phase-specific real gate permanently blocked without a separately reviewed actual-artifact tool |
| Sensitive publication |private paths/banks/checkpoint contents copied |aggregate allowlist/privacy validation; no candidate content, scaler values or private directory names exported |

The supported interface prevents accidental real-artifact reads; it is not an OS sandbox against arbitrary malicious Python or a trusted-host compromise. A loader can detect observed changes before consumption, but cannot promise to observe every transient external mutation; immutable authenticated-byte consumption prevents those races from substituting unverified input. Root-reference authenticity remains an explicit trust assumption; SHA consistency cannot establish provenance if the root itself is forged.

The current CLI accepts no source/checkpoint/root-directory path or authorization override. Future artifact access requires separately reviewed entrypoints and user decisions, path-free logical allowlists mapped privately, pre-deserialization historical-byte authentication and least-privilege processing. No real loading, conversion, source inference or GPU operation is justified by this review.

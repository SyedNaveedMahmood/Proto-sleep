# Frozen source model authentication

The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`.

All four actual local step-1,200 checkpoint byte hashes exactly match the user-specified M0/M2 seed-123/456 identities. Formal loading uses inherited authenticated snapshots and the patched PyTorch 2.10 restricted weights-only loader, with no custom globals, unsafe fallback or optimizer-state restoration. Strict state shapes, arm, seed, final step, M0 protocol hash, 64-by-64 prototype bank, finiteness and 41,429 parameters pass. See checkpoint_receipt.json.

The ten-person source-fit normalization receipt is authenticated through the existing Phase XIII metadata chain: mean 0.2903136955214651 uV; SD 26.31764815403481 uV. No source waveform was read to recover it. For each authenticated checkpoint, generated [16,1,3000] inputs reproduce the unchanged original MISTSleep forward output exactly under patched CPU inference, with the entire state digest unchanged. Historical development performance was not recomputed. This establishes generated full-input compatibility, not new source or external accuracy.

Audit limitation: an initial exploratory metadata inspection used the older Torch 2.5.1 loader with weights_only=False after checking byte hashes. That inspection did not run target inference. Every formal authentication and source behavior check was subsequently performed through the inherited patched restricted loader. GPU correctness used a generated model only and did not deserialize source checkpoints.

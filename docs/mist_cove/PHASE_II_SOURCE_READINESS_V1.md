# Source software readiness and provenance checklist

This package prepares software; it grants **no real-data authorization**. Phase II uses generated arrays and mocked source proofs only. A mock launch receipt is not an approved source receipt. Source preflight itself opens permitted development descriptors and may read data for provenance checks, so it also requires separate source authorization.

## Prepared interfaces

`src/mist_cove/phase2/adapter.py` adapts the q=1 default, availability control, background and rounded selector to the existing `encode → emissions`, masked loss and prediction API. The same factory calls unchanged `make_baseline` for raw_context/evidence using a separately verified TRAIN waveform bank. No old checkpoint identity guard is bypassed and no historical accuracy is reevaluated.

`source.py` and `scripts/run_mist_cove_source_prepared.py` require an explicit source authorization flag **and** a separately reviewed launch receipt before entering the source path. The default CLI is preflight, not training. The frozen prepared JSON preserves all P3B recipe values, source split SHA256/fingerprint, channel/units/geometry, TRAIN-only reuse, earliest-best improvement >1e-8, and all forbidden-access flags. It includes all three seeds, q=1, availability, background, rounded selection, unchanged raw_context and unchanged evidence. A real-data comparison must not silently remove a failed synthetic refinement; a revised candidate list needs a new preregistered version before source scores.

`training.py` wraps the unchanged trainer/evaluator. Role/view rejection precedes file access. Training cores exclude unscored positions and split at physical gaps; full-grid validation runs every supplied physical position before metric masks. The live source runner additionally requires the original full physical grid/counts using `verify_grid`. A per-output nonblocking lock enforces one checkpoint writer. Each epoch atomically seals last/best/history/provenance digests before interruption; completed outputs are also sealed. Config, adapter, source indices/labels/masks, normalization/provenance, bank, implementation files, dependency/device environment and training recipe bind resume. Invalid seals and incompatible identities fail before trusted checkpoint deserialization. Only runner-created local, digest-verified checkpoints are loaded by the unchanged trusted loader; no untrusted checkpoint path is accepted as a resume state.

`traces.py` distinguishes raw source-file SHA256 from the inferred physical-input byte SHA256. It retains subject, recording/night, channel, units, original epoch, physical half-open sample/time interval and scoring mask. Its metadata interface rejects reserved/external roles and never opens a file. It is not permission to obtain or publish participant traces; real traces remain local/ignored after separately authorized source proof verification.

## Checks required before any future source execution

1. Obtain separate explicit source TRAIN/VAL authorization. Freeze the source candidate list and implementation after reviewing W1–W4 failures. Keep RESERVED/ST/SHHS prohibited.
2. Verify unchanged frozen split bytes `1162a41098bb8a724a0b8bf8bb320469c2b1c02bb802d9b5cf7f2e49d2da54e5` and fingerprint `3e7a7260aae50ca522752f1cede41246daf1b2f6a8fd482800cfda97b4564e84`. Use the existing frozen parser; never regenerate the split or inspect reserved descriptors.
3. Bind trusted normalization path/digest to its original completion receipt. Verify TRAIN-only fitted identities/source hashes, 79,984 physical TRAIN epochs, label-independent fitting, and unchanged source/stat proofs. Do not refit any statistic or fit VAL. Bind all three existing TRAIN bank artifacts to their trusted digests and seeds; validate original TRAIN crop provenance. The loader uses `weights_only=True` with **no unsafe fallback**. An incompatible historical bank serialization is a blocker requiring a separately reviewed conversion in its trusted environment.
4. Bind passing CPU and synthetic CUDA correctness receipts, exact interrupted model/optimizer/Python/NumPy/CPU/CUDA RNG recovery, and measured GPU feasibility to the **current** source/config implementation hash. Synthetic GPU memory is not a guarantee of real-data throughput. Recheck free GPU memory and other running work before source use; do not interrupt other jobs.
5. Populate an independently reviewed launch receipt: scope `SC_TRAIN_VAL_ONLY_AUTHORIZED`, authorization reference, canonical config SHA256, implementation SHA256, frozen split hashes, actual normalization/completion paths and hashes, per-seed TRAIN bank paths and hashes, and correctness artifact paths/digests plus explicit CPU/CUDA/feasibility PASS. Phase II scripts deliberately do not generate this authorization receipt.
6. Run an authorized preflight only, then review its result before training. Verify 15 TRAIN / 4 VAL / 55 RESERVED, 29/8 opened development recordings, physical TRAIN/VAL 79,984/21,796, scored 79,025/21,789, no participant/source overlap, calibrated Fpz-Cz 100 Hz/3000 samples and full position masks. No real preflight has been run in Phase II.
7. Preserve the maximum training recipe: seeds 123/456/789; AdamW lr 0.0003, wd 0.0001; max 60, patience 12; core/encoding batch 16; clip 5; deterministic FP32, AMP/TF32 off; radius 0, no CRF. Select earliest best participant-average fixed-five VAL Macro-F1 only for improvement >1e-8. Record realized epochs, updates, wall time, memory, class collapse and negative results separately for every model. Maximum budgets do not imply equal realized compute.
8. Keep all real checkpoints, source traces, participant predictions, bank bytes and console logs under ignored `mist_transfer_runs/mist_cove/phase2_source_v1/`. Source-only completion cannot unlock RESERVED or any external cohort.

The launch receipt is an integrity/operational gate, not a cryptographic substitute for human authorization. Its trusted digests and approval provenance must be independently reviewed. Mock integration replaces the loaders, full-grid verifier and TRAIN-bank provenance checker; it validates wiring and fail-closed behavior, not the correctness of any unseen real artifact. Existing regression tests independently exercise the unchanged proof and grid implementations on generated fixtures.

Environment audit found that loaded NumPy is 1.26.4 while installed distribution metadata reports 2.0.1 in the same prefix. Both values are preserved in the receipts; deterministic CPU/CUDA tests passed with the loaded module. Reconcile the environment and record actual imported versions before a source freeze, then regenerate environment-bound correctness proofs. Do not repair an existing frozen historical environment in place or silently reinterpret its version metadata.

## Reproducible software checks

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests -ra
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/check_mist_cove_phase2_cuda.py \
  --output-dir mist_transfer_runs/mist_cove/phase2_cuda_v1
```

Choose a fresh CUDA directory on repetition; output overwrite is refused. GPU checks use generated inputs and a bounded allocator, leaving existing GPU processes untouched. They raise on absent CUDA or insufficient free memory, never fall back to CPU. Actual hardware, memory and exact-resume results are recorded in the Phase II results report. Real source-data and independent-cohort questions remain NOT TESTED/BLOCKED regardless of software PASS.

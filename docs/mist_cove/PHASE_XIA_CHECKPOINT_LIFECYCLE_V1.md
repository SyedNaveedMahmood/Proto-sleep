# Phase XI-A original checkpoint lifecycle v1

Selected identity: original P3B raw_context123, source commit661dbb3dd7b2584d91b2c52f42547e1cd9c3dc23, run fingerprinta9146071f32c148e30ed84a6763d77cc82c3de83bf6f90719be0f774c114754b. This is source/Git inspection, not execution of the historical runner or inspection of its private outputs.

| Step | Frozen code path | Original contract and evidence |
|---|---|---|
| Initialization | full_experiment.run_full → seed_all → make_baseline | Seed set before construction; all three seeds share the declared matched recipe; no current initialization reproduces actual bytes by assumption |
| Construction | baselines.make_baseline / cli.variant_config / EvidenceModel | Bank config→raw_control=True; epoch-local radius0, no CRF; 169 windows/209 features in default config |
| TRAIN bank | full_experiment.prepare_sources / source_data.training_bank,validate_bank | Seed123 reuses authenticated P3A bank in the original run; other seeds fit TRAIN-only medoids; cfg and normalization fingerprint stored with bank; raw_control bypasses values but bank buffers remain in state |
| Normalization | prepare_sources | Original P3A completion binds normalization bytes; copied exact TRAIN statistics and source proofs checked before fitting artifacts receipt; no values read here |
| Run identity | full_experiment.run_full / provenance.git_info | Version/config/split-file hash/source-code hashes/environment/CUDA precision/settings → run fingerprint; root provenance binds correctness receipt; implementation frozen before original training |
| Training | mist_evidence.runtime.train | AdamW3e-4/wd1e-4; max60/patience12; core/encode16, clip5; deterministic FP32, no AMP/TF32; original TRAIN scored cores only |
| Evaluation | mist_transfer.experiment.evaluate_source | Complete original physical VAL grid before scoring masks; mean of four participant fixed-five F1 values for selection; source development, not clinical test |
| Selection | runtime.train | Improve>1e-8, earliest best on ties; candidate123 best epoch28, actual40,199120 updates in published P3B aggregate |
| Best save | runtime.atomic_torch_save / train | torch.save(state_dict,model_config,bank,signature,best_epoch,metrics); temporary file fsync then os.replace; best may be overwritten at each improved epoch |
| Last save | runtime.train | state_dict+optimizer+signature+epoch/best/stale/history+Python/NumPy/CPU/CUDA RNG; model and optimizer coupled by one serialized object |
| Resume | full_experiment.verify_boundary / runtime.train | Latest epoch output digests checked; paused boundary/signature checks; original unrestricted load then state/signature checks; runtime can resume under same fingerprint, but local receipts alone are not adversarial authentication |
| Epoch seal | full_experiment.EpochObserver | Every epoch JSON hashes current last/best/history/provenance bytes and records resource/metrics/signature. Earlier hashes need not match later overwritten best/last files; latest boundary is the pre-load check |
| Complete | runtime.train | COMPLETE records signature/best_epoch/metrics/best_state_digest; post-selection state loaded/evaluated by the original trusted-local runner |
| Validated complete | full_experiment.seal_result | Verifies finite states, boundary/history and state digest; per-run VALIDATED_COMPLETE hashes selected best/last, completion, history/prediction/provenance and epoch JSON files |
| Root complete | full_experiment.run_full | Root COMPLETE outputs hash results/config-lineage/fitted references, every per-run validated marker and its outputs, including selected serialized checkpoint bytes |
| Aggregate publication | reports/p3b at15995b31821ee8db717f3d6edfa41166de65d4a3 | results.json publishes named model-state digests; validation_summary.json and FINAL_COPY_PASTE.txt publish root COMPLETE digest905ba78cf8326594a431ce7a71ce9624c35814d2e76c94f57abda8afbe7c7e7d. Same publication, not two independent replications |

## What existed then versus what is known now

| Evidence | Produced by original code? | Independently published reference | Current XI-A observation |
|---|---|---|---|
| Raw best/last checkpoint-byte SHA256 | Yes, epoch/per-run/root output maps | Root-marker SHA binds a possible transitive chain; direct selected checkpoint-byte value not yet recovered | Private markers/checkpoint unopened |
| Model-state digest | Yes, sorted tensor names+raw tensor bytes | P3B best1b308f3fc38af22609d2043b0988998b1a193c6d4ace73ac84382bb17e1a4c6d | Verified in historical published Git blob; not equivalent to byte hash; algorithm omits explicit shape/dtype metadata |
| Standalone serialized optimizer digest | No separate semantic optimizer digest shown | Last checkpoint-byte hash covers optimizer together with model/RNG | No optimizer opened |
| Creation-time receipt | Root/run provenance written before fit/train; no independently signed prospective checkpoint-creation receipt shown | Original root may bind those provenance bytes after completion | Do not manufacture creation evidence today |
| Epoch-seal hashes | Epoch JSON output maps; root hashes epoch records | Transitive via published root, contents uninspected | Code contract only |
| Completion-marker hash | Yes | Original root SHA published; selected child marker expected in root map | Level2 root anchor only |
| Implementation/environment manifest | Original run identity records full implementation/reused hashes and environment; source recoverable in Git | Summary publishes implementation/reused hashes; full private environment uninspected | Recover code/config independently, do not infer environment from Phase III |
| Source/split/normalization/bank identities | Original source proofs, exact frozen split, TRAIN-fit lineage and fitted-artifact hashes | Split fingerprint/file SHA and root publication; detailed fit hashes not materialized | 3e7a… is the split fingerprint, not an authenticated normalization digest |
| Independent publication digest | Git blobs bind reports under historical commit15995b3, reachable from user-pinned Phase X base | Content-bound historical record under explicit Git trust root | Unsigned commits; no third-party signer/time claim |

New checksums in XI-A cover approved Git objects/reports and generated fixtures only. They do not retroactively become original artifact hashes. Historical code/protocol recoverability, historically anchored reports and actual artifact authentication remain separate claims.

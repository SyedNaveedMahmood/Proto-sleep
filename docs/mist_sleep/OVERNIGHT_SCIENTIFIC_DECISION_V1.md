# Phase M0 scientific decision: `MPI_FEASIBILITY_NOT_ESTABLISHED`

The original proposal's method is now available in the tracked 13-page text mirror. The default implementation is technically trainable and numerically stable on approved SC TRAIN, but its bounded-nuisance stage-performance gain over the matched ordinary-augmentation control did not reach the frozen +0.02 readiness threshold. Scientific review should keep the original method hypothesis open while declining a transfer-benefit claim from this experiment.

| Original overnight question | Evidence-based answer |
|---|---|
| 1. What is MPI? | Within-instance L2 distance between normalized morphology representations plus directional KL between corresponding 64-prototype assignments for a bounded transformation. Stage loss remains supervised; SupCon is separate and label-based. |
| 2. What was reusable? | Authenticated TRAIN source loader, frozen 10/5 allocation and fit-only normalization; AttnSleep architecture as adapted reference; prototype mathematical conventions but not its 48-by-30 coupled implementation. |
| 3. Why was historical AttnSleep weak? | Specific cause unresolved. The old mean source-VAL F1 was 0.31265; new adapted A0 scored 0.6889/0.6534 on different reused TRAIN development people and exposure. A0 overfit a synthetic batch, arguing against a basic gradient failure. |
| 4. Credible reference? | `ADAPTED_REFERENCE_ONLY`; no claim of faithful published AttnSleep reproduction. |
| 5. Does morphology architecture train? | Yes: M0 reached 0.6902/0.7268 clean participant Macro-F1 in two fixed seeds at 1,200 updates. |
| 6. Correct prototype initialization? | Yes: fit-only 4,096-token k-means; input digests identical across M0–M3 within each seed. |
| 7. Transformation safety? | Numerically bounded on 100 fit samples: RMS ratios 0.950–1.089, spindle-band ratios 0.903–1.120, slow-band ratios 0.899–1.101, finite and aligned. Historical P3B prediction agreement was 98%; generated event-like components retained near-unit projections. Clinical label preservation unverified. |
| 8. Correct MPI gradients/alignment? | Generated identity/shape tests pass; both terms yield nonzero encoder and bank gradients; no time shift in primary stream. |
| 9. Representation collapse? | No constant-embedding collapse detected; M2 feature variance 0.00503/0.00478. |
| 10. Prototype collapse? | No dead M2 units; effective counts 59.2/57.5 of 64, with 49/36 near-duplicate pairs. |
| 11. Embedding consistency improvement? | Yes under the one bounded view: M2 L2 drift 0.0277/0.0212 versus M0 0.0371/0.0257. |
| 12. Assignment consistency improvement? | Yes: KL 0.00297/0.00201 versus 0.00407/0.00228. |
| 13. Distinguishable terms? | Separate nonzero gradients and observed drift/switch measures; isolated term-trained model ablations were not run. |
| 14. Stage-only architecture adequate? | It learned all five classes and scored 0.6902/0.7268 clean participant Macro-F1; adequacy for external transfer remains unknown. |
| 15. SupCon benefit? | M1 changed stage tradeoffs, raising N1 but lowering REM; clean F1 0.6892/0.6842 versus M0 0.6902/0.7268. No consistent aggregate gain. |
| 16. MPI beyond ordinary augmentation? | M2–M0 clean differences −0.00067/+0.00806; bounded +0.00046/+0.01123. Stability improved, F1 readiness gate failed. |
| 17. MPI beyond SupCon? | M2 exceeded M1 by +0.02551 two-seed mean clean F1, but this is secondary and cannot replace the M0 contrast. |
| 18. MPI plus SupCon complementary? | Not consistently: M3–M2 clean two-seed mean −0.02308, with a REM penalty in seed 456. |
| 19. N1 and REM preserved? | M2 retained nonzero recognition; clean N1 mean recall 0.6444 versus M0 0.6623, REM 0.6426 versus 0.6239. Mixed tradeoff. |
| 20. Bounded nuisance robustness? | M2 two-seed participant F1 +0.00585 versus M0, below +0.02; prediction agreement gains were ~0.0015–0.0020. |
| 21. Stage information preserved? | Fit-only linear probe M2 0.7884/0.7895 versus M0 0.7923/0.7885; no consistent gain or gross collapse. |
| 22. Incremental cost? | M2 and M0 each have 41,429 parameters and identical inference graph. Mean training time 8.54 versus 8.13 s per trajectory (~5% more here). |
| 23. Seed consistency? | Drift and KL improvement in both; bounded F1 direction positive in both but effects differed materially and both fell below +0.02. |
| 24. Budget-blocked conclusions? | K=32/128, EMA, single-term MPI fits, five-seed final analysis, capacity-matched AttnSleep, DA, foundation comparator, target adaptation and all external transfer were not tested. |
| 25. External study justified? | No automatic GO: the frozen overnight readiness threshold failed. A separate scientific review may authorize an independent falsification study, not a claim-driven confirmation. |
| 26. Alternative explanations? | Minor gain could arise from regularization interacting with the fixed augmentation or two reused development seeds; transformed stage labels have no clinical validation. |
| 27. What stays unchanged? | The original testable MPI/SupCon separation, shared source-fit prototypes, quantitative morphology naming rule, external-target claim hierarchy and negative-transfer analysis. |
| 28. What needs revision? | Demonstrate token-aligned safe shifts if used, improve deterministic run setup before new training, add term-specific and capacity controls only under a new sealed protocol, and make montage compatibility explicit. |
| 29. Best next source-target study? | A separately authorized SC-to-independent-clinical target such as HMC v1.1 can test severe combined shift, but its C4/M1/C3/M2 256 Hz montage differs from SC Fpz-Cz and must be audited before interpretation. |
| 30. Single decisive experiment? | Freeze unchanged M2/M0 source checkpoints and score one independent, channel-audited external participant test set under a preregistered inductive protocol. No such test ran tonight. |

The original PDF binary was unavailable locally; the complete proposal text and registry were integrated. The completed runs were uninterrupted. Generated CUDA exact-resume passed only with deterministic cuDNN/cuBLAS settings; the actual campaign did not enforce those settings, so its interrupted bitwise replay remains unproven. No participant-level data or model weights are published. No new external EEG was accessed.

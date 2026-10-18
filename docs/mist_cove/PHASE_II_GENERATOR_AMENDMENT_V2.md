# Phase II preregistered generator correction v2

The original v1 experiment completed at implementation `addce7b6327f1a42e3cbbcc99af024a2751300bd`. Its receipt and original preregistration are retained. This amendment is committed before evaluating the corrected generator.

Inspection after v1 revealed a genuine identification error in the complementary-task one-factor fixtures: they always retained the A carrier. Although a local band appears in two different full pairs, the known asymmetric removal policy can identify which factor survived and thereby remove the intended ambiguity. Observed partial-case accuracy alone could not establish irrecoverable information loss. The full dual-pair task and all algebraic decompositions remain well-defined; the one-factor ambiguity claim was not adequately enforced by that generator.

v2 uniformly chooses whether A or B is the surviving factor in complementary single, redundant, correlated, distractor and tied-copy fixtures. Selection of the surviving factor uses only the local seeded generator, independently of class. It does not add either factor's identity to the model input. Thus an observed single band has two possible latent classes under balanced priors. v1 remains selectable and reproducible. Substitutable fixtures and full dual/competition/overlap constructions are unchanged.

No method, model capacity, training recipe, seed list, selection budget, donor strategy, acceptance threshold or primary reference is changed. W1–W4 are rerun and reported separately under `mist-cove-phase2-synthetic-v2`; this is a protocol correction, not a positive-result rescue. Training is unchanged because complementary training already uses full dual pairs. v1 results, including donor failures and strict calibration-tolerance failures, are preserved rather than retroactively relabeled as successes. No real EEG or reserved/external artifact is involved.

The revised config and comparison matrix are hash-bound in `experiments/mist_cove/phase2_preregistration_v2.json` before the revised experiment. All subsequent generator/threshold changes would require another separately labeled version.

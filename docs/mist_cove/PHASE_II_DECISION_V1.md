# Phase II scientific decision and preregistration

Authorization: the user's Phase II execution request on 2026-10-18 permits code, synthetic research, mock integration and available-device synthetic CUDA checks only. No real EEG, reserved descriptor, source normalization, ST or SHHS access is authorized. Base: verified Phase I `1a95a54f95edc6f694836a97091336e1e745e4f4`; clean checkout, fetched origin, new branch `research/mist-cove-phase-ii-witness-v1`. Phase I source, configs, reports and findings are immutable in this phase.

H1's predefined superiority criterion **FAILED** in all Phase I seeds. H2 had directional support but weak contribution/effect magnitude agreement and redundant-witness limitations. H3 did not establish gating necessity. The availability score remains a stage-shared spectral/amplitude heuristic, not calibrated pairwise observability. The primary reference is now **q=1**, with the unchanged availability decision function retained as a control. No MoE, prototypes, new large encoder, extra training loss or tuned regularization is proposed.

## Distinct evidence claims

Let S0 be the originally selected witness identities and S1 the identities selected on the replaced waveform. Write B(X) for background and E(X,S) for the sum of event contributions at S. For a replacement X→X~, report every signed five-stage vector separately:

- Additive contribution C(G): original contributions from selected intervals intersecting the declared group G. This is exact bookkeeping, not a waveform-removal effect. Overlapping group intervals are unioned before replacement and a selected witness is counted once.
- Full waveform effect: B(X)+E(X,S0)−B(X~)−E(X~,S1).
- Fixed-witness effect: B(X)+E(X,S0)−B(X~)−E(X~,S0). Original identities and validity are retained, even if the replacement becomes flat. This is an internal model intervention.
- Event-path effect at fixed original background: E(X,S0)−E(X~,S0). This is also an internal intervention, not asserted to be physiologically realizable.
- Background-only change: B(X)−B(X~).
- Reselection: E(X~,S0)−E(X~,S1).
- Internal score deletion: zero the original selected contributions for G with every other score fixed; its effect equals C(G) by construction. It is explicitly not waveform deletion.

The full waveform effect must reconcile with background + fixed event-path + reselection to absolute tolerance 1e-5. Individual and group effects are computed by their own full interventions; joint effects are not the sum of individually recomputed effects. Redundancy and substitutability can make an individual useful but unnecessary. No per-interval physiological causal effect is identifiable from these bookkeeping identities alone.

## Frozen experiment before execution

`configs/mist_cove_phase2_synthetic_v1.json` and `experiments/mist_cove/phase2_comparison_matrix_v1.json` contain all seeds, exposure budgets, generator constants, methods and acceptance thresholds. Their byte SHA256 values are saved in `experiments/mist_cove/phase2_preregistration_v1.json` and committed before new synthetic experiments. Criteria will not be tuned after seeing outcomes.

Two independent artificial coding tasks avoid relabeling substitutable carriers as complementary information. The substitutable task uses Phase I's five band codes. The complementary task uses adjacent-band pairs on a five-band ring: each individual band is shared by two classes, while the pair identifies the class. Models are trained separately per task with equal exposure across methods. No fixture is clinically scored sleep.

Cases include a single witness, two alternatives/complementary factors, two identical redundant witnesses, six correlated copies, a diagnostic signal with randomized off-band artifact, competition for K=4, and overlapping known components. Groups are declared by generator identity before attribution, never chosen to maximize removal effects. Complementary-task single-factor cases deliberately lose class information; high accuracy is not required there.

W1 tests the algebra and whether reselection contributes materially on correlated/competition cases. W2 compares joint predeclared diagnostic-group removal against mean individual effects on redundant/correlated cases. W3 compares hard spectral-RMS ranking with one simple fixed alternative: round ranking scores into 0.5-uV bins and resolve ties by original sample time. It adds no parameters, retains K=4, and is rejected if its predefined stability/accuracy criterion fails. Primary q=1 remains the default regardless of a cosmetic explanation improvement. Budget probes hold per-event score scale at 2 while varying K=2/4/8 to avoid conflating budget with Phase I's 8/K scaling.

W4 measures boundary discontinuity, energy mismatch, spectral distortion and background changes for bridge, same-epoch background and independently generated background-noise erasers. Phase randomization is a separate information-retaining spectral negative control, not a valid substitute for erasing the diagnostic feature. Donor guards cannot certify physiological validity. Effects and validity diagnostics remain visible even when donor sensitivity contradicts the desired narrative.

Report five-class accuracy/F1/confusion/recall, NLL/Brier/ECE, prediction collapse, all counterfactual components, additive-versus-full disagreement, individual/group and reordered-witness stability, physical-shift alignment, ranking jitter, calibrated amplitude conversion and arbitrary physical amplitude changes separately. Scores remain signed for all stages. Three seeds are generator/optimization repetitions, not participant inference. Report PASS/FAIL by the frozen criteria, NOT TESTED for clinical/transfer questions, and BLOCKED for unapproved data operations.

## Source software boundary

Prepare a separate adapter and guarded runner using existing development loaders, full-grid evaluator, trainer and atomic checkpoint functions without changing them. Source configuration preserves the frozen P3B optimizer/stopping recipe and split digests. Mock tests must cover wrong roles before I/O, immutable provenance, full-grid/masked validation, physical gaps, atomic saves, tamper rejection, incompatible resumes and exact optimizer/model/Python/NumPy/CPU/CUDA RNG recovery. Historical raw_context and evidence definitions remain explicit future comparisons. Any historical artifact use must verify its trusted digest and TRAIN provenance; no unsafe deserialization fallback.

An available RTX 4080 SUPER may be used only for generated-data smoke/resource/resume checks. Source training remains blocked until a separate authorization, frozen source implementation, trusted normalization/bank proofs, correctness receipt, GPU feasibility and source preflight are approved. No Phase II test receipt is itself source-data authorization.

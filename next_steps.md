# Next steps: MIST-WIC -- Witness-Interference Circuits

**Project:** MIST-Transfer / MIST-COVE, Proto-sleep  
**Plan version:** Phase IX proposal v1.0, 2026-10-20  
**Reviewed base:** `b50e41efb3b09a9cb7fa4142366424725fc603e5`  
**Base branch:** `research/mist-cove-phase8-robust-evidence-v1`  
**Proposed working branch:** `research/mist-cove-phase9-witness-circuits-v1`  
**Repository:** https://github.com/SyedNaveedMahmood/Proto-sleep  
**Status:** Proposed method and execution plan. No Phase IX implementation, experiment, performance result, novelty certification, or real-data authorization is implied by this file.

> **Decision:** Stop treating a waveform explanation as one importance score per interval. Investigate an operator-specific, signed response circuit that distinguishes alternative witnesses, jointly needed witnesses, interfering context, and the limits of the explanation. Test clean-domain explanation quality independently of severe-corruption robustness. Retain the existing classifier initially; do not start by making it larger.

## 0. How to use this plan

The coding agent should read this file completely, then execute the numbered work packages in Section 13, in order. The task IDs, not an expanding list of new phase names, define completion. Keep a single progress ledger. A blocked experimental branch does not block independent correctness, explanation, or reporting work.

This plan distinguishes three kinds of statement:

- **Repository evidence:** Findings already reported at the reviewed commit, referenced as S1-S4.
- **Proposed method or deduction:** New design choices and mathematical consequences developed here. They require implementation and testing.
- **External research:** Related published work or preprints, referenced as R1-R9. The related-work scan is targeted, not exhaustive.

The numeric thresholds and budgets below are **proposed prespecifications**, not observed results. Freeze them, implementation identity, and evaluation streams before confirmatory execution. Any necessary change before that freeze must be documented. After evaluation, a scientific amendment creates a new study identity; it never replaces a failed historical result.

**Scope of the accompanying execution prompt:** code, repository/publication review, trusted synthetic artifacts, generated experiments, CPU tests, bounded optional CUDA checks, and a draft PR. Real Sleep-Cassette TRAIN/VAL EEG, data-bearing descriptors/annotations, real fitted artifacts, the 55 RESERVED subjects, Sleep-Telemetry, SHHS, legacy NPZ, and target adaptation remain inaccessible. A future source study is a proposal only.

---

## 1. The scientific pivot

### 1.1 What the completed work actually supports

| Established repository observation | Meaning for this plan |
|---|---|
| Phase III q1 source-VAL Macro-F1 was about 0.478, versus 0.724 for raw_context and 0.718 for evidence. | The minimal analytic implementation was not competitive; this is not a disproof of all morphology-aware methods. Preserve these source-development results. [S2] |
| Phase VIII B1 improved mean competitive scores but failed four of six complete task/seed gates. | Better average corruption performance is not clean-performance preservation or universal robustness. [S1] |
| Phase VIII B1 single clean/competitive means were 0.893429/0.823372; multiple means were 0.844029/0.732019. | Keep both axes visible. Do not select the best seed or a secondary model after evaluation. [S1] |
| Consecutive robustness prerequisites prevented confirmatory additive-head comparisons. | Explanation quality on an adequately learned clean task can be tested without claiming severe-corruption robustness. This is a prospective change to the dependency structure, not a revision of old FAILs. [S1, S3] |
| Individual contributions differed from whole-model effects; overlapping windows and reselection matter. | Explain sets and interactions, and explicitly define what removal means. A sum of scores is not a proof of necessity. [S2, S3] |
| All 1,500 Phase VIII donor attempts were rejected. | Stop requiring a successful intervention to both remove diagnostic spectral information and preserve that same information. Specify the intended intervention first. [S1] |
| The working local encoder has 3,408 parameters; conventional M1 has 3,593 total. | Reuse the tested architecture first. Do not accidentally describe 3,593 as the encoder-only count. [S4] |
| Phase VIII verified 18 prospective references; older unauthenticated Phase VI weights remained unopened. | Use the sealed prospective states, or an explicitly new reproduction. Current hashes cannot retrospectively authenticate old training. [S1] |

### 1.2 Four changes to the research strategy

**First: explain response structure, not only a selected patch.**

An illustrative model response may look like:

```text
(A OR B) AND NOT C
```

A and B are alternative retained regions. C is a region whose inclusion disrupts the target response under a specified model intervention. Another case may require `A AND B`. These are hypothetical logical examples, not measured EEG events or clinical labels.

A single heat map can obscure these distinctions. A response circuit can show them explicitly. This does not mean that an additive classifier can never represent such decisions; it means that one vector of attribution values is not a complete description of the intervention-response function.

**Second: separate three research tracks.**

- **Track A -- explanation:** Can we characterize the same frozen model's clean-domain witness and interference relationships faithfully and compactly?
- **Track B -- classification:** Does consistency of responses to retained evidence improve robustness beyond ordinary full-input consistency training?
- **Track C -- intervention validity:** Do internal withholding, exact generated-component removal, and waveform replacement agree or differ in understandable ways?

A Track B failure does not block Track A. Track C may report that two interventions answer different questions rather than forcing their effects to be equal.

**Third: preserve the predictor in the first experiment.**

The first MIST-WIC implementation is a verified analysis layer on a frozen predictor. At full visibility it must reproduce that predictor exactly. This immediately makes it possible to study explanations without imposing a new accuracy bottleneck. A later, separately controlled training experiment tests whether the same intervention structure can improve the predictor.

**Fourth: bound the next study.**

One core method, one matched-head comparison, one three-arm robustness comparison, and one intervention audit are sufficient. Do not create another 90-file package or an open-ended architecture search. Reuse existing tests, data generation, metrics, checkpoint machinery, and reporting utilities where their contracts fit.

---

## 2. What could be novel -- and what is already known

The working name is **MIST-WIC: Witness-Interference Circuits**. It is terminology proposed for this project, not the name of an established method.

### 2.1 Proposed contribution

The candidate contribution is the combination of:

1. **Overlap-closed evidence withholding:** a physical-sample availability mask that cannot leave hidden samples visible through neighboring receptive fields.
2. **Signed, nonmonotone response circuits:** an exact finite description of which retained and withheld regions support a model response, including distracting context.
3. **Two distinct stability claims:** one fixed explanation surviving all tested views versus different valid witnesses supporting the same unchanged decision in different views.
4. **Operator-specific validation:** finite internal certificates are checked separately against exact generated signal interventions; neither is called clinical causality.
5. **An optional response-consistency training objective:** compare consistency across retained-evidence queries with ordinary full-view consistency, under matched data and compute accounting.

The potential scientific result is **not** "we invented subsets, Boolean circuits, or consistency regularization." It would be evidence that these particular semantics expose and potentially reduce EEG-like distractor failures while giving truthful accounts of redundancy and interference.

### 2.2 Related-work overlap that must be acknowledged

| Prior work | Established overlap | What this project must demonstrate beyond relabeling it |
|---|---|---|
| Sufficient Input Subsets [R1] | Minimal observed feature subsets sufficient for a decision. | Interfering context, receptive-field overlap, and finite-view witness switching rather than only a short sufficient subset. |
| Abductive/contrastive explanation duality [R2] | Minimal reasons, contrastive changes, and hitting-set relationships. | Precisely scoped time-series intervention semantics; do not claim a new hitting-set theorem. |
| Explaining by Removing [R3] | Removal operator, explained behavior, and summary are distinct design choices. | An implementable operator contract and measured differences between withholding and generated-component deletion. |
| TimeX [R4] | Time-series explanations and consistency with a predictor's behavior. | Same-model exact finite response analysis, without presenting a separately learned surrogate as exact. |
| TimeX++ [R5] | Explanation objectives can suffer trivial solutions and distribution shift. | Explicit handling of missing-evidence inputs and no unqualified in-distribution claim. |
| Concept-based abductive/contrastive explanations, 2026 preprint [R6] | Concept erasure, sets of explanations, and enumeration; some reductions use a head-monotonicity assumption. | A close comparator. Test nonmonotone window competition explicitly and do not assume adding evidence can only help. This is not proof that all prior algorithms require monotonicity. |
| Binary decision diagrams [R7] | Exact compact representation of Boolean functions; worst-case exponential size. | BDDs are implementation infrastructure, not this project's novelty. |
| Attention-based multiple-instance learning [R8] | Learning from sets of instances with a bag label. | No claim that patch aggregation or instance scoring alone is new. |
| TRADES [R9] | A principled accuracy/robustness trade-off and consistency-related training. | Artifact experiments are not automatically adversarial-robustness proofs; test the additional retained-evidence objective against a full-view consistency control. |

**Novelty gate:** Before coding scientific comparisons, the agent must extend this targeted review with primary-source searches for nonmonotone explanations, overlapping time-series masks, multiple rationales, stable sufficient sets, and EEG explanations. Record the nearest two methods and a concrete differentiator. If an essentially identical method exists, position this work as a rigorous extension or application. Do not fabricate novelty, conceal close predecessors, or launch unrelated literature searches.

---

## 3. Core architecture: keep the model, change the explanation object

### 3.1 The base computation

Use the existing local encoder first:

```text
3,000 physical samples
    -> fourteen 400-sample windows at stride 200
    -> fourteen 18-dimensional local feature vectors
    -> unchanged conventional or additive decision head
    -> five logits
```

Read the actual implementation in `src/mist_cove/phase4/model.py` before adapting it. The current conventional decision uses feature mean and maximum; the additive model uses local scores and its own background computation. Do not assume their internals are interchangeable. [S4]

Full-visibility inference must remain exactly the native method's computation. No mask choice, explanatory score, oracle interval, clean twin, or generated component label may alter the deployed full-input prediction in Track A.

### 3.2 Physical atoms, not independently erased overlapping windows

Partition the epoch into 15 disjoint atoms:

\[
A_j=[200j,200(j+1)),\qquad j=0,\ldots,14.
\]

The existing candidate window is:

\[
W_i=A_i\cup A_{i+1},\qquad i=0,\ldots,13.
\]

Let `m[j]` mean that atom j is available. A candidate is visible only when **all** its samples are available:

\[
v_i(m)=m_i m_{i+1}.
\]

Set hidden feature vectors and every related auxiliary value, including candidate RMS and validity, to the prespecified neutral state. For the initial internal operator use zero features and zero RMS, preserving the native head's pooling definition. Do not silently renormalize mean pooling by the number of retained candidates: that would be a different operator.

This conservative policy discards partially visible candidates. It can hide useful neighboring information; report this collateral support loss. It is preferable to pretending that withholding one window removes its samples from overlapping neighbors.

A mask is an internal evidence-availability intervention, **not a plausible reconstructed EEG waveform**. Zero embeddings are not necessarily in distribution. Baseline dependence must be measured; do not sell masking as physiological erasure.

### 3.3 Two correctness properties

**Full-input identity:**

\[
f_\theta^{\rm int}(x,\mathbf 1)=f_\theta(x).
\]

Test identical logits, prediction, and feature path; use an explicit floating-point tolerance only where batching changes arithmetic order. A discrepancy beyond that tolerance blocks this operator.

**No hidden-sample leakage:** if two waveforms agree on all available atoms, and each retained encoder output depends only on its declared local window, then

\[
f_\theta^{\rm int}(x,m)=f_\theta^{\rm int}(x',m).
\]

This follows because every active feature sees identical samples and all inactive feature/auxiliary values are neutral. Validate it with arbitrary changes in withheld samples, including large artifacts. A global unmasked summary, input-dependent full-epoch normalization, or unmasked RMS invalidates the premise. A fixed TRAIN normalizer does not by itself violate locality.

These are new plan deductions and software contracts, not novel general mathematical theorems or clinical guarantees.

### 3.4 Why 15 atoms are useful

There are exactly `2**15 = 32,768` internal masks. For a fixed input, encode the fourteen candidates once and evaluate masks in batches through the small decision head. Do not re-encode 32,768 waveforms.

Exhaustive finite evaluation permits an exact response table for a **bounded audit subset**. It is not a scalable solution for arbitrary signal length. A smaller or partial search must be labeled incomplete; do not claim a complete certificate from a top-K search.

---

## 4. From mask responses to witness-interference circuits

### 4.1 Fix the explained response before intervening

For the main explanation of an example, fix:

\[
c=\operatorname*{argmax}_k f_\theta(x)_k.
\]

Do not switch to the ground-truth class when the model is wrong. Correctness-stratified analyses may additionally inspect the true class, but must label this as an error-analysis target. Compute all five classes if convenient; the predicted-class explanation remains identifiable in the output.

For a fixed mask and one input view, define the target margin:

\[
r_c(x,m)=\ell_c(x,m)-\max_{k\ne c}\ell_k(x,m).
\]

This margin is invariant to adding the same scalar to all logits. Raw single-class logit changes need not have that property. Keep raw vectors for diagnostics, but use margins or calibrated probability differences for cross-intervention claims.

The default finite response is `r_c > 1e-6`. Treat ties/near-ties as unresolved, not success. A separately reported high-confidence response also requires calibrated `p_c >= 0.60`; its calibration is fitted on a disjoint synthetic calibration split. The probability threshold is a reporting convention, not a statistical guarantee.

### 4.2 Signed certificate clauses

A clause contains:

- `P`: atoms required to be retained;
- `N`: atoms required to be withheld;
- all remaining atoms free to be retained or withheld.

Require `P` and `N` to be disjoint. Define:

\[
\mathcal M(P,N)=\{m:m_j=1\ (j\in P),\ m_j=0\ (j\in N)\}.
\]

For a **specified finite** set of views `V`, a certificate is valid when:

\[
\min_{x'\in V}\;\min_{m\in\mathcal M(P,N)}r_c(x',m)>10^{-6}.
\]

This means the retained/withheld pattern is sufficient for that response under every enumerated mask completion and every listed view. It says nothing about unenumerated perturbations, other patients, other montages, physiological causes, or the correctness of c.

A negative literal does **not** mean that a clinical symptom is absent. It means withholding a region is a condition of the model response under this operator.

At the original all-retained mask, a valid clause describing the actual response cannot require any atom to be absent. Clauses requiring absences describe alternative model-intervention states or potential repairs, not a reason that was active in the original full-input prediction. The report must make this distinction visually and in its schema.

### 4.3 Nonmonotonicity is a required test, not an inconvenience

Include exact toy response functions for:

```text
Alternative witnesses:           A OR B
Complementary witnesses:         A AND B
Interfering context:              (A OR B) AND NOT C
Mixed pathways:                  (A AND B) OR (D AND NOT E)
Nonmonotone sanity control:        A XOR B
Prior-only prediction:            constant true
Unsupported response:             constant false
```

The compiler must recover their truth tables without assuming monotonicity. A prior-only response has no waveform-support claim. A constant-false response has no certificate. Do not conflate either with a small successful explanation.

A greedy sufficient set such as `{A}` tested only with all other inputs withheld does not prove that A remains sufficient when C is restored. This is the distinction the signed clause verifies.

### 4.4 Exact representation, readable partial summary

Compile the complete Boolean response table into a reduced ordered binary decision diagram (ROBDD), using physical-time atom order. BDD construction is established prior art [R7]. Verify equality against every truth-table row.

Then extract a bounded readable collection of signed clauses. Minimize each clause by removing literals only when exhaustive completion checking still passes. After convergence, test every single-literal removal. Report **subset-minimal**, not globally minimum cardinality or a complete enumeration of all prime implicants.

Why the local minimality check is sufficient here: removing a literal enlarges the set of completions. If removing any one literal invalidates a clause, removing more cannot restore universal validity. This reasoning concerns universally checked clauses; do not apply it to nonmonotone single-mask sufficient-set search.

Use at most four displayed clauses initially; archive the exact circuit and coverage statistics. A compact display may omit valid routes. Report its coverage and never substitute the display for the full verified response table.

### 4.5 Negative interference and genuine diagnostic evidence are different

If removing C recovers the true label from a wrong prediction, the model has a possible internal repair. It has not demonstrated that C is an actual recording artifact or that an automatic deployed procedure could find and remove it safely.

An interference flag requires measured nonmonotone response behavior. The label **nuisance** additionally requires the synthetic generator's independent metadata or later external clinical evidence. Keep these separate.

---

## 5. The creative robustness question: stable witness versus witness substitution

The original proposal distinguished preserving a decision from preserving one waveform. Make that distinction executable.

For finite nuisance views `v`:

**Fixed-witness stability:** one explanation is valid across every view.

\[
\exists C\;\forall v:\;\operatorname{valid}(C,v).
\]

**Witness substitution:** each view has a valid explanation, potentially a different one.

\[
\forall v\;\exists C_v:\;\operatorname{valid}(C_v,v).
\]

The first implies the second; the converse does not generally hold. Measure both. Do not require identical selected intervals when one redundant event is obscured but another remains available.

For either to count as explanation stability of a robust full-input prediction, the full-input prediction must also remain c in every tested view. Finding a favorable masked subset after the full classifier becomes wrong is an internal repair analysis, not robust classification.

Use discovery and verification views separately:

- Discover clauses on a frozen small set of generated views.
- Verify unchanged clauses on independent views without refitting their literals.
- Report fixed-clause survival, alternative-route coverage, full-input decision stability, and the fraction of cases with no short certificate.

A certificate exact on discovery views is still only empirical on a different sampled verification distribution. Do not call either universally certified adversarial robustness.

---

## 6. Optional model improvement: retained-evidence response consistency

This is Track B, not a prerequisite for Track A.

### 6.1 Hypothesis

Ordinary consistency aligns predictions of full clean/corrupted inputs. It may leave the model's reliance on retained local evidence unstable. A consistency term over **identical evidence-availability queries** could reduce nuisance-dependent decision pathways without modifying the encoder size.

This is a testable extension of established consistency ideas [R4, R9], not a claim that consistency training is new.

### 6.2 Three matched training arms

Use identical encoder/head architecture, initialization policy, paired data, minibatch indices, update count and optimizer for:

| Arm | Objective | Role |
|---|---|---|
| C0 | Paired clean/corrupted cross-entropy only | Exposure control |
| C1 | C0 plus full-input Jensen-Shannon consistency | Standard consistency control |
| C2 | C1 plus retained-evidence response consistency | Prespecified primary proposal |

For each generated pair `(x, x_plus)`:

\[
L_{\rm cls}=\tfrac12\{CE(y,p(x,\mathbf1))+CE(y,p(x_+,\mathbf1))\},
\]

\[
L_{\rm full}=JS(p(x,\mathbf1),p(x_+,\mathbf1)),
\]

\[
L_{\rm query}=\frac{1}{|Q|}\sum_{m\in Q}JS(p(x,m),p(x_+,m)).
\]

Default proposed weights: `C1 = L_cls + 0.5 L_full`; `C2 = L_cls + 0.5 L_full + 0.25 L_query`. Use a deterministic linear warm-up from zero to these regularizer weights over the first 20% of updates. No loss-weight sweep is authorized.

Sample two atom masks per pair, independently of class and event positions, from fixed retention probabilities 0.50 and 0.75. Reject only masks that retain no complete candidate or all candidates; log that geometry-only sampling policy. Reuse encoded features, but preserve gradients through the head and encoder for the chosen loss.

**Do not assign the original class label to every masked subset.** Some subsets have no class information. The extra subset objective aligns paired model responses; it does not assert that a missing-evidence query must predict y. Full-view CE remains the only class target in this core comparison.

Train using only additive, prespecified nuisance pairs where the generated diagnostic component is unchanged. This is a training assumption, not proof that arbitrary high-amplitude corruption preserves observability. Complete erasure, clipping away a diagnostic component, ambiguous partial factors, and deliberate class swaps are not label-preserving consistency pairs. They are separate evaluations.

No clean twin, oracle event interval, or generator label enters inference. Label and known transform identity may enter training. Report any use of privileged generator information explicitly.

### 6.3 Important failure modes

C2 may flatten responses on all small subsets, memorize corruption families, preserve a wrong clean prediction, or harm clean accuracy. Full-view CE does not mathematically prevent these outcomes. Measure prior-only circuits, class collapse, complete-erasure confidence, and clean/robust trade-offs.

Do not train toward Boolean truth tables extracted from the model and then treat agreement with those same tables as evidence of independent explanatory validity.

---

## 7. Repair the intervention question, not the old donor threshold

Keep every historical donor rejection and every original guard unchanged.

### 7.1 Specify three operators separately

| Operator | What changes | Claim it can support |
|---|---|---|
| Internal availability | Hidden local features and their auxiliaries are withheld using the atom-closure rule. | Exact response structure of this model/operator pair. |
| Synthetic component intervention | Recompose the generated signal without a specified component, preserving the other stored components. | Effect of that known generated component on this model. |
| Waveform donor replacement | Replace signal samples using a declared donor population and blending rule. | A donor-conditional model effect; physiological validity remains unproven. |

Do not equate their effect magnitudes. A mismatched pair of operators cannot isolate a unique mechanism.

### 7.2 Use an intervention contract with separate channels

For `x = b + d + n`, removing diagnostic d should preserve b and n, not preserve the spectral evidence contributed by d.

Evaluate:

1. **Target change:** did the intended component/information actually change?
2. **Nontarget preservation:** were the declared retained components preserved?
3. **Boundary/physical plausibility:** did the operator introduce an unintended discontinuity or invalid waveform?
4. **Predictive effect:** how did the model respond?

A criterion that requires both removal of diagnostic spectral information and near-identity of that same diagnostic spectrum can be internally conflicting. That is a hypothesis to demonstrate with the exact generated oracle and identity controls, not a license to weaken old thresholds after results. [S1, R3, R5]

For example, the identity intervention must fail an erasure-success check even though it perfectly preserves energy. Exact generated diagnostic erasure must pass target-removal/nontarget-preservation checks, even when total spectrum changes. It may still fail a separate smoothness or model-distribution check; report all channels independently.

### 7.3 Resolve earlier unmatched-control and ambiguity problems

Use component recomposition for both diagnostic and non-diagnostic controls. Do not compare diagnostic component subtraction with unrelated whole-window zeroing and call the difference uniquely diagnostic.

For complementary factors, either retain the existing partial condition strictly as a descriptive, rule-identifiable stress case, or introduce a separately frozen symmetric hidden-member deletion endpoint with an explicit compatible-label set. Never train or evaluate an unidentifiable partial signal as though its original single label is uniquely recoverable. Do not rewrite the old generator.

### 7.4 No donor-training campaign by default

Start with a small no-fit contract audit: identity, exact generated erasure, exact nuisance removal, historical bridge, and historical independent-baseline replacement. Report the reason each check passes or fails. Another 1,500 rejected attempts would add little without a new estimand.

A learned real-EEG inpainting or donor generator is **out of scope** for this implementation.

---

## 8. Experimental program: independent tracks with bounded scope

### Track A1 -- clean-domain response circuits, no new model fits

Use the six trusted prospective P0 states: two tasks x three seeds. Verify their original creation evidence. If unavailable, a fresh six-fit reproduction is allowed only under a new identity with an explicit 1,800-update budget and an up-front accounting amendment; never relabel it historical recovery.

Run the exact finite response analysis on a deterministic audit sample of **two examples per class per task/seed**: 60 generated examples total. Choose them by generation order, not prediction success. Report correct and incorrect strata as well as full-sample coverage.

For each example use three fixed discovery views: base input, additive high-frequency nuisance, and an abrupt-transient view. Severity/profile must be frozen in the run config. Use independent nuisance verification draws later; never refit clauses to those verification outcomes.

Compare on the **same frozen predictor**:

- Native/internal ranking currently available in the repository.
- A greedy sufficient-subset baseline inspired by SIS [R1], honestly labeled a local reimplementation unless the official method is reproduced exactly.
- The same atom-mask search constrained to positive-only clauses.
- Full signed MIST-WIC clauses and exact circuit.
- Matched random/RMS interval summaries where applicable.

The exact circuit's truth-table equality is a correctness result, not evidence of clinical interpretability. The scientific question is whether a concise, validated summary reveals real model reliance on diagnostic and interfering regions and remains useful on independent views.

### Track A2 -- finally run the clean matched-head comparison

This new study explicitly authorizes a **clean-domain** comparison without requiring every severe-corruption criterion to pass. That changes only the new study's dependency graph.

Use the same sealed P0 encoder within each task/seed, freeze its 3,408 parameters, reset all heads with a paired initialization policy, and fit:

- E0: conventional head;
- E1: native sparse additive head;
- E2: event-disabled control;
- E3: RMS-ranked additive control.

Use 150 batch-16 updates per head at the historical synthetic AdamW recipe. This gives **24 fits / 3,600 head updates**. Keep all four methods even if one collapses. Evaluate clean and corruption conditions, but make the primary explanatory claim clean-domain only.

The parent prerequisite is numerical integrity, source-free provenance, and clean F1 >=0.80 on the independent development split, separately for each task/seed. A parent that fails is reported as an ineligible parent; do not silently replace its seed. Clean explanation diagnostics on wrong predictions may still run, with no claim of accurate diagnostic reasoning. Report every ineligible cell.

Attach MIST-WIC to E0 and E1. Analyze each predictor against its own internal truth function; do not compare E0's ground truth with E1's explanation as though they explain the same learned function. Across-head comparisons measure the complete model/head choices, including function-class differences.

### Track B -- C0/C1/C2 robustness comparison

Two tasks x three seeds x three arms gives **18 fits / 5,400 updates** at 300 batch-16 updates. The total new confirmatory program is therefore **42 fits / 9,000 updates**, excluding an explicitly approved missing-reference reproduction.

Each Track B update sees paired clean/corrupted views; log `base_examples`, `view_forwards`, `head_query_forwards`, and `optimizer_updates` separately. Do not merge cached-head exposure and full-waveform exposure into one misleading compute number.

C2 is the primary arm before any locked scores. Do not replace it with C0, C1, an old seed, or a post-hoc ensemble.

### Track C -- operator audit, no new fits

Run the independent intervention contract on the same fixed audit sample. Measure oracle-removal effects, token-withholding effects, and donor effects separately. Continue this track even if A2 or B fails.

### External baseline policy

The conceptual prior-art comparison in Section 2 is mandatory. A benchmark-adapted TimeX/TimeX++ execution is desirable for a later publication, but not mandatory in this bounded implementation because it would add new training and compatibility work. Mark external implementations NOT RUN. Do not claim superiority to methods that were only read.

---

## 9. Data splits, validation, and claims

### 9.1 Independent generated streams

Use optimization seeds 123, 456, 789, but new independent sample streams. Suggested split namespaces are strings passed to `numpy.random.SeedSequence` through a recorded deterministic mapping:

```text
phase9/train
phase9/development
phase9/calibration
phase9/circuit_discovery
phase9/circuit_verification
phase9/locked_evaluation
```

Do not use Python's process-randomized `hash()` to generate persistent seeds. Record the mapping and test disjoint identities. Preserve paired underlying draws across methods and nuisance conditions; resample the underlying draw, not individual transformed views, for uncertainty intervals.

Default sizes per task/seed: 1,000 TRAIN, 500 development, 500 calibration, 1,000 locked examples per main condition. The exact-circuit audit is the bounded 60-example sample above, not all locked epochs.

Only development data may support an explicitly recorded engineering choice before freeze. Locked evaluation is read once per frozen study. No repeated generator redesign to achieve a desired winner.

### 9.2 Primary and secondary endpoints

| Claim | Proposed criterion | Failure interpretation |
|---|---|---|
| Operator/compiler correctness | Full-input identity within declared tolerance; no hidden-sample leakage; exact Boolean circuit/table agreement on all 32,768 masks; every displayed certificate verifies. | Software FAIL blocks scientific use of that operator, not independent tracks. |
| Clean explanation compactness | At least 80% of correctly predicted clean audit cases admit a verified summary within 8 retained seconds and at most 8 signed literals; report full-population coverage too. | Compact explanatory representation not established; exact but huge circuits are not a win. |
| Clean matched additive utility | E1 clean F1 >= E0-0.03 in every eligible cell, plus at least 0.10 improvement over RMS ranking in prespecified localization precision on the single-witness task without lower event coverage. | Clean additive organization FAIL; do not use any-hit localization alone. |
| Stable explanation extension | Unchanged clauses verified on independent nuisance views retain at least 80% validity among cases with unchanged full-input class; always report the fraction whose class changed and all-case coverage. | Limited explanation stability, distinct from classifier robustness. |
| C2 robustness | Clean F1 >=0.80 and >=C0-0.03; competitive F1 >=0.70; diagnostic-swap and nuisance-pair appropriate behavior >=0.80, per task/seed. | Reproduce the failed clause; no all-seed claim from means. |
| Incremental query-consistency benefit | C2 improves competitive F1 over C1 by >=0.03 on the mean of six prespecified cells, positive in at least four cells, with the clean-retention conditions above. | No evidence for the extra objective beyond ordinary consistency. |
| Oracle intervention correctness | Exact target removal and nontarget recomposition under the stored generator, with numerical tolerances verified. | Operator FAIL, not a clinical conclusion. |

These thresholds are proposed decision rules. They are not population significance tests. Report uncertainty intervals and effect sizes alongside all rules. Both mean improvements and worst-cell behavior matter; do not round a failure into a pass.

A circuit may faithfully explain a wrong model. Separate **faithfulness to the model**, **alignment with known generated diagnostic components**, **predictive performance**, **finite-view stability**, and **clinical validity**. No composite score may hide a failed dimension.

### 9.3 Metrics that prevent easy but misleading successes

Report false nuisance selection, support-union duration, localization precision and event coverage; do not reward redundant overlapping windows as multiple independent witnesses. Retain negative contributions and unfavorable context effects.

Use IoU>=0.25 only as the historical interval-hit convention; explain why a 400-sample receptive window and 200-sample event have limited maximum IoU. Report actual interval geometry and duration, not just the binary hit.

For output probabilities report NLL, Brier, ECE15, incorrect confidence>=0.9, and confidence on diagnostic-deleted input. Temperature scaling is calibration-only and does not establish absence detection. No clinical safety or distribution-free guarantee is claimed.

---

## 10. Algorithm and implementation contracts

All names below are **proposed Phase IX interfaces**, not assertions that these functions already exist.

```python
# Physical availability, with receptive-field closure.
atom_to_candidate_mask(atom_mask, candidate_support) -> candidate_mask

# Must match the native predictor when all atoms are visible.
internal_logits(model, cached_features, cached_aux, atom_mask, neutral_state)

# Complete finite truth table on a bounded example/view set.
response_table(model, example_views, target_class, atom_geometry, threshold)

# Exact fixed-order Boolean response representation.
compile_robdd(truth_table, variable_order)

# Retain/withhold clauses checked against all matching completions.
verify_clause(truth_table, positive_atoms, negative_atoms)
extract_readable_clauses(truth_table, deterministic_anchor_policy, max_display=4)

# Disjoint semantic operators; never hide their type.
audit_internal_withholding(...)
audit_generated_component_removal(...)
audit_waveform_replacement(...)
```

Each circuit record must include:

```text
model/checkpoint/config/operator SHA256
input and finite-view identities
physical atom/window geometry and dependency footprints
target class and original full-input prediction
neutral feature-state definition
margin threshold; calibration identity if used
mask search size; exact/partial status
BDD variable order and circuit hash
retained/withheld literal sets
minimum verified margin per clause
subset-minimal versus minimum-size claim
readable-summary coverage
prior-only, no-support, low-margin, or unstable status
original-prediction explanation versus counterfactual-repair flag
verification-view results
clinical_validity = NOT_TESTED
```

Do not expose true synthetic event locations to the blind predictor or clause search. Oracle locations enter only independently defined evaluation metrics and component interventions.

---

## 11. Correctness suite and proofs before expensive experiments

### 11.1 Required unit/property tests

1. Atom/window closure on both ends of the epoch and interior overlaps.
2. Full-input logits equal native logits for conventional and additive models.
3. Arbitrary hidden-sample perturbations cannot alter retained features/output.
4. Hidden RMS, masks, or global summaries cannot bypass withholding.
5. Zero-mask response is explicit and not automatically called uncertainty.
6. Exact compiler agreement on constant, OR, AND, XOR, and blocker truth tables.
7. Signed clause exhaustive verification, including nonmonotone counterexamples.
8. Literal-minimality verification and rejection of unsupported minimum-size claims.
9. Full circuit versus bounded readable-summary coverage.
10. Consistent class-margin behavior under common-logit shifts.
11. Finite-view versus unseen-view certificate labels.
12. No favorable-mask substitution for a wrong full-input prediction.
13. Known random-label and model-randomization sanity controls.
14. Same-encoder hash equality across matched heads.
15. Class-independent query-mask sampling; no oracle metadata in blind paths.
16. No original-class CE on arbitrary insufficient subsets.
17. Exact generated recomposition and matched intervention types.
18. Null conditional donor agreement when no donor is accepted.
19. Atomic checkpoint/creation/seal publication and digest checks before deserialization.
20. Correct model/optimizer/global/minibatch RNG resume and incompatible-resume refusal.
21. Data-role rejection before filesystem access; public output privacy scan.
22. Track independence: robustness FAIL cannot suppress an otherwise permitted clean-head experiment.

Tests should test behavior, not assert an expected number of tests. Preserve historical suites and their original contracts. Historical frozen gates remain unchanged; the new runner has the revised, explicit Track A/B/C dependencies.

### 11.2 Short proof obligations

Write and test the assumptions behind full-input identity, hidden-sample noninterference, exhaustive finite-certificate soundness, and literal subset-minimality. A reviewer should be able to find a concrete counterexample when an assumption is removed.

Do not assume prediction monotonicity under adding candidate features. Do not infer a unique physiological cause from a necessary internal literal. Do not claim a hitting-set/contrastive duality for an incomplete candidate family or a different erasure operator. Established formal explanation results have their own definitions and assumptions [R2, R6].

---

## 12. Resources and stopping policy

### 12.1 Bounded execution

Target two deterministic CPU threads and FP32. Keep current constrained environments and CI unless an actual incompatibility is demonstrated.

Proposed scientific limits:

- Core new fits: 24 matched heads + 18 robustness fits = 42.
- Core optimizer updates: 3,600 cached-head + 5,400 paired-model = 9,000.
- Optional missing-reference reproduction: six additional 300-update fits, explicitly separate.
- Exact finite audit: 60 examples per nominated predictor set, with bounded discovery/verification views and mask batches.
- Total scientific CPU wall ceiling: 3,600 seconds, excluding separately reported CI and environment setup.
- Optional CUDA engineering: at most 120 seconds of checker workload and <=10% of measured GPU memory; no scientific speed claim from cold-start smoke tests.

Before freezing the full execution manifest, run a geometry-only resource pilot on at most two disjoint development examples. Measure the complete mask enumeration, not just a forward pass. If the planned exact audit does not fit, reduce the number of audit examples or views **before freeze**, retaining all seeds/tasks/classes, and record the reduced precision. Do not switch to approximate search while labeling certificates exhaustive.

Use mask batches, cache features, and share deterministic truth tables between comparator analyses. Do not save enormous per-mask tensors publicly. BDDs have exponential worst-case size; stop or return an exact table without a compact explanatory claim if compression fails. [R7]

### 12.2 Checkpoint integrity without another dead end

Only prospectively authenticated synthetic states may be restored. If an old artifact lacks its creation receipt, record historical authentication BLOCKED. Current hashes are inventory only.

This plan allows a **new, sealed, explicitly labeled six-model prospective reproduction** when needed for the core study; it is not permission to fabricate original model identity. Freeze it first, retain the old files unopened, and count its compute separately. Missing optional old baselines should not force rerunning every old phase.

### 12.3 No credentials or interference with other jobs

Query GPU availability only when needed. Never kill another process, request/store administrator passwords, or install a background polling service. GPU contention blocks new CUDA measurements, not the CPU scientific study. Report actual measurements separately from historical CUDA receipts.

---

## 13. Sequential work packages for the coding agent

Use `reports/mist_cove/phase9_v1/progress.json` with one record per task: status, commit, config hash, dependencies, observed outputs, tests, and blocker. Complete the following in order, retaining independent-track behavior.

| ID | Task and actual work | Required output / completion check |
|---|---|---|
| 01 | Create a safe branch/worktree from the reviewed base; copy this plan to repository-root `next_steps.md` without overwriting an unrelated file. | Verified ancestry, clean base, historical artifacts unchanged. |
| 02 | Read S1-S4 and R1-R9; audit the nearest literature and write a short claim-to-prior-art matrix. | At most one focused novelty note; candidate differentiators and unresolved novelty explicit. |
| 03 | Verify prospective reference creation/seal identities; choose restoration or the explicitly new reproduction path before analysis. | Trust inventory; no unauthenticated Phase VI deserialization. |
| 04 | Implement atom-closure and native-head withholding adapters. | Full-input identity and noninterference tests pass on generated fixtures. |
| 05 | Implement exact response tables, ROBDD compilation, signed-clause verification, and the toy-circuit suite. | Exact equality on all masks; no monotonicity shortcut. |
| 06 | Run the bounded resource pilot, fix sample budgets, and freeze Track A/B/C protocols and sample-stream identities. | Committed preregistration plus execution manifest before confirmatory outcomes. |
| 07 | Run frozen-model clean and finite-view circuits on the preselected audit sample. | Track A1 results, missing/large/prior-only certificates retained, same-model comparators. |
| 08 | Run the 24 clean matched-head fits on eligible shared parents, regardless of Track B status. | E0-E3 classification/explanation report, encoder digests, all ineligible cells explicit. |
| 09 | Run the 18 C0/C1/C2 paired-training fits without tuning the primary after scores. | Full loss/view/update accounting, clean/competitive/paired outcomes, sealed states. |
| 10 | Evaluate unchanged discovered clauses on independent nuisance views; distinguish fixed-witness and substitute-witness stability. | Full-input stability plus all-case and conditional explanation coverage. |
| 11 | Execute the five-operator contract audit, using identical component semantics for matched controls. | Separate target removal, nontarget preservation, physical compatibility, and model response. |
| 12 | Run calibration/error strata and oracle localization diagnostics; explain both correct and wrong decisions without relabeling them. | NLL/Brier/ECE, false confidence, null-prior cases, interval precision/coverage. |
| 13 | Run full CPU regression; optional CUDA exact-resume checks when idle; verify all artifact seals. | Actual pass/fail/skip receipts, no inherited CUDA-success claim. |
| 14 | Produce one concise research report, one aggregate dashboard, and only plots needed for the claim matrix. | Truthful track-level decisions; at most six principal figures, no cosmetic file-count target. |
| 15 | Write a bounded future source-only proposal based on actual results, not on whether a composite synthetic score happens to pass. | Specific clinical-development question, required source permissions, risk/limitation review; no data access. |
| 16 | Commit safe outputs, push, open draft PR against the Phase VIII branch, verify CI on the exact final head. | Final commit/PR/CI URLs and completed progress ledger; no automatic Phase X. |

If a software/provenance gate fails, stop that dependency and continue truly independent work. If a scientific hypothesis fails, finish its prespecified comparisons, retain the failure, and complete the other tracks. Never use extra fits to search for a passing seed.

---

## 14. Minimal repository layout

Prefer a small implementation package, not dozens of copied runners.

```text
next_steps.md
src/mist_cove/phase9/
    __init__.py
    operators.py
    circuits.py
    clauses.py
    objectives.py
    experiments.py
    reporting.py
configs/mist_cove_phase9_v1.json
experiments/mist_cove/phase9_preregistration_v1.json
experiments/mist_cove/phase9_execution_manifest_v1.json
scripts/run_mist_cove_phase9.py
scripts/check_mist_cove_phase9_cuda.py
tests/mist_cove/test_phase9_operators.py
tests/mist_cove/test_phase9_circuits.py
tests/mist_cove/test_phase9_tracks.py
docs/mist_cove/PHASE_IX_METHOD_AND_NOVELTY_V1.md
docs/mist_cove/PHASE_IX_RESULTS_V1.md
docs/mist_cove/PHASE_IX_SOURCE_PROPOSAL_V1.md
reports/mist_cove/phase9_v1/
    progress.json
    results.json
    publication_manifest.json
    dashboard.html
    figures/
```

These are proposed paths. Reuse existing provenance, metrics, and checkpoint code after inspecting it. Add files only for a real separation of responsibility. Do not create empty future-phase modules, shell commands that do not exist, or copied historical implementations merely to increase deliverable count.

The CLI may expose `verify`, `circuits`, `heads`, `robustness`, `operators`, and `report`, but none is an existing command until implemented and tested. Default to metadata verification, not training. Reports must distinguish what ran from what was merely prepared.

---

## 15. What a useful final report should look like

### Main results table

For each task/seed/predictor, give:

- Clean and corrupted F1; all-case prediction stability.
- Same-model full-mask fidelity.
- Exact circuit size and readable-clause complexity.
- Summary/certificate coverage, including wrong predictions and empty/prior-only cases.
- Diagnostic-region precision/coverage and interfering-region behavior.
- Fixed-witness versus substituted-witness survival on independent views.
- Track A2 additive versus conventional trade-offs.
- C2 versus C1 incremental robustness and clean-performance changes.
- Operator-specific effects and donor acceptance/null status.
- Actual compute, model size, added-query cost, and cache memory.

### Six useful visualizations

1. A generated waveform with retained atoms, withheld atoms, native receptive fields, and one verified signed clause.
2. OR, AND, and interference example circuits, explicitly generated/model-specific rather than clinical.
3. Clean versus corruption performance for C0/C1/C2.
4. Fixed-witness versus substitution stability and full-input prediction stability.
5. Certificate compactness/coverage versus positive-only and greedy subset summaries.
6. Target-removal versus nontarget-preservation plot for the intervention operators.

Every example is chosen by a frozen rule, not handpicked to favor MIST-WIC. An exact but unreadably large circuit is a reported failure of explanatory compression. A compact circuit supported by nuisance is a faithful model pathology, not a successful diagnostic explanation.

---

## 16. Real EEG translation: a proposal, not an automatic launch

No current step opens the actual EEG folders or loads real fitted banks/checkpoints. Preserve the original 15 TRAIN / 4 development VAL / 55 RESERVED contract and all canonical data proofs. The four VAL participants have been used repeatedly for development; no independent-test claim is available. [S2]

A future source-only proposal should ask a narrow translational question, for example:

> Can witness-interference analysis expose reproducible source-model dependencies and failure cases on real EEG without materially degrading an already competent source classifier?

That question does **not logically require** perfect synthetic corruption robustness. Conversely, perfect synthetic circuits do not establish clinical usefulness. Evidence for an explicitly scoped source pilot, its privacy/provenance controls, measured computational feasibility, and actual user authorization are the relevant requirements.

For strong historical models, inspect receptive-field dependencies first. A whole-epoch feature or unmasked global summary cannot be treated as a local patch. Include it as a global dependency or use a separately reviewed recomputation operator. Do not claim sparse local sufficiency while retaining an unrestricted global bypass.

Initially annotate outputs as model-retained or model-interfering regions. Do not call them sleep spindles, K-complexes, clinical slow waves, or artifacts without independently defined event criteria. Future blinded event annotation could test those concepts. Neither the 55 RESERVED subjects nor ST/SHHS is unlocked by this file.

The new plan changes the future study design explicitly: a failed toy robustness gate is not a universal ban on every scientifically justified real-data question. Nevertheless, the current execution scope is synthetic-only, and a separate source authorization is still mandatory.

---

## 17. Stop conditions and credible outcomes

A credible success is **not** "all tests pass" or "more files were added." It is a supported combination of:

- A precisely scoped same-model explanation object that exposes redundancy and interference.
- A readable subset of that object with verified finite-domain validity and disclosed coverage.
- Better understanding of when explanation support changes across nuisance conditions.
- A direct clean-domain additive-versus-conventional result, even if negative.
- An incremental paired-query training benefit beyond a standard consistency control, if it occurs.
- An intervention audit that no longer confuses target removal with preservation of the target itself.

Valid negative outcomes include no compact circuits, strong neutral-state dependence, explanations dominated by nuisance, no additive benefit, no C2 benefit, or raw-signal and internal-mask effects disagreeing. Preserve each independently. Do not start a ninth synthetic redesign automatically.

**Final execution boundary:** finish this bounded Phase IX package; state which claims are supported; propose only the next justified step; wait for the user before real data or another phase.

---

## 18. Source records and references

### Repository sources -- version-pinned, not independent peer review

**S1.** SyedNaveedMahmood. (2026). *Phase VIII robust evidence study v1: Research closeout* [Repository research report, commit b50e41efb3b09a9cb7fa4142366424725fc603e5]. Proto-sleep. https://github.com/SyedNaveedMahmood/Proto-sleep/blob/b50e41efb3b09a9cb7fa4142366424725fc603e5/docs/mist_cove/PHASE_VIII_RESULTS_V1.md

**S2.** SyedNaveedMahmood. (2026). *Phase III real Sleep-Cassette source experiment v1* [Repository research report]. Proto-sleep. https://github.com/SyedNaveedMahmood/Proto-sleep/blob/4095c805cfd48fbb1e9ef938dbc46da28ed1dfb1/docs/mist_cove/PHASE_III_SOURCE_RESULTS_V1.md

**S3.** SyedNaveedMahmood. (2026). *Phase VII final closeout v4* [Repository research report]. Proto-sleep. https://github.com/SyedNaveedMahmood/Proto-sleep/blob/4262a1a9680f9e8c1747363553204aaeb1ce976b/docs/mist_cove/PHASE_VII_CLOSEOUT_V4.md

**S4.** SyedNaveedMahmood. (2026). *Patch-local learned waveform encoder and capacity-matched decision heads* [Source code, src/mist_cove/phase4/model.py]. Proto-sleep. https://github.com/SyedNaveedMahmood/Proto-sleep/blob/b50e41efb3b09a9cb7fa4142366424725fc603e5/src/mist_cove/phase4/model.py

### External primary literature -- APA style

**R1.** Carter, B., Mueller, J., Jain, S., & Gifford, D. (2019). What made you do this? Understanding black-box decisions with sufficient input subsets. In *Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics* (Vol. 89, pp. 567-576). PMLR. https://proceedings.mlr.press/v89/carter19a.html

**R2.** Ignatiev, A., Narodytska, N., Asher, N., & Marques-Silva, J. (2021). From contrastive to abductive explanations and back again. In M. Baldoni & S. Bandini (Eds.), *AIxIA 2020: Advances in Artificial Intelligence* (Lecture Notes in Computer Science, Vol. 12414, pp. 335-355). Springer. https://doi.org/10.1007/978-3-030-77091-4_21

**R3.** Covert, I., Lundberg, S., & Lee, S.-I. (2021). Explaining by removing: A unified framework for model explanation. *Journal of Machine Learning Research, 22*(209), 1-90. https://jmlr.org/papers/v22/20-1316.html

**R4.** Queen, O., Hartvigsen, T., Koker, T., He, H., Tsiligkaridis, T., & Zitnik, M. (2023). Encoding time-series explanations through self-supervised model behavior consistency. *Advances in Neural Information Processing Systems, 36*. https://doi.org/10.52202/075280-1395

**R5.** Liu, Z., Wang, T., Shi, J., Zheng, X., Chen, Z., Song, L., Dong, W., Obeysekera, J., Shirani, F., & Luo, D. (2024). TimeX++: Learning time-series explanations with information bottleneck. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 32062-32082). PMLR. https://proceedings.mlr.press/v235/liu24bl.html

**R6.** Canizales, R., Gopinath, D., Păsăreanu, C., & Mangal, R. (2026). *Concept-based abductive and contrastive explanations for behaviors of vision models* [Preprint]. arXiv. https://arxiv.org/abs/2605.06640

**R7.** Bryant, R. E. (1986). Graph-based algorithms for Boolean function manipulation. *IEEE Transactions on Computers, C-35*(8), 677-691. https://doi.org/10.1109/TC.1986.1676819

**R8.** Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. In *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 2127-2136). PMLR. https://proceedings.mlr.press/v80/ilse18a.html

**R9.** Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). Theoretically principled trade-off between robustness and accuracy. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 7472-7482). PMLR. https://proceedings.mlr.press/v97/zhang19p.html

DOIs are supplied where verified; otherwise a primary publication URL is used. The R6 source is a preprint, not represented here as a peer-reviewed result. This review establishes overlap, not priority or exhaustive novelty clearance.

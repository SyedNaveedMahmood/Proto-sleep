# Phase XII-A MIST-TRIAD and matched model results v1

Exact circuits successfully characterize this trained classifier under the declared internal operator. They do not establish clinical explanations. TRIAD adds a useful record of mismatches between finite structural dependence, physical modification and explanation persistence; its incremental predictive or clinical utility remains unproven.

All1,350 real response tables passed exhaustive reconstruction:750 zero-neutral originals plus150 .01-neutral and450 fixed-target nuisance views. That is44,236,800 real table rows. No target was changed to the true class or to a nuisance-view prediction. All750 original native/full-adapter comparisons were bitwise identical. The two generated and two real pilot tables were separately verified and accounted.

For the primary150, exact positive-support search finds100 nonempty universally sufficient supports of at most8 seconds. The bounded four-clause display exposes only86 compact active explanations. It exposes111 compact signed explanations when counterfactual states are included; that latter number must not be presented as original-full-input coverage. Twenty-three cases require all30 seconds for any universally sufficient positive support. Twenty-seven original targets also win at the canonical empty state, but zero truth functions are universally constant. Prior response and universal prior-only behavior are different findings.

The median ROBDD has475 nonterminal nodes, maximum1,915. A mathematically exact graph can be too complex for a useful short explanation. The displayed signed summaries are incomplete for138/150 examples and cover only38.34% of true masks on average; the positive-only display covers44.76%. Thus the bounded signed-display heuristic did **not** dominate positive-only summary coverage. The complete ROBDD remains exact. A globally minimum canonical mask is not a universally sufficient positive explanation.

Nonmonotone response loss when adding an atom occurs in139/150 examples; signed interfering literals appear in138. These are conditional model dependencies under artificial feature availability. They are not diagnosed physiological interference. Multiple displayed clauses appear in140 cases, but alternatives may be active only in different counterfactual states. Joint clause sufficiency is not proof that a pair of clinical events is necessary.

Correct predictions have58/99 compact active summaries; wrong predictions have28/51. The participant-level compactness-rate difference is .05119 with95% bootstrap[−.09566,.19963]. This does not support compactness as a correctness certificate. Incorrect predictions have higher median graph complexity and can still have very short canonical retained inputs. The deterministic first examples for failure categories remain private.

Global suppression changes31/150 predictions; global-only features preserve37/150. Sixteen cases retain the original prediction with globals alone while changing under local-only features. This suggests a branch-dependence concern, not proof of a spurious physiological shortcut. Local and global features can both encode valid stage information, and neutral feature states can be out of distribution.

| Comparator | Canonical target retained | Universally positive-valid | Mean retained seconds | Standalone masked-head queries |
|---|---:|---:|---:|---:|
| Native gradient×feature ranking |150/150|48/150|7.253|694 plus gradient access|
| Absolute-gradient ranking |150/150|48/150|8.733|805 plus gradient access|
| Greedy canonical deletion |150/150|41/150|7.920|2,694|
| Random, matched to native ranking size |105/150|31/150|matched|150|
| Random, matched to absolute-gradient size |116/150|38/150|matched|150|

All comparator truth checks share the finite tables; the two gradient rankings share one gradient calculation per example. Their standalone costs are counterfactual cost accounts, not extra training exposures. Native ranking excludes global features from physical-region scores and reports global ablations separately. Verified positive/signed clauses and exact support enumeration provide stronger bounded-domain statements at much higher query cost. Dynamask and other external implementations were reviewed but not fairly reproduced against this unusual internal operator/access budget: NOT TESTED, no superiority claim.

| View | Prediction unchanged | Original first active clause valid | Any original active clause valid | Active substitution |
|---|---:|---:|---:|---:|
| Mild32Hz addition |147/150|109/150|116/150|7/150|
| .2Hz drift |149/150|135/150|139/150|4/150|
| Gain1.05 |145/150|145/150|145/150|0/150|

Counting any signed clause as persistence would inflate these results because a negative-literal clause is not active at the original full mask. TRIAD keeps this distinction explicit. Zero versus .01 neutral changes some mask responses in140/150 examples. Operator sensitivity survives full-input identity and must accompany any explanation claim.

The whole-epoch stress probes yield accuracy .66 for identity/mild HF/transient, .30 for strong HF, .6667 for drift, .4533 for broadband, .68 for gain and .5467 for phase randomization. These are sensitivity outcomes against original labels; stage preservation is not independently known. The separately frozen localized HF follow-up yields.58, motivating the bounded model study.

Of2,250 linear bridge replacements,33 pass the unchanged energy/spectral/boundary thresholds. Six compatible replacements lose the original target; internal removal and bridge replacement disagree in2/33 compatible comparisons. The other2,217 replacements cannot support compatibility-conditioned conclusions. All2,250 background requests lack a different within-epoch atom meeting the unchanged≤3uV RMS eligibility rule, so no background replacement was attempted and donor-conditional prediction effects are NOT TESTED. No historical guard was weakened. Across the nine new models, a fixed50-example post-fit operator check has only7 compatible bridges per model; it is exploratory, sparse and not a new success gate.

The nine prospective fits completed exactly300 updates each, with matched index/atom/phase streams and4,800 clean plus4,800 paired exposures per fit. All new normalization moments came from the ten fit participants'55,268 physical epochs. There were4,884 selected fit examples, retaining116 missing requested stage positions, and1,000 internal-dev examples with no missing positions. No internal-dev or VAL normalization fit occurred.

| Arm/seed | Clean F1 | Local HF F1 | Strong global HF F1 |
|---|---:|---:|---:|
| M0/123 |.369635|.365354|.366795|
| M1/123 |.365722|.355825|.358914|
| M2/123 |.377086|.368738|.381097|
| M0/456 |.393457|.384960|.385872|
| M1/456 |.389750|.385233|.383368|
| M2/456 |.363429|.360501|.363330|
| M0/789 |.434874|.447260|.437264|
| M1/789 |.425756|.435500|.429624|
| M2/789 |.422368|.443119|.435341|

M2−M1 local-HF differences are+.012913,−.024732,+.007619; clean differences are+.011364,−.026321,−.003388. Every seed fails the frozen requirement of local improvement≥.02 with clean penalty no worse than−.02. Mean local difference is−.001400, clean−.006115. No M3, seed replacement, hyperparameter sweep, best-seed promotion or tenth all-TRAIN fit was run.

All arms have poor stage recognition: new-model N1 recall is0–.06, and final classification loss remains about1.05–1.09. The small-capacity, roughly one-pass exposure study cannot distinguish insufficient representation from insufficient optimization. It is a negative result for the registered setting, not a proof that every reliability model fails. Exact additive accounting and nontrivial signed local contributions pass, but arithmetic transparency did not deliver a competent replacement. The global branch remains unconstrained by the local suppression loss. The original model differs in capacity, cohort exposure and selection history, so comparisons with it are reference comparisons rather than matched causal ablations.

Total measured scientific CPU wall is664.292 seconds; no GPU was allocated. There are44,378,416 cached-head queries,236,034 native full-example forwards and1,440 feature-cache example encodings,237,474 combined against250,000. Nine fits used2,700 optimizer updates. Peak measured process RSS is951,032KiB. Source I/O is separately counted, including repeated reads. Generated engineering updates, dependency installation, tests and report rendering are separated. See [resource reconciliation](../../reports/mist_cove/phase11b_v1/resource_receipts.json), [paired uncertainty](../../reports/mist_cove/phase11b_v1/model_paired_uncertainty.json), [all-arm comparison](../../reports/mist_cove/phase11b_v1/model_comparison.json), and [dashboard](../../reports/mist_cove/phase11b_v1/DASHBOARD.md).

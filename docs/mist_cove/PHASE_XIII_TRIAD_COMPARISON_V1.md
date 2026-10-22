# Phase XIII MIST-TRIAD comparison v1

The original availability adapter is unchanged. New WidthLedger models have an independently checked support map: fifteen nonoverlapping200sample atoms; each local encoder/evidence/gate depends only on its atom, while every global feature is withheld unless all atoms are available. Global INPUT neutral0/.01 passes through the global head; bias remains. This is a different operator from suppressing the global output logit and from P3B's overlapping169windows/209features. Independent generated proofs passed128noninterference comparisons,8full tables and4deliberately broken global-path controls.

The real campaign checked3,750tables ×32,768masks =122,880,000finite state responses: original150×5tables, plus36new states×25cases×3base/mild/gain tables and12final states×25neutral tables. Every table's fixed-order ROBDD exactly reconstructs every mask; every displayed signed clause is verified on all completions. Full native input logits are bitwise identical for every case. Total cached queries including auxiliary checks are122,884,350, distinct from full-network passes. No human morphology is inferred from this verification.

| Original model view | Prediction stable /150 | Original clause valid /150 | Active substitution /150 |
|---|---:|---:|---:|
|mild_hf|142|119|5|
|drift|149|141|3|
|gain|148|148|0|

Under mildHF,23cases retain the prediction while invalidating the selected original clause;5have a different original active clause that remains valid. Prediction stability and fixed-explanation stability are different observations. Counterfactual clauses requiring withheld atoms never count as active substitutions.

Original-model zero→.01 neutral changes127/150response tables, averaging1.067%ofstates;34/150minimum positive masks change, with no prior-response class change. These conclusions depend on internal neutral semantics. The full-input prediction stays native by construction.

Original primary physical bridges:2,250attempts preserve every nontarget sample; only42pass unchanged energy≤.5/spectral-L1≤.5/boundary≤5µVguards. Among42compatible probes,11lose the target response and10disagree with single-atom structural withholding. Background requests have0eligible donors out of2,250because the historical≤3µVRMS donor guard remains unchanged. No target physiological event was independently identified or deleted. Ineligible probes are not counted as compatible evidence. Generated-component deletion cannot be claimed on real EEG.

| New family/route | Horizon | Correct /75seed-case observations | Displayed compact active /75 | Global-stable/local-changed /75 |
|---|---:|---:|---:|---:|
|S/R0|300|33|18|21|
|S/R0|1000|41|4|27|
|S/R0|3000|50|10|28|
|S/R1|300|32|18|18|
|S/R1|1000|39|2|28|
|S/R1|3000|50|7|24|
|L/R0|300|39|12|11|
|L/R0|1000|47|0|22|
|L/R0|3000|45|8|16|
|L/R1|300|38|15|10|
|L/R1|1000|48|0|22|
|L/R1|3000|45|8|16|

The75observations above repeat25cases from5people over3seeds; they are not75people. Full1,000case development results improve with exposure and modestly with width, but the25case mechanistic panel does not rank models identically. S/R0 displayed compactness falls18→10/75while correct predictions rise33→50/75. L/R0 finishes45/75correct with8/75displayed compact active supports. Better classification does not establish more compact or physiological explanations. Local/global branch ablations are separate artificial feature experiments; a preserved branch prediction is not a causal or morphological attribution.

Across900new state-case observations, mild-local-HF preserves900predictions but only892selected original clauses; physical gain preserves838predictions and701selected clauses. Fixed original clause validity is recorded separately for every state/seed and can fall below prediction stability. At the final horizon, zero→.01 neutral changes20/75S/R0,17/75S/R1,36/75L/R0 and41/75L/R1tables; minimum positive masks change10/75,8/75,22/75and27/75respectively. Each final family has15compatible bridges among1,125attempts over3seeds, with12/12/13/12structural-versus-physical disagreements, respectively, and no eligible background donors. These are repeated same-case interventions, not independent clinical observations. All neutral and waveform outcomes are retained per family/seed in `explanation_validity.json`, including absent donors and sparse compatible replacements. Detailed graphs, masks, coordinates and margins remain private. Operator disagreement is retained as a result.

The original and new families use different normalization/features/training histories and mildHF definitions. No cross-family clinical robustness win is claimed. Arithmetic local evidence, internal structural support, physical sensitivity and unchanged-clause stability answer different questions. Human alignment, artifact/event deletion and clinical label preservation remain NOT TESTED.

![Support and global dependence](../../reports/mist_cove/phase13_v1/figures/05_support_and_global_dependence.png)
![Original stability and operators](../../reports/mist_cove/phase13_v1/figures/06_original_stability_and_operators.png)

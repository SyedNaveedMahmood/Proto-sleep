# Prospective diagnostic findings v2

**Completed before robustness-method selection. Historical checkpoint verification remains BLOCKED.** These are newly authenticated prospective P0 models, not audited historical Phase VI weights.

| Task | Seed | Clean F1 | Added-nuisance F1 | Oracle-local nonlinear probe F1 | Pooled nonlinear probe F1 | Diagnostic-first / nondiagnostic-second absolute logit path |
|---|---:|---:|---:|---:|---:|---:|
| single | 123 | 1.000000 | 0.930334 | 0.959858 | 0.931858 | 1.5224 / 0.7533 |
| single | 456 | 0.989994 | 0.727618 | 0.844636 | 0.871087 | 1.9765 / 1.0433 |
| single | 789 | 1.000000 | 0.950491 | 0.998000 | 0.996000 | 1.6081 / 0.6167 |
| multiple | 123 | 0.930501 | 0.788362 | 0.856998 | 0.839002 | 1.2974 / 0.3939 |
| multiple | 456 | 0.981984 | 0.706642 | 0.828341 | 0.825966 | 2.9893 / 0.6323 |
| multiple | 789 | 0.996000 | 0.919217 | 0.912985 | 0.947947 | 2.1175 / 0.3563 |

**H7.1: mixed scoped findings.** Every oracle-local nonlinear decoder retains competitive F1≥.80, contradicting complete diagnostic-information destruction for this restricted decoder. Diagnostic-containing candidate changes dominate the controlled logit paths; overlap/interference and distribution sensitivity remain plausible. Large drift alone is not information loss.

**H7.2: the prespecified dominant non-diagnostic-path prediction is CONTRADICTED.** Diagnostic changes exceed non-diagnostic changes in both substitution orders in all six cases. This does not establish that pooling is harmless; oracle pooling substantially worsens some unchanged-head results and is potentially off-manifold. Blind TRAIN-clipped pooling also often worsens performance.

**H7.3: INCONCLUSIVE as a universal head-sensitivity explanation.** Pooled nonlinear probes improve some weak heads by>.10, but not all six comparisons. They also use matched nuisance exposure and separate TRAIN statistics, so the comparison is not a unique causal attribution to the head alone. Exact signed feature/path accounting passes; it is arithmetic bookkeeping.

**H7.4: NOT TESTED until matched training.** Ordinary nuisance replacement and diagnostic removal remain different endpoints; removed confidence remains high, approximately.46–.96 across models.

**Condition scope matters:** the prospective competitive pair adds four independent strong high-frequency bursts and retains the original ordinary artifact. The historical Phase VI competitive renderer also raises artifact amplitude. All six new nuisance-only comparisons exceed.70, but that does not overturn historical failure under its different combined corruption. A robustness follow-up must disclose this distinction, retain an original-renderer combined-stress endpoint and measure strong artifacts separately.

Resource accounting:30fits/9000updates/144000exposures including the preserved first fit. Corrected invocation40.854400s; preceding failed audit3.080431s; known total43.934831s (excluding imports/setup). Peak corrected-process RSS1,388,460KiB. No original checkpoints were loaded. All six new states have checkpoint/seal/creation receipts with optimizer and RNG.

No new donor procedure was selected; accepted-donor robustness is NOT TESTED and original0/210 remains unchanged. No real EEG or external cohort was accessed. Full safe aggregate metrics and signed paths are in `reports/mist_cove/phase7_v1/prospective_diagnostics_v2.json`.

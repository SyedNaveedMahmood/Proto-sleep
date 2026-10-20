# Phase X measured resource feasibility v1

These are CPU measurements of in-memory random historical raw_context architectures, not the actual trained source checkpoint. No EEG IO, fitted bank retrieval, source deserialization or training is included. Three fixed seeds ×three repetitions; torch2.5.1+cpu, Python3.12.9, two CPU threads. Final raw measurements are in `resource_estimates_guard_v4.json`; the original `resource_estimates.json` is preserved separately. The final guard rejects altered internal normalization/topology and late hooks; its overhead is included in these measurements. The final architecture-guard-v4 benchmark enclosing internal wall was4.123352s and process high-water RSS662,532KiB; RSS includes imported libraries, not just tensors.

| Operation | Median seconds |
|---|---:|
| One native prediction, including encoding | 0.002489 |
| Cache construction | 0.002080 |
| One cached masked prediction | 0.001599 |
| 512 cached masks | 0.004686 |
| Masked features plus global suppression | 0.001482 |
| Complete32768-mask table | 0.304813 |
| ROBDD on native random mock tables | 0.000111 |
| Clause extraction/completion verification | 0.004967 |
| Circuit/summary atomic serialization | 0.003603 |

Cached masking includes identity/content checks and geometry-based aggregation; it is not32768 full-network encodings. Each default cache is21,700 bytes/example (169×32 local floats plus17 globals).150 resident caches would use3,255,000 tensor bytes, excluding Python/storage overhead. A float32 margin and Boolean truth table use163,840 bytes/example; the shared mask grid491,520 bytes. These tensor counts are exact, not process-memory projections. Bank/state provenance and device transfer are excluded.

Nine complete timed tables plus three prespecified .01-neutral sensitivity tables add393,216 availability queries. Native warmups3, timed native predictions9, single masked predictions9, batch masked queries4,608 and full-target queries12 are separately counted. The initial17-fixture study adds557,056 table queries. The final architecture-guard benchmark repeats the same fixed seeds/repetitions in a separate preserved namespace, adding another393,216 table queries. Total measured scientific table queries across study and both benchmark versions:1,343,488. This rerun measures changed verification overhead, not improved outcomes or a new seed search. Native-gradient comparisons use16 actual head backpropagations across17 successful rows: the literal constant-bias row needs no backward operation. The original constant row counter was corrected in accounting freeze v3 without rerunning tables or attributions. An initial constant-bias export stopped and its partial backward count is unavailable. No training updates occurred. Generated regression tests are separate from these scientific query counts.

The .01 control changes1/32768 Boolean responses at seed123 and none at456/789; original full-input targets are unchanged. This small finite change does not establish neutral invariance of the trained classifier. Default zero remains primary.

Native random tables are unusually simple (zero or15 nonterminal nodes). A separately generated random Boolean stress table has4,318 nonterminals and bounded displayed coverage. Its three repetitions measure compiler/summary/serialization scaling, not a plausible EEG response. The theoretical full binary-tree ceiling is32,767 nonterminals; the measured random stress is not a worst-case bound.

For150 examples, cache+complete table+BDD+clauses+serialization yield these **conditional CPU scenarios**:

| Scenario | Projected seconds | Basis |
|---|---:|---|
| Observed fast |46.879|150×fastest measured native pipeline |
| Observed expected |47.320|150×median measured native pipeline |
| Conservative stress scenario |119.386|150×twice(sum of maximum observed cache/table times and random-Boolean compiler/summary/serialization times) |

These are neither confidence intervals nor guarantees for real weights, real circuit complexity, source IO or shared hardware.150 examples were not run. The prespecified600s mock scientific budget was not approached by the measured stages; proof/test/setup/report times are separately scoped, and no exact total kernel/core-time claim is made. Failed comparator command enclosing wall3.389663s is conservatively recorded separately; isolated failed-comparator time is unknown.

CUDA is **BLOCKED**: another compute process occupied the RTX4080SUPER (14,532MiB used,1,413MiB free of16,376MiB at the single necessary query). No process was interrupted, and no new CUDA model operation ran. Idle-only synthetic smoke is prepared with60s and10% allocator ceilings; it supplies no actual GPU measurement here. A later separately authorized two-example actual-artifact resource pilot is mandatory. One GPU-hour remains a proposed policy ceiling, **not established feasibility**. CPU architecture timing cannot be converted into GPU throughput or licensed source-audit readiness.

The preserved pre-guard benchmark took4.085524s, with initial150-example scenarios46.895/47.406/115.886s. It is not silently replaced or added to throughput. The final run took4.123352s with the guard active; both campaigns and their393,216 queries each count toward execution accounting. No CUDA availability re-query or allocation was needed for this CPU-only follow-up.

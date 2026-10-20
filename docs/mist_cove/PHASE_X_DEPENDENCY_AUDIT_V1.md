# Phase X source dependency audit v1

Gate X-A: code-level geometry established; no source artifacts opened. Source hashes and all 169 half-open window supports are in `reports/mist_cove/phase10_v1/dependency_summary.json`. The reviewed definitions are historical `EvidenceModel`, `LocalEncoder`, `RawTemporal`, `variant_config`, `make_baseline`, `training_bank`, and the Phase II adapter. Phase II's q-model is a different architecture and cannot supply this proof.

`training_bank` constructs `ModelConfig(radius=0,use_crf=False)`; `make_baseline('raw_context',...)` preserves its bank configuration and enables raw_control. The default bank config is scales100/200/400, stride50, dim32, 16 anchors per scale. The historical source checkpoint configuration is not independently authenticated by this code audit. Unsupported temporal/CRF variants fail closed.

| Computation | Input → output | Original sample support | Normalization / epoch statistics | Aggregation | Parameters/buffers | Availability action |
|---|---|---|---|---|---|---|
| Fixed source normalization | calibrated [N,1,3000] → same | each corresponding sample | fixed TRAIN mean/std; no per-input epoch fit | none | external TRAIN lineage required | adapter receives normalized input; no fit/IO |
| Native unfold at each scale | [N,3000] → [N,P,L] | [50j,50j+L) | none | none | bank waveform length, cfg stride | require every intersected atom |
| Window centering/RMS clamp | [NP,L] → same | complete individual window | data-dependent window mean and RMS; clamp1e-12 | within window only | none | cached embedding withheld as a unit |
| Conv7 stride2 + GroupNorm4/16 + GELU | [NP,1,L] → [NP,16,ceil(L/2)] | complete window conservatively | GroupNorm reduces channels and temporal positions within each window, not batch | no cross-window mix | learned conv/norm weights and biases | unit window suppression |
| Conv5 stride2 + GroupNorm4/32 + GELU | previous → [NP,32,ceil(ceil(L/2)/2)] | complete window | within-window GroupNorm | no cross-window mix | learned conv/norm | same |
| TemporalMean + Linear32→32 + L2 normalization | previous → [NP,32] | complete window | L2 over embedding, eps1e-6 | time within window | linear weights/bias | same |
| Mean over windows | [N,P,32] → [N,32] | union of scale's windows; full epoch in default geometry | none | native mean / P | no extra state | fixed native denominator, invalid embeddings zero by default |
| Max over windows | [N,P,32] → [N,32] | union of scale's windows | none | native coordinate max, embeddings may be negative | no extra state | exclude invalid windows; empty→fixed neutral |
| epoch_summary normalization | [N,3000] → same | all3000 | epoch mean/RMS | whole epoch | none | all atoms or neutral |
| Full Hann FFT relative five-band power | [N,3000] → [N,5] | all3000 | epoch normalization, band mass +1e-8 and normalization | full epoch | deterministic Hann, frequency bands | all atoms or neutral |
| Six500-sample spectral summaries | [N,6,500] → [N,10] | all3000 after native normalization and std/max over all six segments | per-segment normalization/band fractions | std(unbiased=False) and max over6 | no fitted buffers | all ten features globally gated |
| Squashed normalized line length | [N,3000] → [N,1] | all3000 | epoch normalization | mean of2999 differences | none | globally gated |
| Centered log-RMS, fixed TRAIN scaling, sigmoid | [N,3000] → [N,1] | all3000 | epoch centered RMS; fixed amplitude_stats | whole epoch | two TRAIN-fitted amplitude scalars | globally gated; never inspect real scalars here |
| Concatenation | 3×(32mean+32max)+16+1 → [N,209] | union of explicitly above supports | none | fixed historical order | feature_names | trace per-path availability |
| RawTemporal project209→64 | [1,N,209] → [1,N,64] | per-epoch feature inputs | no input-dependent epoch preprocessing | within feature channels | learned weight/bias | unchanged historical head |
| Two residual pointwise Conv1d64→64/GELU/LayerNorm64 | previous → same | same single epoch for radius0 | LayerNorm within64 hidden units per epoch | no other epochs; kernel1 | learned weights/biases | unchanged head on masked feature vector |
| Classifier64→5 | previous → [1,N,5] | same | none | five logits | learned weights/biases | Wake/N1/N2/N3/REM order retained |

WaveformBank's prototype scores, observed/neural metric parameters, anchor encodings and inverse temperatures are bypassed by raw_control. Its waveform *shape* determines unfold length, while values are unused in this forward path. Banks still exist in state_dict and require historical provenance later; this is not permission to replace a real bank or claim its authenticity. Synthetic bank tensors are generated in memory here. RawTemporal radius0 and no CRF are mandatory: radius10 or sequence decoding would add inter-epoch dependencies.

The 100/200/400 scales yield 59/57/53 windows. A 400-sample window starting at50 touches three atoms; 100/200-sample windows can cross two. The map uses actual half-open support intersections, not the Phase IX two-atom shortcut. Global spectral summaries remain conservatively global even where algebraic cancellations could sometimes remove normalization dependence; no unproved cancellation narrows their support.

Native aggregation is a global function of local representations. The partial operator replaces that aggregation explicitly; it does not pretend the pooled feature is originally local. A frozen external TRAIN affine transform preserves sample locality, but an unknown fitted or recomputed scaler cannot be accepted. This proof is conditional on the exact reviewed implementation, not arbitrary model subclasses or mutated encoders.

Final guard v4 enforces this conditional proof: module types, topology, normalization settings and feature names must match a newly constructed historical skeleton using only generated ramp buffers. Unknown/altered layers and hooks fail before caching; late hooks/overrides fail at query time. The reference is not attached to the classifier, and validation preserves caller CPU RNG. This prevents an altered encoder such as cross-window BatchNorm from inheriting the local-support declaration. It still does not authenticate real weights or defend arbitrary malicious runtime code.

# Phase XVI source and scientific review

**Status: SOURCE ACCESS BLOCKED; synthetic infrastructure tested; human-event results NOT TESTED.** This report uses public owner documentation and already published MIST-COVE aggregate evidence. It does not include an external EEG download, a human annotation file, or a new model result.

## The unresolved claim

Phase XV established exact finite response tables for the authenticated historical P3B Fpz-Cz sleep-stage classifier. It did not establish that the positive supports identify expert-scored EEG events. In an exploratory comparison against an automated Pz-Oz spindle-like proxy, exact support was worse than equal-duration sigma selection: 0.171 versus 0.217 precision in 55 eligible cases from 20 people, with person-level difference −0.0462 [−0.0797, −0.0167]. The registered Phase XIV sigma gate had also failed. These are proxy comparisons on previously exposed ST data, not independent human-event results. See [Phase XV](../PHASE_XV_RESEARCH_REPORT_V1.md).

The Phase XVI questions are therefore separate: whether an automatic detector matches human-marked events, whether exact explanation machinery can audit a separately trained compatible event model, and whether the original P3B model's explanations correspond to physiological events. The third question cannot be answered by simply inputting central DREAMS EEG to a classifier trained on Fpz-Cz.

## Original DREAMS source and rights

The original [Zenodo record](https://zenodo.org/records/2650142) publishes the four named archives and owner MD5 values. The owner [DREAMS documentation PDF](https://zenodo.org/records/2650142/files/The%20DREAMS%20Databases%20and%20Assessment%20Algorithm.pdf?download=1) was read as a public source document; its bytes matched the Zenodo-published MD5 `2e7bb815026bd32fd019c8f51eb1bffd`. The owner [license text](https://zenodo.org/records/2650142/preview/Dreams%20License.txt?include_deleted=0) and PDF identify an Attribution–NonCommercial–NoDerivs 3.0 license. It restricts commercial use and adaptations, requires attribution for distribution, and does not itself resolve this project's contemplated private derivative processing, model training, trained-weight release, or aggregate publication. The owner's page says the recordings were originally collected to train and test detection algorithms; that historical description is not a project-specific rights grant. The user has said project permission is **“No permission or unclear.”** Consequently the acquisition, extraction, real benchmark, and event-model gates are blocked. No mirror is a substitute for owner permission. The narrow resolution is a documented project-use classification and, if needed, written owner permission covering the planned private signal transformations, model fitting, and aggregate publication. This report is a research access decision, not a legal opinion.

The source record lists owner MD5 checksums for the four approved archives. SHA256 values cannot be claimed until original bytes have been acquired and measured. The restricted downloader checks both. It is not invoked in this phase. No archive inventory has been inferred from a mirror.

## Candidate suitability and source discrepancies

| Candidate | Owner-reported coverage | Scientific use if access passes | Current gate |
|---|---|---|---|
| DREAMS spindles | Eight 30-minute central-EEG excerpts; expert 1's counts cut off after 1,000 s; expert 2 scored only six excerpts. Native rates 50/100/200 Hz, C3-A1 or Cz-A1. | Provisional event benchmark using actual jointly covered intervals, then reviewer-specific sensitivity. | Rights and actual-file audit blocked. |
| DREAMS K-complexes | Zenodo summary says **five** excerpts; owner PDF describes **ten**, with expert 2 scoring the first five. Native 200 Hz central EEG. | Operational inventory must resolve the count; two-reviewer analysis cannot assume ten jointly scored. | Discrepancy open; rights blocked. |
| DREAMS rapid eye movements | Nine 30-minute excerpts from **five people**, one expert; native 200 Hz PSG with EOG. | Event-level EOG benchmark, not REM-stage accuracy. Person grouping matters. | Rights blocked. |
| DREAMS artifacts | Twenty 15-minute PSG excerpts, one expert; artifact type and channel appear in bracketed text sections. | Category-specific proxy analysis; the annotation vocabulary is not exhaustive quality truth. | Rights blocked. |
| MODA | Published [primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7305234/) reports 180 subjects and human spindle consensus; annotations are on OSF, while waveforms derive from MASS. | Potential future larger benchmark once waveform rights and participant identity are resolved. | No waveform access authorized. |
| MASS | The [owner access page](https://ceams-carsm.ca/en/mass/) permits some annotation downloads, but biosignals require a project description and local ethics-board approval, then acceptance of a use license. | Possible future source after formal access. | No download authorized. |

The owner PDF documents two-column start/duration text for the four visual-scoring tasks. The artifact files add bracketed artifact-type/channel sections. It reports no uniform cross-archive participant linkage. Exact identifiers and overlap therefore remain UNKNOWN until licensed archive inspection. A published excerpt count is not proof of participant-disjointness. For the primary spindle task, unscored time after expert 1's cutoff and excerpts absent for expert 2 are `UNKNOWN_ANNOTATION_COVERAGE`, never covered negatives. A later archive audit must verify exact duration, header units, channels, scorer files and file-level time origins before any metric is valid.

## Compatibility and decision

Historical P3B receives 100-Hz, 30-second calibrated **Fpz-Cz**, with its original training normalization and sleep-stage target. DREAMS spindle and K-complex annotations refer to **C3-A1/Cz-A1**, variable native sampling, and event intervals. Their signals cannot become Fpz-Cz by resampling, amplitude scaling, or renaming. `ORIGINAL_P3B_EXTERNAL_EVENT_VALIDATION = NOT_TESTED_CHANNEL_INCOMPATIBLE`. The exact internal P3B audit remains valid for its own operator but is not clinical event validation. A separately identified central-channel event model could test explanation-method generality only after rights, coverage and person-disjointness pass.

The present strongest claim remains: exact dependency auditing exposes a model's response to a declared internal withholding operation, including alternatives and interference. Physiological localization beyond sigma, event-model generality on real annotated data, clinical validity, and the cause of ST REM failures remain untested. A methods paper with candid negative results may be supportable from prior phases; an expert-event validation paper is not supported by Phase XVI data.

## Next independent scientific decision

Resolve DREAMS rights with the owner, then audit the four original archives and lock a participant-grouped benchmark before inspecting held-out event outcomes. If rights cannot be obtained, seek an owner-approved alternative with both waveform and interval-label access. A separate REM study should compare matched Fpz-Cz-only and EEG+EOG+EMG classifiers on a genuinely independent boundary; it should not reuse the exposed ST outcomes for candidate selection or interpret EOG events as complete REM-stage labels. No Phase XVII experiment is initiated here.

# EDF scoring and physical alignment

The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`.

The loader disables pyEDFlib annotation reading in Stage A and permits only waveform file identities. Generated EDF+ tests forbid annotation access and verify complete 30-second/3,000-sample ordering. A strict one-time scoring claim is persisted only after content verification of the sealed predictions and committed public seal receipt. Subsequent attempts reject; interrupted scoring stays exposed.

The [EDF+ standard texts](https://www.edfplus.info/specs/edftexts.html) govern explicit Sleep stage W/N1/N2/N3/R mappings. Unknown/movement intervals map only as registered. Nonfinite, negative, fractional-grid, overlapping, repeated, contradictory and out-of-record stage intervals reject. Gaps remain unscored. Absolute waveform/scoring start times and complete EDF+/text event agreement are required; no label-dependent repair is permitted.

The real HMC scoring text schema is not verified. The implementation supports the explicitly named generated CSV fixture schema and fails closed on other formats. No real annotation has been opened to extend that parser. Full original-file verification and schema compatibility are outstanding gates, not passing tests or established clinical alignment.

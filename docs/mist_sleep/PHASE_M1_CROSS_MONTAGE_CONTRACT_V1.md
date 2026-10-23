# Clinical and cross-montage contract

The primary outcome is **`PRIMARY_EVALUATION_BLOCKED`**. No HMC waveform payload, original scoring outcome, target prediction or clinical metric was read or produced. All clinical values are null. The Phase M0 decision remains `MPI_FEASIBILITY_NOT_ESTABLISHED`.

The source is SC Fpz-Cz at 100 Hz; primary HMC is C4/M1 and secondary HMC is F4/M1 at 256 Hz. This combines cohort, clinical physiology, acquisition system, reference and electrode-location changes. Resampling is not montage correction. No missing electrode is synthesized, no polarity is selected after errors and no target mapping is fitted.

Physical EDF calibration precedes conversion to uV. Continuous per-record 25/64 rational resampling uses the explicit symmetric 1,281-tap FIR, fixed Kaiser beta 5, cutoff 1/64, float64 arithmetic and zero padding at physical record ends. The grid begins at original t=0; only the incomplete trailing epoch is discarded. Source-fit mean/SD are used directly, without target moments, BatchNorm adaptation or stage-conditioned transforms. Generated tone/impulse/anti-alias/constant-offset controls pass. Actual HMC conversion remains unexecuted.

All headers have both intended electrode pairs at 256 Hz/uV, but whole-file and annotation verification remain blocked. Provider lights-off/on clipping differs from SC complete physical recording treatment. Neither success nor failure on this particular shift would isolate its individual causal components.

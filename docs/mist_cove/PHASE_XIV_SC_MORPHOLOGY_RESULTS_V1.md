# Phase XIV: SC TRAIN automated morphology

The authenticated SC TRAIN scope has 15 people, 29 PSG/hypnogram pairs, 79,984 physical 30-second epochs and 79,025 scored epochs. All pairs passed original-file checksums, strict annotation geometry and source-channel review. No detector execution failed. Detection ran on continuous calibrated channels without stage labels or classifier predictions. YASA 0.6.5, fixed 30-minute cores with padding, and the exact frozen settings are in `configs/mist_cove_phase14_v1.json` and `detector_manifest.json`.

| Fpz-Cz / Pz-Oz proxy | Candidate events | Pooled cross-channel F1 | Interpretation |
|---|---:|---:|---|
| Spindle-like default | 21,596 / 24,632 | 0.5601 | Moderate agreement between two correlated EEG views |
| Spindle-like stricter RMS | 20,298 / 22,813 | 0.5714 | Modest sensitivity to detector setting |
| Slow-wave-like default | 96,746 / 10,747 | 0.0643 | Strong montage/detector disagreement |
| Slow-wave-like relaxed amplitude | 340,959 / 51,575 | 0.1381 | Even relaxed candidates disagree markedly |

The default spindle detector gives 32.48 Fpz-Cz and 36.38 Pz-Oz candidate events per eligible hour on participant means. The corresponding slow-wave rates are 145.30 and 15.80. These are automated candidate rates, not expert verified physiological event rates. Low slow-wave cross-channel agreement means a waveform label from either view is unsuitable as unqualified ground truth. The separate sigma/delta controls and event duration/frequency/amplitude distributions are preserved in `sc_proxy_aggregates.json`.

EOG excursions and chin EMG summaries are useful independent descriptors of error strata, but the SC chin EMG is a native 1 Hz processed envelope. It cannot support raw 20–45 Hz muscle-frequency conclusions or direct rate matching to ST's 100 Hz raw EMG. Artifact flags include flatness, steps, clipping/extremes and channel association. Their presence cannot establish a particular event as true or false. A generated positive/flat detector fixture, filter frequency checks and a private automated proxy atlas supply implementation checks; no qualified human reviewed the events.

The flat generated fixture initially produced a missing coherence value that strict JSON rejected. That engineering failure and its null-value correction were preserved before source detection. The wheel contains unused pretrained classifier assets; this investigation invoked deterministic event algorithms only. The exact package and asset clarification are published separately. Human event validity remains **NOT TESTED**.

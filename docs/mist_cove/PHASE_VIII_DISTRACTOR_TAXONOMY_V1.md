# Phase VIII distractor taxonomy v1 — frozen before diagnostic execution

The unchanged Phase VI `generate` function supplies complete 3,000-sample physical tensors and latent records. Its `clean` condition already includes moderate high-frequency nuisance and small artifacts. Phase VIII "clean" retains that historical meaning. `intervene` adds separately generated nuisance; it does not edit the Phase VI definition. Labels, baseline realization, diagnostic events and locations are exactly matched across intensity, except for the explicitly named weak-diagnostic regime (events scaled .2 at every strength).

All new random draws use stream 8801 with original seed/index, independent of class. Strengths are exactly 0, .5, 1, 2. Default waveforms: 200-sample tapered unit-RMS bursts, scaled 40 uV RMS; low frequency 1 Hz, other oscillations 30 Hz. Box transients last 20 samples at 100 uV; impulses 3 samples at 100 uV; broadband uses Gaussian tapered bursts at 40 uV RMS. Sustained oscillations span 3,000 samples at 12 uV RMS. Correlated events reuse phase, not class. Multiple/competing use four events. A count/position-based taxonomy is not a claim of mutually exclusive physiological artifact classes.

| Name | Controlled construction |
|---|---|
| high_frequency | random 30-Hz local oscillation |
| low_frequency | random 1-Hz local oscillation |
| transient | 20-sample signed box |
| impulse | 3-sample signed box |
| broadband | tapered independent Gaussian burst |
| sustained | full-epoch amplitude increase via 30-Hz oscillation |
| multiple | four independent-phase local oscillations |
| correlated | four shared-phase local oscillations |
| overlapping | starts at first diagnostic interval |
| disjoint | samples positions disjoint from every diagnostic interval |
| boundary | starts at sample 0 or 2800 |
| inside_candidate | centered inside a four-second candidate grid cell |
| crossing_candidate | straddles a four-second grid boundary |
| weak_diagnostic | .2 diagnostic amplitude with random strong nuisance |
| competing | four distractors with intact single or multiple diagnostic components |

The two four-event independent families intentionally share construction; their names support semantic grouping, not independent replication. Overlapping candidate windows mean a grid-boundary artifact can be inside another candidate. Placement is stated relative to the four-second grid, not claimed disjoint from every overlapping patch.

Diagnostic endpoint: six sealed P0 states, 100 balanced draws/task/seed, seed offset 8,200,000. Train-only privileged linear probes use offset 8,100,000 and 500 generated examples, 150 batch-16 updates, balanced clean/high-frequency mixture. Every taxonomy/strength retains fixed-five metrics, margin, entropy, NLL/Brier/ECE15, incorrect confidence >=.9, draw-paired accuracy bootstrap (500 resamples), candidate-set norms/drift and separability. Strength 1 adds mean/max/combined/trimmed/bounded/oracle unchanged-head contrasts, exact logit/path decomposition and privileged local probe scores. This is a descriptive mechanism audit; it makes no exclusive causal attribution.

Eventless feature norms are measured separately. Diagnostic-containing, nuisance-only, mixed and background sets use explicit support overlap and partition all fourteen candidates. Candidate sets overlap in physical input and are never independent experimental units.

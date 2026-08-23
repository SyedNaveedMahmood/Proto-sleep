# MorphMAE representation redesign: MorphSpec-R1

## Why this experiment exists

The downstream-transfer explanation has now been tested and rejected on development folds 0..4.

- Frozen MorphMAE MRCNN vs frozen random MRCNN: mean probe gain = -0.029388, positive in 1/5 folds.
- Staged unfreeze at encoder LR 1e-4 vs standard A1: mean = -0.036610, positive in 1/5 folds.
- Staged unfreeze at encoder LR 2e-5 vs standard A1: mean = -0.042136, positive in 1/5 folds.

Therefore the next scientific question is not whether a smaller downstream LR rescues MorphMAE-v2. The representation itself is insufficiently stage-informative under the current probe.

## MorphSpec-R1 hypothesis

MorphMAE-v2's reconstruction diagnostics showed broad waveform/spectral preservation but did not establish that its encoder makes sleep-stage morphology easily accessible to a downstream classifier. MorphSpec-R1 adds a second, label-free representation-refinement stage starting from each leakage-safe v2 encoder.

The encoder receives a patch-masked 30-s EEG epoch and predicts targets computed from the clean epoch:

1. log relative power in delta, theta, alpha, sigma, and beta bands;
2. within-epoch standard deviation of those log relative powers across six 5-s segments;
3. within-epoch maximum of those segment-level log relative powers;
4. normalized line length.

These targets are amplitude-robust and are intended to expose stage-relevant spectral and transient morphology directly in the MRCNN representation. The refinement does not read or use sleep-stage labels.

Frozen recipe:

- initialize from fold-specific leakage-safe MorphMAE-v2;
- 30 epochs;
- batch size 128;
- patch size 25 samples;
- mask ratio 0.30;
- encoder LR 2e-5;
- morphology-head LR 1e-3;
- AdamW;
- FP32 refinement;
- train subjects only; validation and designated test files are not opened.

This is an exploratory redesign on already-observed development folds 0..4. It cannot support a confirmatory claim by itself.

## Next gate

After refinement, run only the frozen-representation probe. The random-probe reference is reused from the completed transfer-recovery audit.

Exploratory representation gate:

- mean fold-level `MorphSpec probe - random probe > 0`; and
- positive in at least 3/5 development folds.

If the gate passes, freeze MorphSpec-R1 and confirm it on new folds with newly prepared fold-specific source checkpoints before making an abstract claim. If it fails, do not keep tuning downstream transfer; the next redesign must change the representation-learning objective more substantially.

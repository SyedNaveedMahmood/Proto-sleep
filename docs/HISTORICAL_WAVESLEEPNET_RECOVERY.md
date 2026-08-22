# Historical WaveSleepNet A3/A4 Recovery Gate

## Why this exists

The five-fold screening of the modern spherical `ProtoAttnSleep` implementation did not support a stable MorphMAE benefit. That experiment is informative about the current prototype architecture, but it is not an exact replication of the historical WaveSleepNet prototype mechanism.

The historical source audit recovered a different mechanism:

- `ProtoPNet` from `external/WaveSleepNet-main/models/protop.py`;
- 10 learned prototypes;
- waveform-distance / wave-similarity pathway;
- prototype diversity, identity, distance, cross-entropy, and classifier-weight terms;
- the original Sleep-EDF-2013 WaveSleepNet training configuration;
- a documented MorphMAE bridge that requires `afr_reduced_dim=30` and `prototype_shape[1]=30` for strict MRCNN transfer.

The recovery therefore proceeds in gates rather than immediately launching another expensive training sweep.

## Gate 1: frozen-source compatibility smoke

`smoke_legacy_wavesleepnet.py` does **no training**. It:

1. verifies SHA-256 hashes for the audited historical model, trainer, loader, utility, integration patch, and Sleep-EDF-2013 config;
2. verifies the audited historical prototype/loss/training hyperparameters;
3. changes only the documented `27 -> 30` MRCNN/prototype channel bridge in memory;
4. imports the historical `ProtoPNet` implementation directly from the legacy codebase;
5. constructs matched A3/A4 models from one template;
6. strictly loads a leakage-safe fold-specific MorphMAE MRCNN into A4;
7. verifies all non-MRCNN A3/A4 parameters are identical at initialization;
8. loads only training-subject EEG for the smoke batch;
9. checks the legacy MRCNN output is `[B,30,80]` and the full model emits 5-class logits.

Run fold 0 first:

```bash
python scripts/smoke_legacy_wavesleepnet.py \
  --legacy-root "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --data-dir "/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset" \
  --fold 0 \
  --seed 123 \
  --mae-checkpoint mist_sleep_runs/morphmae_pretrain/fold_00/ssl_seed_1337/encoder.pt \
  --report legacy_wavesleepnet_smoke_fold0.json
```

Expected terminal gates:

```text
STRICT MRCNN LOAD: PASS
NON-MRCNN INITIALIZATION MATCH: PASS
FORWARD PASS: PASS
HISTORICAL WAVESLEEPNET COMPATIBILITY: PASS
NO TRAINING STARTED: YES
```

Do not use `--allow-source-drift` for confirmatory experiments. It exists only to diagnose intentional edits to the archived legacy source.

## Next gate

Only after Gate 1 passes should the repository add the matched historical WaveSleepNet training objective and validation-only fold runner. That runner must preserve the audited loss coefficients and optimization settings, use the current subject-level split policy, keep designated test subjects untouched, and compare random-MRCNN A3 against strict MorphMAE-MRCNN A4 under matched non-MRCNN initialization.

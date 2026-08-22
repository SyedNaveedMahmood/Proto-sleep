# MIST five-fold validation screening result (2026-08)

This note records the first leakage-safe five-fold replication of the current MIST mechanism audit on Sleep-EDF-20.

## Protocol

- Folds: 0,1,2,3,4 using the repository's AttnSleep subject permutation.
- One fold-specific MorphMAE-v2 SSL checkpoint per fold, pretrained only on that fold's 18 training subjects.
- SSL seed: 1337.
- Supervised seeds: 123, 456, 789.
- Validation subjects only were evaluated.
- Designated test subjects were not evaluated.
- `A1`: AttnSleep.
- `A3_current`: current `ProtoAttnSleep` with random MRCNN initialization.
- `A4_current`: the same current prototype model with a fold-specific MorphMAE-v2 MRCNN initialization.
- `A3_current` and `A4_current` are matched at initialization for all non-MRCNN parameters.

## Seed-averaged fold results

| fold | A1 F1 | A3_current F1 | A4_current F1 | A3-A1 | A4-A3 | A4-A1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.822220 | 0.813167 | 0.820646 | -0.009053 | +0.007479 | -0.001574 |
| 1 | 0.762661 | 0.767205 | 0.774596 | +0.004545 | +0.007390 | +0.011935 |
| 2 | 0.665054 | 0.681233 | 0.662869 | +0.016179 | -0.018365 | -0.002185 |
| 3 | 0.694207 | 0.704410 | 0.688037 | +0.010203 | -0.016373 | -0.006170 |
| 4 | 0.820878 | 0.821651 | 0.816090 | +0.000773 | -0.005561 | -0.004788 |

## Fold-level paired summary

| effect | mean | SD across folds | median | positive folds |
|---|---:|---:|---:|---:|
| A3_current - A1 | +0.004530 | 0.009572 | +0.004545 | 4/5 |
| A4_current - A3_current | -0.005086 | 0.012425 | -0.005561 | 2/5 |
| A4_current - A1 | -0.000556 | 0.007231 | -0.002185 | 1/5 |

## Decision

The predeclared screening rule for advancing the current A4 mechanism was:

1. mean fold-level `A4_current - A3_current > 0`, and
2. `A4_current > A3_current` in at least 3/5 folds.

The current implementation fails both criteria. The mean A4-vs-A3 effect is negative and only 2/5 folds are positive.

This result does **not** establish that MorphMAE is universally harmful and does **not** falsify the original historical MIST implementation. The current repository's spherical `ProtoAttnSleep` is not claimed to be byte-for-byte identical to the historical WaveSleepNet-derived prototype pathway. The current result therefore supports freezing `A3_current/A4_current` as a completed screening branch and recovering the exact historical prototype implementation before deciding whether the original A4 mechanism deserves another clean replication.

A separate observation is that `A3_current - A1` is positive in 4/5 folds with a small positive mean. This is descriptive development evidence for the current prototype pathway, not a test-set or external-transfer result.

## Next action

Run the read-only historical prototype audit:

```bash
python scripts/audit_legacy_prototype.py \
  "/home/FA006/Desktop/transfer/MorphMAE_Sleep_Codebase" \
  --report legacy_prototype_audit.txt \
  --json legacy_prototype_audit.json
```

The next implementation should be based on the audited historical `WaveSleepNet-main/models/protop.py`, `train_mtcl.py`, configuration files, and MorphMAE-to-MRCNN bridge rather than tuning the current spherical prototype model after observing these five validation folds.

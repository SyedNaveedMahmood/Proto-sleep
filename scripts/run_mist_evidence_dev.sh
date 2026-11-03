#!/usr/bin/env bash
# One fixed development fold, sequential controls, unique logs, no test scoring.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
DATA="${DATA:-/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset}"
OUT="${OUT:-mist_evidence_runs/fold_00_v1}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-123}"
VARIANTS="${VARIANTS:-evidence,summary_only,raw_context}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"
mkdir -p "$OUT/console"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT/console/run_${STAMP}.txt"
# Optional CLI flags, e.g. --epoch-only. Never infer temporal continuity here.
{
  echo "MIST-Evidence development run. Not an independent confirmatory experiment."
  echo "Git revision: $(git rev-parse HEAD)"
  echo "Started: $(date -Is)"
  python -m pytest tests/evidence -q
  python -u scripts/run_mist_evidence.py preflight \
    --data-dir "$DATA" --fold "$FOLD" --output-dir "$OUT/preflight" --device "$DEVICE" "$@"
  time python -u scripts/run_mist_evidence.py train \
    --data-dir "$DATA" --fold "$FOLD" --output-dir "$OUT/results" \
    --variants "$VARIANTS" --seeds "$SEEDS" --max-epochs "$EPOCHS" \
    --patience "$PATIENCE" --device "$DEVICE" --resume "$@"
  echo "Finished: $(date -Is)"
  cat "$OUT/results/results.csv"
} 2>&1 | tee "$LOG"
printf '\nSaved complete console output to %s\n' "$LOG"

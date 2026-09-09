#!/usr/bin/env bash
# Fixed MIST-Morph v2 development runner. No reserved-test scoring.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

DATA="${DATA:-/home/FA006/Desktop/Dimension/dataset/Preprocessed Sleep-EDF-20 dataset}"
OUT="${OUT:-mist_evidence_runs/mist_morph_transport_fold0}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-123}"
VARIANTS="${VARIANTS:-transport,evidence,summary_only,raw_context}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"

mkdir -p "$OUT/console"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT/console/run_${STAMP}.txt"

{
  echo "MIST-Morph transport v2 development run. Not independent confirmation."
  echo "Git revision: $(git rev-parse HEAD)"
  echo "Started: $(date -Is)"
  echo "Variants: $VARIANTS"
  echo "Seeds: $SEEDS"
  echo "Fold: $FOLD"

  python -m pytest tests/evidence -q

  python -u scripts/run_mist_evidence.py preflight \
    --data-dir "$DATA" \
    --fold "$FOLD" \
    --output-dir "$OUT/preflight" \
    --device "$DEVICE" \
    "$@"

  time python -u scripts/run_mist_evidence.py train \
    --data-dir "$DATA" \
    --fold "$FOLD" \
    --output-dir "$OUT/results" \
    --variants "$VARIANTS" \
    --seeds "$SEEDS" \
    --max-epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --device "$DEVICE" \
    --resume \
    "$@"

  echo "Finished: $(date -Is)"
  echo
  echo "RESULTS"
  cat "$OUT/results/results.csv"

  echo
  echo "TRANSPORT EXPLANATIONS"
  find "$OUT/results" -type f \
    \( -path '*/transport/explanations/*/index.html' \
       -o -path '*/transport/explanations/*/explanation.json' \
       -o -path '*/transport/explanations/*/all_contributions.csv' \) \
    | sort | head -n 30

} 2>&1 | tee "$LOG"

printf '\nSaved complete console output to %s\n' "$LOG"

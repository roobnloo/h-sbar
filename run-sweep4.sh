#!/usr/bin/env bash
# run-sweep4.sh — Run Scenario 4 across a grid of variance ratios.
# Usage: bash run-sweep4.sh [nrep]
set -euo pipefail

NREP=${1:-100}
OUTDIR=sim/sweep4
VAR_RATIOS="1 1.5 2 3 4 5 7 10"

mkdir -p "$OUTDIR"

for R in $VAR_RATIOS; do
  LOG="$OUTDIR/sweep4-varratio${R}.log"
  echo "=== var_ratio=$R  nrep=$NREP ===" | tee -a "$LOG"
  Rscript run-sim.R \
    --scenario=4 \
    --varratio="$R" \
    --nrep="$NREP" \
    --outdir="$OUTDIR" \
    2>&1 | tee -a "$LOG"
done

echo "Sweep complete. Results in $OUTDIR/"

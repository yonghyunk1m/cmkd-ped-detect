#!/bin/bash
# run_baseline_loso.sh — Audio-only baseline LOSO (no KD), ASPED_v1
# Paired with run_kd_loso.sh for ablation comparison.
#
# Usage:
#   ./run_baseline_loso.sh               # binary (default)
#   ./run_baseline_loso.sh --no-wandb

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="configs/baseline_binary_v1.yaml"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-wandb) EXTRA_ARGS="$EXTRA_ARGS --no-wandb"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SESSIONS=(
    "Session_5242023"
    "Session_6012023"
    "Session_6072023"
    "Session_6212023"
    "Session_6282023"
)

echo "============================================================"
echo "Baseline LOSO Training (no KD, ASPED_v1)"
echo "Config : $CONFIG"
echo "Sessions: ${SESSIONS[*]}"
echo "============================================================"

for SESSION in "${SESSIONS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo ">>> Starting fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "------------------------------------------------------------"
    $PYTHON train.py \
        --config "$CONFIG" \
        --test_session "$SESSION" \
        --wandb-project AcousticPedestrianCounting_KD \
        $EXTRA_ARGS
    echo ">>> Finished fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================================"
echo "All baseline LOSO folds complete."
echo "============================================================"

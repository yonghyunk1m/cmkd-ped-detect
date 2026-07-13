#!/bin/bash
# Hybrid (LogitKD + Cosine) + TFD-Label LOSO — 4 remaining folds
# Session_5242023 already complete (F1=0.3128, best so far)
# This script fills in folds 601/607/621/628 for statistical significance.

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/ykim/miniconda3/envs/Ped_KD_v3/bin/python3
CONFIG="configs/kd_hybrid_logit_cosine_immune.yaml"

SESSIONS=(
    "Session_6012023"
    "Session_6072023"
    "Session_6212023"
    "Session_6282023"
)

echo "============================================================"
echo "Hybrid + TFD-Label LOSO (4 folds)"
echo "Config  : $CONFIG"
echo "Sessions: ${SESSIONS[*]}"
echo "Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for SESSION in "${SESSIONS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo ">>> Starting fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "------------------------------------------------------------"
    $PYTHON train_kd.py \
        --config "$CONFIG" \
        --test_session "$SESSION"
    echo ">>> Finished fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================================"
echo "All 4 folds complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

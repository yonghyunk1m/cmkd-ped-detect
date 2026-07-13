#!/bin/bash
# run_kd_loso.sh — Run KD training for all 5 LOSO sessions.
#
# Usage:
#   ./run_kd_loso.sh                        # binary, attention (default)
#   ./run_kd_loso.sh --config kd_binary.yaml --agg gap
#   ./run_kd_loso.sh --config kd_4class.yaml
#   ./run_kd_loso.sh --no-wandb             # skip WandB logging
#
# Options:
#   --config <file>   Config file under configs/  (default: kd_binary.yaml)
#   --agg <type>      Override query_agg_type: gap | attention (default: from config)
#   --no-wandb        Pass --no-wandb to train_kd.py

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/ykim/miniconda3/envs/Ped_KD_v3/bin/python3

CONFIG="configs/kd_binary.yaml"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="configs/$2"; shift 2 ;;
        --agg)    AGG="$2"; shift 2 ;;
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
echo "KD LOSO Training"
echo "Config : $CONFIG"
echo "Sessions: ${SESSIONS[*]}"
echo "============================================================"

for SESSION in "${SESSIONS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo ">>> Starting fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "------------------------------------------------------------"
    $PYTHON train_kd.py \
        --config "$CONFIG" \
        --test_session "$SESSION" \
        $EXTRA_ARGS
    echo ">>> Finished fold: $SESSION  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "============================================================"
echo "All LOSO folds complete."
echo "============================================================"

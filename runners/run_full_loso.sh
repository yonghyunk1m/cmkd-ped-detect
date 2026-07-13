#!/bin/bash
# ============================================================
# Full LOSO: 3 configs × 5 sessions = 15 runs
# For DCASE 2026 paper final results
# ============================================================
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/media/ykim/Linux/.tmp_training
export TMP=$TMPDIR TEMP=$TMPDIR
export WANDB_DIR=/media/ykim/Linux/.wandb_data
export WANDB_DATA_DIR=/media/ykim/Linux/.wandb_data
export WANDB_CACHE_DIR=/media/ykim/Linux/.wandb_data/cache
export WANDB_MODE=offline
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

PYTHON=/media/ykim/Linux/Ped_KD_v3/bin/python3
SESSIONS=(Session_5242023 Session_6012023 Session_6072023 Session_6212023 Session_6282023)

# Three methods for the paper
LOSO_CONFIGS=(
    baseline_binary_v1
    kd_logit_unw
    kd_logit_unw_classweight
)

echo "============================================================"
echo "Full LOSO for DCASE Paper  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "Configs  : ${LOSO_CONFIGS[*]}"
echo "Sessions : ${SESSIONS[*]}"
echo "============================================================"

for cfg in "${LOSO_CONFIGS[@]}"; do
    echo ""
    echo "========== CONFIG: $cfg =========="
    for session in "${SESSIONS[@]}"; do
        out_dir="work_dir/${cfg}/${session}"
        best=$(find "$out_dir" -maxdepth 2 -name "best-*.ckpt" 2>/dev/null | head -1)
        if [[ -n "$best" ]]; then
            echo "  [SKIP] $session — $(basename $best)"
            continue
        fi
        echo "  >>> $session  ($(date '+%H:%M:%S'))"

        EXTRA_ARGS=""
        if [[ "$cfg" == baseline_* ]]; then
            $PYTHON train.py \
                --config "configs/${cfg}.yaml" \
                --test_session "$session" \
                --no-wandb
        else
            # For KD configs, teacher auto-detected from work_dir/teacher_video_only/
            $PYTHON train_kd.py \
                --config "configs/${cfg}.yaml" \
                --test_session "$session" \
                --no-wandb
        fi
        echo "  <<< $session done  ($(date '+%H:%M:%S'))"
    done
done

echo ""
echo "============================================================"
echo "Full LOSO Done.  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "============================================================"

# Collect results
echo ""
echo "========== RESULTS SUMMARY =========="
for cfg in "${LOSO_CONFIGS[@]}"; do
    echo "--- $cfg ---"
    for session in "${SESSIONS[@]}"; do
        logdir="work_dir/${cfg}/${session}"
        # Try to find test results in CSV logger
        csv=$(find "$logdir" -name "test_results.csv" 2>/dev/null | head -1)
        if [[ -n "$csv" ]]; then
            echo "  $session: $(cat $csv | tail -1)"
        fi
    done
done

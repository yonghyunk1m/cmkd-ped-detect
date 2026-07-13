#!/bin/bash
# Crash-resilient stride=1 inference with auto-resume
# Run inside tmux: tmux new -s inference './run_inference_stride1.sh'

export TMPDIR=/media/ykim/Linux/.tmp_ykim

PYTHON=/home/ykim/miniconda3/envs/Pedestrian_DCASE/bin/python
SCRIPT=inference.py
CONFIG=configs/baseline_4class_v3.yaml
DATA=/media/backup_SSD/ASPED_v3_npy
OUTPUT=results/predictions_4class_v3_stride1
CKPT_DIR=work_dir/baseline_4class_v3

cd /media/backup_SSD/AcousticPedestrianCounting

MAX_RETRIES=20
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date)] Attempt $((RETRY+1))/$MAX_RETRIES"

    $PYTHON $SCRIPT \
        --checkpoint_dir $CKPT_DIR \
        --config $CONFIG \
        --data_dir $DATA \
        --output_dir $OUTPUT \
        --stride 1 \
        --skip-existing

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Inference completed successfully!"
        break
    fi

    RETRY=$((RETRY+1))
    echo "[$(date)] Process exited with code $EXIT_CODE. Waiting 10s before retry..."
    sleep 10
done

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "[$(date)] Failed after $MAX_RETRIES attempts."
fi

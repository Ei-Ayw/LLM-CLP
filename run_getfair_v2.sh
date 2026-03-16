#!/bin/bash
set -e

EPOCHS=6
PATIENCE=3
BATCH_SIZE=16
GRAD_ACCUM=2
LR=2e-5
MAX_LEN=128
SEEDS=(42 123 2024)
DATASETS=(hatexplain toxigen dynahate)

echo "===== GetFair rerun (9 runs) ====="
echo "Start: $(date)"
COUNT=0
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo ">>> [$COUNT/9] GetFair | $DATASET | seed=$SEED"
        echo ">>> Start: $(date)"
        python3 src_script/train/train_baseline_getfair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --lambda_gf 0.5 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN
        echo ">>> Done: $(date)"
    done
done
echo ""
echo "===== GetFair done: $(date) ====="

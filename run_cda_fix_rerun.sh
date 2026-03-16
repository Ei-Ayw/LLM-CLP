#!/bin/bash
# =============================================================================
# CDA fix rerun: CF samples now contribute to CE loss
# Re-run all experiments that use cf_method != "none"
# CDA-Swap, LLM-only, Swap+CLP, Ours(LLM+CLP) × 3 datasets × 3 seeds = 36 runs
# =============================================================================

set -e

EPOCHS=6
PATIENCE=3
BATCH_SIZE=16
GRAD_ACCUM=2
LR=2e-5
MAX_LEN=128

SEEDS=(42 123 2024)
DATASETS=(hatexplain toxigen dynahate)

LOG_DIR="logs/full_runs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "CDA fix rerun (36 runs): $(date)"
echo "============================================="

COUNT=0
TOTAL=36

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        # --- 1. CDA-Swap (swap CF, no CLP) ---
        COUNT=$((COUNT + 1))
        echo ""
        echo ">>> [$COUNT/$TOTAL] CDA-Swap | $DATASET | seed=$SEED"
        echo ">>> Start: $(date)"
        python3 src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method swap --lambda_clp 0.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/cda_swap_${DATASET}_s${SEED}.log"
        echo ">>> Done: $(date)"

        # --- 2. LLM-only (LLM CF, no CLP) ---
        COUNT=$((COUNT + 1))
        echo ""
        echo ">>> [$COUNT/$TOTAL] LLM-only | $DATASET | seed=$SEED"
        echo ">>> Start: $(date)"
        python3 src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method llm --lambda_clp 0.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ablation_llm_only_${DATASET}_s${SEED}.log"
        echo ">>> Done: $(date)"

        # --- 3. Swap+CLP ---
        COUNT=$((COUNT + 1))
        echo ""
        echo ">>> [$COUNT/$TOTAL] Swap+CLP | $DATASET | seed=$SEED"
        echo ">>> Start: $(date)"
        python3 src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method swap --lambda_clp 1.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ablation_swap_clp_${DATASET}_s${SEED}.log"
        echo ">>> Done: $(date)"

        # --- 4. Ours (LLM+CLP) ---
        COUNT=$((COUNT + 1))
        echo ""
        echo ">>> [$COUNT/$TOTAL] Ours(LLM+CLP) | $DATASET | seed=$SEED"
        echo ">>> Start: $(date)"
        python3 src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method llm --lambda_clp 1.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ours_llm_clp_${DATASET}_s${SEED}.log"
        echo ">>> Done: $(date)"

    done
done

echo ""
echo "============================================="
echo "CDA fix rerun done: $(date)"
echo "Total: $COUNT runs"
echo "============================================="

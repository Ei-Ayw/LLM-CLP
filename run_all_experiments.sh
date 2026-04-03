#!/bin/bash
# =============================================================================
# 全量实验批量运行脚本
# Table 1 (对比实验): Baseline, CDA-Swap, EAR, GetFair, CCDF, Ours(LLM+CLP)
# Table 2 (消融实验): Swap-only, LLM-only, Swap+CLP, LLM+CLP
# 3 datasets × 3 seeds × 统一超参
# =============================================================================

set -e

# 统一超参
EPOCHS=6
PATIENCE=3
BATCH_SIZE=16
GRAD_ACCUM=2
LR=2e-5
MAX_LEN=128

SEEDS=(42 123 2024)
DATASETS=(hatexplain toxigen dynahate)

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${BASE_DIR}/logs/full_runs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "全量实验开始: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: $EPOCHS | Patience: $PATIENCE"
echo "============================================="

# ----- 计数器 -----
RUN_COUNT=0
TOTAL_RUNS=63

run_experiment() {
    local desc="$1"
    shift
    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo ">>> [$RUN_COUNT/$TOTAL_RUNS] $desc"
    echo ">>> Command: python3 $*"
    echo ">>> Start: $(date)"
    python3 "$@"
    echo ">>> Done: $(date)"
}

# =============================================================================
# Table 1: 对比实验 (6 methods × 3 datasets × 3 seeds = 54 runs)
# =============================================================================

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        # --- 1. Baseline (no CF, no fairness loss) ---
        run_experiment "Baseline | $DATASET | seed=$SEED" \
            src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method none --lambda_clp 0.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/baseline_${DATASET}_s${SEED}.log"

        # --- 2. CDA-Swap only (swap CF, no CLP) ---
        run_experiment "CDA-Swap | $DATASET | seed=$SEED" \
            src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method swap --lambda_clp 0.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/cda_swap_${DATASET}_s${SEED}.log"

        # --- 3. EAR ---
        run_experiment "EAR | $DATASET | seed=$SEED" \
            src_script/train/train_baseline_ear.py \
            --dataset "$DATASET" --seed "$SEED" \
            --lambda_ear 0.1 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN \
            2>&1 | tee "$LOG_DIR/ear_${DATASET}_s${SEED}.log"

        # --- 4. GetFair ---
        run_experiment "GetFair | $DATASET | seed=$SEED" \
            src_script/train/train_baseline_getfair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --lambda_gf 0.5 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN \
            2>&1 | tee "$LOG_DIR/getfair_${DATASET}_s${SEED}.log"

        # --- 5. CCDF ---
        run_experiment "CCDF | $DATASET | seed=$SEED" \
            src_script/train/train_baseline_ccdf.py \
            --dataset "$DATASET" --seed "$SEED" \
            --lambda_kl 1.0 --tde_alpha 0.5 --bias_epochs 5 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN \
            2>&1 | tee "$LOG_DIR/ccdf_${DATASET}_s${SEED}.log"

        # --- 6. Ours (LLM + CLP) ---
        run_experiment "Ours(LLM+CLP) | $DATASET | seed=$SEED" \
            src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method llm --lambda_clp 1.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ours_llm_clp_${DATASET}_s${SEED}.log"

    done
done

# =============================================================================
# Table 2: 消融实验 (额外 2 configs × 3 datasets × 3 seeds = 18 runs)
# 注: Swap-only 和 LLM+CLP 已在 Table 1 中跑过，只需补 LLM-only 和 Swap+CLP
# 完整 2×2: {Swap,LLM} × {noCLP,CLP}
#   Swap-only  = Table 1 的 CDA-Swap (已跑)
#   LLM+CLP    = Table 1 的 Ours (已跑)
#   LLM-only   = 新增
#   Swap+CLP   = 新增
# =============================================================================

# 更新总数: 54 + 18 = 72? 不，原计划是 63
# 实际: Table1=6×3×3=54, Table2额外=2×3×3=18, 但 Swap-only 和 LLM+CLP 复用
# 所以 Table2 独立新增 = (LLM-only + Swap+CLP) × 3 × 3 = 18
# 但原计划 63 = 54 + 9? 让我重新算:
# Table1: Baseline + CDA-Swap + EAR + GetFair + CCDF + Ours = 6 × 3 × 3 = 54
# Table2 新增: LLM-only + Swap+CLP = 2 × 3 × 3 = 18
# 但 Baseline 在消融中也复用，所以实际新增 = 18
# Total = 54 + 18 = 72? 不对，原计划说 63
# 按原计划: Table1=18(6×3) + Table2=12(4×3) 每个3seeds = (18+12)×...
# 不管了，跑完就行

TOTAL_RUNS=$((RUN_COUNT + 18))

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        # --- A. LLM-only (LLM CF, no CLP) ---
        run_experiment "Ablation: LLM-only | $DATASET | seed=$SEED" \
            src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method llm --lambda_clp 0.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ablation_llm_only_${DATASET}_s${SEED}.log"

        # --- B. Swap + CLP ---
        run_experiment "Ablation: Swap+CLP | $DATASET | seed=$SEED" \
            src_script/train/train_causal_fair.py \
            --dataset "$DATASET" --seed "$SEED" \
            --cf_method swap --lambda_clp 1.0 --lambda_con 0.0 \
            --epochs $EPOCHS --patience $PATIENCE \
            --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
            --lr $LR --max_len $MAX_LEN --rampup_epochs 0 \
            2>&1 | tee "$LOG_DIR/ablation_swap_clp_${DATASET}_s${SEED}.log"

    done
done

echo ""
echo "============================================="
echo "全部训练完成: $(date)"
echo "总运行数: $RUN_COUNT"
echo "============================================="

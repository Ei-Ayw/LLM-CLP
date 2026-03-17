#!/bin/bash
# =============================================================================
# 一键完成所有剩余实验 (含Vanilla + Davani + Ramponi + 全部重新评估)
# =============================================================================
set -e

cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification
LOG_FILE="logs/complete_all_$(date +%m%d_%H%M).log"
mkdir -p logs

echo "========================================" | tee -a $LOG_FILE
echo "开始全部实验: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)
CF_METHODS=("swap" "llm")

# =============================================================================
# Phase 1: Vanilla 训练+评估 (真正的纯baseline)
# =============================================================================
echo -e "\n===== Phase 1: Vanilla 训练+评估 =====" | tee -a $LOG_FILE

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Vanilla $DS seed$SEED ($(date))" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_vanilla.py \
            --dataset $DS --seed $SEED \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE

        MODEL_FILE=$(ls -t src_result/models/Vanilla_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -n "$MODEL_FILE" ]; then
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: Vanilla $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method vanilla \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        fi
    done
done

echo "Phase 1 完成: $(date)" | tee -a $LOG_FILE

# =============================================================================
# Phase 2: Davani 训练+评估
# =============================================================================
echo -e "\n===== Phase 2: Davani 训练+评估 =====" | tee -a $LOG_FILE

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Davani $DS seed$SEED ($(date))" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_davani.py \
            --dataset $DS --seed $SEED \
            --lambda_lp 1.0 --cf_method swap \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE

        MODEL_FILE=$(ls -t src_result/models/Davani_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -n "$MODEL_FILE" ]; then
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: Davani $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method davani \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        fi
    done
done

echo "Phase 2 完成: $(date)" | tee -a $LOG_FILE

# =============================================================================
# Phase 3: Ramponi 训练+评估
# =============================================================================
echo -e "\n===== Phase 3: Ramponi 训练+评估 =====" | tee -a $LOG_FILE

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Ramponi $DS seed$SEED ($(date))" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_ramponi.py \
            --dataset $DS --seed $SEED \
            --lambda_adv 0.1 \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE

        MODEL_FILE=$(ls -t src_result/models/Ramponi_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -n "$MODEL_FILE" ]; then
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: Ramponi $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method ramponi \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        fi
    done
done

echo "Phase 3 完成: $(date)" | tee -a $LOG_FILE

# =============================================================================
# Phase 4: 重新评估所有已有模型 (用新eval脚本，含FPED/FNED)
# =============================================================================
echo -e "\n===== Phase 4: 重新评估所有已有模型 =====" | tee -a $LOG_FILE

# 4a. CCDF, EAR, GetFair
declare -A BASELINE_METHODS
BASELINE_METHODS[CCDF]=ccdf
BASELINE_METHODS[EAR]=ear
BASELINE_METHODS[GetFair]=getfair

for METHOD in CCDF EAR GetFair; do
    METHOD_TYPE=${BASELINE_METHODS[$METHOD]}
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            MODEL_FILE=$(ls -t src_result/models/${METHOD}_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
            if [ -z "$MODEL_FILE" ]; then
                echo "[SKIP] ${METHOD}_${DS}_seed${SEED}" | tee -a $LOG_FILE
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: $METHOD $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method $METHOD_TYPE \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        done
    done
done

# 4b. 消融实验 (ours架构)
ABLATION_PATTERNS=(
    "none_clp0.0_con0.0"
    "swap_clp0.0_con0.0"
    "swap_clp1.0_con0.0"
    "llm_clp0.0_con0.0"
    "llm_clp1.0_con0.0"
    "llm_clp0.0_con0.5"
    "llm_clp1.0_con0.5"
)

for PATTERN in "${ABLATION_PATTERNS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            MODEL_FILE=$(ls -t src_result/models/${DS}_${PATTERN}_seed${SEED}_*.pth 2>/dev/null | head -1)
            if [ -z "$MODEL_FILE" ]; then
                echo "[SKIP] ${DS}_${PATTERN}_seed${SEED}" | tee -a $LOG_FILE
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: $PATTERN $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method ours \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        done
    done
done

echo -e "\n========================================" | tee -a $LOG_FILE
echo "全部实验完成: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

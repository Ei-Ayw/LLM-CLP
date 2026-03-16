#!/bin/bash

# 训练和评估新的baseline方法
# Davani et al., 2021 和 Ramponi & Tonelli, 2022

LOG_FILE="logs/new_baselines_train_eval.log"
mkdir -p logs

echo "开始训练新baseline方法: $(date)" | tee -a $LOG_FILE

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)

# 1. 训练 Davani et al., 2021
echo "===== 训练 Davani et al., 2021 =====" | tee -a $LOG_FILE
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Training: Davani $DATASET seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_davani.py \
            --dataset $DATASET \
            --seed $SEED \
            --lambda_lp 1.0 \
            --epochs 5 \
            --batch_size 48 \
            --lr 2e-5 2>&1 | tee -a $LOG_FILE
    done
done

# 2. 训练 Ramponi & Tonelli, 2022
echo "===== 训练 Ramponi & Tonelli, 2022 =====" | tee -a $LOG_FILE
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Training: Ramponi $DATASET seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_ramponi.py \
            --dataset $DATASET \
            --seed $SEED \
            --lambda_adv 1.0 \
            --epochs 5 \
            --batch_size 48 \
            --lr 2e-5 2>&1 | tee -a $LOG_FILE
    done
done

# 3. 评估所有新训练的模型
echo "===== 评估新模型 =====" | tee -a $LOG_FILE
for DATASET in "${DATASETS[@]}"; do
    # Davani
    for SEED in "${SEEDS[@]}"; do
        MODEL_FILE=$(ls src_result/models/Davani_${DATASET}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -f "$MODEL_FILE" ]; then
            echo ">>> Eval: Davani $DATASET seed$SEED" | tee -a $LOG_FILE

            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method swap \
                --model_type davani 2>&1 | tee -a $LOG_FILE

            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method llm \
                --model_type davani 2>&1 | tee -a $LOG_FILE
        fi
    done

    # Ramponi
    for SEED in "${SEEDS[@]}"; do
        MODEL_FILE=$(ls src_result/models/Ramponi_${DATASET}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -f "$MODEL_FILE" ]; then
            echo ">>> Eval: Ramponi $DATASET seed$SEED" | tee -a $LOG_FILE

            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method swap \
                --model_type ramponi 2>&1 | tee -a $LOG_FILE

            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method llm \
                --model_type ramponi 2>&1 | tee -a $LOG_FILE
        fi
    done
done

echo "训练和评估完成: $(date)" | tee -a $LOG_FILE

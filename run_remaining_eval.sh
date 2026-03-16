#!/bin/bash

# 完成剩余的公平性评估
LOG_FILE="logs/remaining_eval.log"
mkdir -p logs

echo "开始剩余评估: $(date)" | tee -a $LOG_FILE

# DynaHate 数据集评估
DATASET="dynahate"
SEEDS=(42 123 2024)

echo "===== 评估 DynaHate 数据集 =====" | tee -a $LOG_FILE

# Baseline方法
for MODEL in CCDF EAR GetFair; do
    for SEED in "${SEEDS[@]}"; do
        MODEL_FILE=$(ls src_result/models/${MODEL}_${DATASET}_seed${SEED}_*.pth 2>/dev/null | head -1)
        if [ -f "$MODEL_FILE" ]; then
            echo ">>> Eval: $MODEL seed$SEED" | tee -a $LOG_FILE

            # swap CF
            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method swap \
                --model_type $(echo $MODEL | tr '[:upper:]' '[:lower:]') 2>&1 | tee -a $LOG_FILE

            # llm CF
            python3 src_script/eval/eval_causal_fairness.py \
                --checkpoint "$MODEL_FILE" \
                --dataset "$DATASET" \
                --cf_method llm \
                --model_type $(echo $MODEL | tr '[:upper:]' '[:lower:]') 2>&1 | tee -a $LOG_FILE
        fi
    done
done

# CF方法
for CF in none swap llm; do
    for CLP in 0.0 1.0; do
        for SEED in "${SEEDS[@]}"; do
            MODEL_FILE=$(ls src_result/models/${DATASET}_${CF}_clp${CLP}_con*_seed${SEED}_*.pth 2>/dev/null | head -1)
            if [ -f "$MODEL_FILE" ]; then
                echo ">>> Eval: CF-$CF λ=$CLP seed$SEED" | tee -a $LOG_FILE

                # swap CF
                python3 src_script/eval/eval_causal_fairness.py \
                    --checkpoint "$MODEL_FILE" \
                    --dataset "$DATASET" \
                    --cf_method swap \
                    --model_type ours 2>&1 | tee -a $LOG_FILE

                # llm CF
                python3 src_script/eval/eval_causal_fairness.py \
                    --checkpoint "$MODEL_FILE" \
                    --dataset "$DATASET" \
                    --cf_method llm \
                    --model_type ours 2>&1 | tee -a $LOG_FILE
            fi
        done
    done
done

# ToxiGen 缺失的评估
echo "===== 补充 ToxiGen 缺失评估 =====" | tee -a $LOG_FILE
MODEL_FILE=$(ls src_result/models/toxigen_swap_clp0.0_con*_seed2024_*.pth 2>/dev/null | head -1)
if [ -f "$MODEL_FILE" ]; then
    echo ">>> Eval: toxigen swap λ=0.0 seed2024" | tee -a $LOG_FILE

    python3 src_script/eval/eval_causal_fairness.py \
        --checkpoint "$MODEL_FILE" \
        --dataset toxigen \
        --cf_method swap \
        --model_type ours 2>&1 | tee -a $LOG_FILE

    python3 src_script/eval/eval_causal_fairness.py \
        --checkpoint "$MODEL_FILE" \
        --dataset toxigen \
        --cf_method llm \
        --model_type ours 2>&1 | tee -a $LOG_FILE
fi

echo "评估完成: $(date)" | tee -a $LOG_FILE

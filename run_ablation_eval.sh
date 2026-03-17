#!/bin/bash
# 消融实验评估: swap(λ=0), swap+CLP(λ=1), llm(λ=0), none(λ=0)
# LLM+CLP(λ=1) 即 Ours，已有eval结果

cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification
export PYTHONPATH="src_script/train:src_script/eval:src_script/utils:src_script/data:$PYTHONPATH"

LOG_FILE="logs/ablation_eval_0317.log"
echo "消融实验评估开始: $(date)" | tee $LOG_FILE

DATASETS=(hatexplain toxigen dynahate)
SEEDS=(42 123 2024)

# 消融变体: cf_type, clp_value
ABLATIONS=(
    "none 0.0"
    "swap 0.0"
    "swap 1.0"
    "llm 0.0"
)

for ds in "${DATASETS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
        read cf clp <<< "$ablation"
        for seed in "${SEEDS[@]}"; do
            # 找最新的 checkpoint
            CKPT=$(ls -t src_result/models/${ds}_${cf}_clp${clp}_con*_seed${seed}_*.pth 2>/dev/null | head -1)
            if [ -z "$CKPT" ]; then
                echo "[SKIP] 无checkpoint: ${ds}_${cf}_clp${clp}_seed${seed}" | tee -a $LOG_FILE
                continue
            fi

            # 检查是否已有eval结果
            CKPT_NAME=$(basename "$CKPT" .pth)
            EVAL_FILE="src_result/eval/${CKPT_NAME}_7metrics_llm.json"
            if [ -f "$EVAL_FILE" ]; then
                echo "[SKIP] 已有结果: $EVAL_FILE" | tee -a $LOG_FILE
                continue
            fi

            echo ">>> Eval: ${ds} ${cf} λ=${clp} seed${seed}" | tee -a $LOG_FILE
            python3 src_script/eval/eval_all_baselines.py \
                --method ours \
                --checkpoint "$CKPT" \
                --dataset "$ds" \
                --cf_method llm 2>&1 | tee -a $LOG_FILE

            if [ $? -ne 0 ]; then
                echo "[ERROR] 失败: ${ds}_${cf}_clp${clp}_seed${seed}" | tee -a $LOG_FILE
            fi
        done
    done
done

echo "消融实验评估完成: $(date)" | tee -a $LOG_FILE

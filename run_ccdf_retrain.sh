#!/bin/bash
# 重新训练 CCDF (保存 bias_model) + 评估
set -e
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)
CF_METHODS=("swap" "llm")

echo "===== CCDF 重新训练+评估 ====="
echo "开始: $(date)"

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: CCDF $DS seed$SEED ($(date))"
        python3 src_script/train/train_baseline_ccdf.py \
            --dataset $DS --seed $SEED \
            --lambda_kl 1.0 --tde_alpha 0.5 --bias_epochs 5 \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1

        MODEL_FILE=$(ls -t src_result/models/CCDF_${DS}_seed${SEED}_0317_*.pth 2>/dev/null | head -1)
        if [ -n "$MODEL_FILE" ]; then
            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: CCDF $DS seed$SEED cf=$CF"
                python3 src_script/eval/eval_all_baselines.py \
                    --method ccdf \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1
            done
        else
            echo "[WARN] No CCDF checkpoint found for ${DS} seed${SEED}"
        fi
    done
done

echo "===== CCDF 完成: $(date) ====="

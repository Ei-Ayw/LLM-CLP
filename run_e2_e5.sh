#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification
MODEL_PATH="models/deberta-v3-base"

echo "========== E2: Swap+CLP =========="
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method swap \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

echo "========== E3: LLM+CLP =========="
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

echo "========== E4: LLM+SupCon =========="
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 0.0 --lambda_con 0.5 \
    --seed 42

echo "========== E5: LLM+CLP+SupCon =========="
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.5 \
    --seed 42

echo "========== 因果公平评估 =========="
for CKPT in src_result/models/hatexplain_*.pth; do
    echo "评估: $CKPT"
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" \
        --dataset hatexplain \
        --cf_method llm
done

echo "========== 全部完成 =========="

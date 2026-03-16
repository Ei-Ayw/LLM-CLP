#!/bin/bash
# =============================================================
# 全量实验脚本: E5调参 + 多Seed + ToxiGen
# 目标: CCFA 论文完整实验
# =============================================================
set -e
export TOKENIZERS_PARALLELISM=false
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification
MODEL_PATH="models/deberta-v3-base"
API_KEYS="ad443e864e1046838e00b2072c798320.9k8PEtdmWiQV8Yw2,5f2e493b563348e28d9748fe0020f760.Y2yxIZICZPk1XQBn"

echo "========================================================"
echo " Phase 0: 生成 ToxiGen 反事实数据 (Swap + LLM)"
echo " $(date)"
echo "========================================================"

# 0a. ToxiGen Swap 反事实 (train + test, 本地生成, 秒级)
python3 -c "
import sys
sys.path.append('src_script/counterfactual')
from cf_generator_swap import batch_generate_swap
import pandas as pd

for split in ['train', 'test']:
    df = pd.read_parquet(f'data/causal_fair/toxigen_{split}.parquet')
    cf = batch_generate_swap(df['text'].tolist())
    if len(cf) > 0:
        cf.to_parquet(f'data/causal_fair/toxigen_{split}_cf_swap.parquet', index=False)
        print(f'ToxiGen {split} swap: {len(cf)} 条')
    else:
        print(f'ToxiGen {split} swap: 0 条 (无身份词匹配)')
"

# 0b. ToxiGen LLM 反事实 (train, API调用)
echo "--- ToxiGen train LLM 反事实 ---"
python src_script/counterfactual/cf_generator_llm.py \
    --dataset toxigen --split train \
    --api zhipu --api_key "$API_KEYS" \
    --max_workers 30

# 0c. ToxiGen LLM 反事实 (test, API调用)
echo "--- ToxiGen test LLM 反事实 ---"
python src_script/counterfactual/cf_generator_llm.py \
    --dataset toxigen --split test \
    --api zhipu --api_key "$API_KEYS" \
    --max_workers 30

echo ">>> Phase 0 完成: $(date)"

echo "========================================================"
echo " Phase 1: E5 超参调优 (HateXplain, seed=42)"
echo " $(date)"
echo "========================================================"

# E5a: clp=1.0, con=0.1
echo "--- E5a: clp=1.0, con=0.1 ---"
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.1 --seed 42

# E5b: clp=1.0, con=0.2
echo "--- E5b: clp=1.0, con=0.2 ---"
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.2 --seed 42

# E5c: clp=1.0, con=0.3
echo "--- E5c: clp=1.0, con=0.3 ---"
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.3 --seed 42

# E5d: clp=0.5, con=0.2
echo "--- E5d: clp=0.5, con=0.2 ---"
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 0.5 --lambda_con 0.2 --seed 42

# E5e: clp=0.5, con=0.1
echo "--- E5e: clp=0.5, con=0.1 ---"
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 0.5 --lambda_con 0.1 --seed 42

echo ">>> Phase 1 完成: $(date)"

echo "========================================================"
echo " Phase 2: HateXplain 多 Seed (seed=123, 2024)"
echo " $(date)"
echo "========================================================"

for SEED in 123 2024; do
    echo "===== Seed=$SEED ====="

    # E1: Baseline
    echo "--- E1 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset hatexplain --cf_method none \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 0.0 --lambda_con 0.0 --seed $SEED

    # E2: Swap+CLP
    echo "--- E2 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset hatexplain --cf_method swap \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED

    # E3: LLM+CLP
    echo "--- E3 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset hatexplain --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED

    # E4: LLM+SupCon
    echo "--- E4 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset hatexplain --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 0.0 --lambda_con 0.5 --seed $SEED

    # E5: 用原始 clp=1.0 con=0.5 保持一致
    echo "--- E5 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset hatexplain --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.5 --seed $SEED
done

echo ">>> Phase 2 完成: $(date)"

echo "========================================================"
echo " Phase 3: ToxiGen 全实验 (3 seeds)"
echo " $(date)"
echo "========================================================"

for SEED in 42 123 2024; do
    echo "===== ToxiGen Seed=$SEED ====="

    # E1: Baseline
    echo "--- ToxiGen E1 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset toxigen --cf_method none \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 0.0 --lambda_con 0.0 --seed $SEED

    # E2: Swap+CLP
    echo "--- ToxiGen E2 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset toxigen --cf_method swap \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED

    # E3: LLM+CLP
    echo "--- ToxiGen E3 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset toxigen --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED

    # E4: LLM+SupCon
    echo "--- ToxiGen E4 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset toxigen --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 0.0 --lambda_con 0.5 --seed $SEED

    # E5: LLM+CLP+SupCon
    echo "--- ToxiGen E5 seed=$SEED ---"
    python src_script/train/train_causal_fair.py \
        --dataset toxigen --cf_method llm \
        --model_name "$MODEL_PATH" \
        --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
        --lambda_clp 1.0 --lambda_con 0.5 --seed $SEED
done

echo ">>> Phase 3 完成: $(date)"

echo "========================================================"
echo " Phase 4: 全量因果公平评估"
echo " $(date)"
echo "========================================================"

# HateXplain 评估 (用 LLM 反事实)
for CKPT in src_result/models/hatexplain_*.pth; do
    echo "评估(HX): $CKPT"
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" --dataset hatexplain --cf_method llm || true
done

# ToxiGen 评估 (用 LLM 反事实)
for CKPT in src_result/models/toxigen_*.pth; do
    echo "评估(TG): $CKPT"
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" --dataset toxigen --cf_method llm || true
done

echo "========================================================"
echo " 全部实验完成! $(date)"
echo "========================================================"

#!/bin/bash
set -e

# =============================================================================
# 全流程运行脚本: LLM 反事实增强 + 因果公平
# 用法: bash run_causal_fair.sh
# =============================================================================

cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="data/causal_fair"
ZHIPU_API_KEY="${ZHIPU_API_KEY:-5f2e493b563348e28d9748fe0020f760.Y2yxIZICZPk1XQBn}"

echo "============================================="
echo " Step 1: 下载数据集"
echo "============================================="
if [ ! -f "$DATA_DIR/hatexplain_train.parquet" ]; then
    python src_script/data/download_datasets.py
else
    echo "  数据集已存在，跳过下载"
fi

echo ""
echo "============================================="
echo " Step 2: 生成传统换词反事实 (CDA-Swap, 对照组)"
echo "============================================="
if [ ! -f "$DATA_DIR/hatexplain_train_cf_swap.parquet" ]; then
    python -c "
import sys, os
sys.path.append('src_script/counterfactual')
from cf_generator_swap import batch_generate_swap
import pandas as pd

for split in ['train', 'test']:
    df = pd.read_parquet('$DATA_DIR/hatexplain_{}.parquet'.format(split))
    cf = batch_generate_swap(df['text'].tolist(),
                              post_ids=df.get('post_id', None),
                              max_cf_per_sample=2)
    save_path = '$DATA_DIR/hatexplain_{}_cf_swap.parquet'.format(split)
    cf.to_parquet(save_path, index=False)
    print(f'  [{split}] 生成 {len(cf)} 条换词反事实 → {save_path}')
"
else
    echo "  换词反事实已存在，跳过"
fi

echo ""
echo "============================================="
echo " Step 3: 生成 LLM 反事实 (核心创新)"
echo "============================================="
for SPLIT in train test; do
    if [ ! -f "$DATA_DIR/hatexplain_${SPLIT}_cf_llm.parquet" ]; then
        echo "  生成 $SPLIT 集 LLM 反事实..."
        python src_script/counterfactual/cf_generator_llm.py \
            --dataset hatexplain \
            --split $SPLIT \
            --api zhipu \
            --api_key "$ZHIPU_API_KEY" \
            --model glm-4-flash \
            --max_cf 2 \
            --data_dir "$DATA_DIR"
    else
        echo "  $SPLIT 集 LLM 反事实已存在，跳过"
    fi
done

echo ""
echo "============================================="
echo " Step 4: 验证并过滤反事实质量"
echo "============================================="
for SPLIT in train test; do
    python src_script/counterfactual/cf_validator.py \
        --input "$DATA_DIR/hatexplain_${SPLIT}_cf_llm.parquet" \
        --output "$DATA_DIR/hatexplain_${SPLIT}_cf_llm.parquet"
done

echo ""
echo "============================================="
echo " Step 5: 训练实验"
echo "============================================="

# E1: Baseline (无反事实)
echo "[E1] Baseline..."
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method none \
    --epochs 5 --batch_size 16 --lr 2e-5 \
    --lambda_clp 0.0 --lambda_con 0.0 \
    --seed 42

# E2: 传统换词反事实 + CLP
echo "[E2] CDA-Swap + CLP..."
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method swap \
    --epochs 5 --batch_size 16 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

# E3: LLM 反���实 + CLP
echo "[E3] CDA-LLM + CLP..."
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --epochs 5 --batch_size 16 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

# E4: LLM 反事实 + SupCon
echo "[E4] CDA-LLM + SupCon..."
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --epochs 5 --batch_size 16 --lr 2e-5 \
    --lambda_clp 0.0 --lambda_con 0.5 \
    --seed 42

# E5: LLM 反事实 + CLP + SupCon (完整方法)
echo "[E5] CDA-LLM + CLP + SupCon (FULL)..."
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --epochs 5 --batch_size 16 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.5 \
    --seed 42

echo ""
echo "============================================="
echo " Step 6: 因果公平评估 (对所有模型)"
echo "============================================="
for CKPT in src_result/models/hatexplain_*.pth; do
    echo "评估: $CKPT"
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" \
        --dataset hatexplain \
        --cf_method llm
done

echo ""
echo "============================================="
echo " 全部完成!"
echo "============================================="

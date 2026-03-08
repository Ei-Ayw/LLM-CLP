#!/bin/bash
set -e

export HF_HOME=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models
export HF_HUB_CACHE=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /root/lanyun-fs/01_nlp_toxicity_classification

DATA_DIR=data/identity

# S1 checkpoint (自然分布预训练的)
S1_CKPT=$(ls -t src_result/models/DebertaV3MTL_S1_Natural_Seed42_*.pth 2>/dev/null | head -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] 找不到 S1 checkpoint!"
    exit 1
fi
echo "[$(date)] S1 checkpoint: $S1_CKPT"

# =====================================================
# HCMA: Hierarchical Conditional Metric-Aligned Debiasing
# 改进 1: Soft gate
# 改进 2: Slice-aware sampler
# 改进 3: Metric-aligned AUC surrogate loss
# 改进 4: Hierarchical conditional adversary
# =====================================================
echo "[$(date)] === HCMA 训练 (40.5万身份标签数据) ==="
torchrun --nproc_per_node=4 \
    src_script/train/train_deberta_v3_hcma.py \
    --s1_checkpoint "$S1_CKPT" \
    --data_dir "$DATA_DIR" \
    --seed 42 --data_seed 42 \
    --batch_size 48 --grad_accum 2 \
    --warmup_epochs 5 --warmup_lr 1e-4 \
    --adv_epochs 6 --adv_lr 2e-6 \
    --lambda_max 0.3 --gamma 10.0 \
    --adv_lr_mult 3.0 \
    --aux_scale 0.15 \
    --w_metric 0.30 \
    --layer_decay 0.95 \
    --ema_decay 0.999 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1

echo "[$(date)] 训练完成"

HCMA_CKPT=$(ls -t src_result/models/HCMA_Seed42_*.pth 2>/dev/null | head -1)
if [ -z "$HCMA_CKPT" ]; then
    echo "[ERROR] 找不到 HCMA checkpoint!"
    exit 1
fi

echo "[$(date)] 评估: $HCMA_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$HCMA_CKPT" \
    --model_type hcma \
    --data_dir "$DATA_DIR" \
    --output_prefix "HCMA_$(basename $HCMA_CKPT .pth)"

echo "[$(date)] === HCMA 完成 ==="

# =====================================================
# Vanilla DeBERTa-V3 Baseline (同一数据集对比)
# =====================================================
echo "[$(date)] === Vanilla Baseline 训练 (40.5万身份标签数据) ==="
torchrun --nproc_per_node=4 \
    src_script/train/train_vanilla_deberta_v3.py \
    --data_dir "$DATA_DIR" \
    --seed 42 --data_seed 42 \
    --sample_size 500000 \
    --batch_size 48 --grad_accum 2 \
    --epochs 2 --lr 2e-5

echo "[$(date)] Vanilla 训练完成"

VANILLA_CKPT=$(ls -t src_result/models/VanillaDeBERTa_Seed42_*.pth 2>/dev/null | head -1)
if [ -z "$VANILLA_CKPT" ]; then
    echo "[ERROR] 找不到 Vanilla checkpoint!"
    exit 1
fi

echo "[$(date)] Vanilla 评估: $VANILLA_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$VANILLA_CKPT" \
    --model_type vanilla_deberta \
    --data_dir "$DATA_DIR" \
    --output_prefix "Vanilla_Identity_$(basename $VANILLA_CKPT .pth)"

echo "[$(date)] === 全部完成 ==="

#!/bin/bash
set -e

export HF_HOME=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models
export HF_HUB_CACHE=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /root/lanyun-fs/01_nlp_toxicity_classification

# 1. HCMA 重新评估（修复了tokenizer问题）
HCMA_CKPT=$(ls -t src_result/models/HCMA_Seed42_*.pth 2>/dev/null | head -1)
echo "[$(date)] HCMA 评估: $HCMA_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$HCMA_CKPT" \
    --model_type hcma \
    --data_dir data/identity \
    --output_prefix "HCMA_$(basename $HCMA_CKPT .pth)"
echo "[$(date)] HCMA 评估完成"

# 2. Vanilla Baseline 训练（去掉了 --grad_accum）
echo "[$(date)] === Vanilla Baseline 训练 (2 epochs) ==="
torchrun --nproc_per_node=4 \
    src_script/train/train_vanilla_deberta_v3.py \
    --data_dir data/identity \
    --seed 42 --data_seed 42 \
    --sample_size 500000 \
    --batch_size 48 \
    --epochs 2 --lr 2e-5
echo "[$(date)] Vanilla 训练完成"

# 3. Vanilla 评估
VANILLA_CKPT=$(ls -t src_result/models/VanillaDeBERTa_Seed42_*.pth 2>/dev/null | head -1)
echo "[$(date)] Vanilla 评估: $VANILLA_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$VANILLA_CKPT" \
    --model_type vanilla_deberta \
    --data_dir data/identity \
    --output_prefix "Vanilla_Identity_$(basename $VANILLA_CKPT .pth)"

echo "[$(date)] === 全部完成 ==="

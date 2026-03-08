#!/bin/bash
set -e

export HF_HOME=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models
export HF_HUB_CACHE=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /root/lanyun-fs/01_nlp_toxicity_classification

DATA_DIR=data/natural

# =====================================================
# 1. Vanilla DeBERTa (baseline) on natural distribution
# =====================================================
echo "[$(date)] === 开始 Vanilla DeBERTa 训练 (自然分布) ==="
torchrun --nproc_per_node=5 \
    src_script/train/train_vanilla_deberta_v3.py \
    --data_dir "$DATA_DIR" \
    --seed 42 --data_seed 42 \
    --sample_size 300000 \
    --epochs 4 --batch_size 48 --lr 2e-5

echo "[$(date)] Vanilla 训练完成"

VANILLA_CKPT=$(ls -t src_result/models/VanillaDeBERTa_Seed42_*.pth 2>/dev/null | head -1)
if [ -z "$VANILLA_CKPT" ]; then
    echo "[ERROR] 找不到 Vanilla checkpoint!"
    exit 1
fi

echo "[$(date)] 评估 Vanilla: $VANILLA_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$VANILLA_CKPT" \
    --model_type vanilla_deberta \
    --data_dir "$DATA_DIR" \
    --output_prefix "Natural_$(basename $VANILLA_CKPT .pth)"

# =====================================================
# 2. GRL Adversarial on natural distribution
# =====================================================
echo "[$(date)] === 开始 GRL 对抗去偏训练 (自然分布) ==="

# 使用已有 S1 checkpoint 作为 backbone
S1_CKPT=$(ls -t src_result/models/DebertaV3MTL_S1_Seed42_Sample300000_*.pth 2>/dev/null | head -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] 找不到 S1 checkpoint!"
    exit 1
fi
echo "[$(date)] 使用 S1 checkpoint: $S1_CKPT"

torchrun --nproc_per_node=5 \
    src_script/train/train_deberta_v3_adv.py \
    --s1_checkpoint "$S1_CKPT" \
    --data_dir "$DATA_DIR" \
    --seed 42 --data_seed 42 \
    --sample_size 300000 \
    --batch_size 48 --grad_accum 2 \
    --warmup_epochs 2 --warmup_lr 1e-4 \
    --adv_epochs 4 --adv_lr 2e-6 \
    --lambda_max 0.1 --gamma 10.0 \
    --aux_scale 0.3 \
    --layer_decay 0.95 \
    --ema_decay 0.999 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1

echo "[$(date)] GRL 训练完成"

ADV_CKPT=$(ls -t src_result/models/DebertaV3Adv_Seed42_*.pth 2>/dev/null | head -1)
if [ -z "$ADV_CKPT" ]; then
    echo "[ERROR] 找不到 Adv checkpoint!"
    exit 1
fi

echo "[$(date)] 评估 GRL: $ADV_CKPT"
python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$ADV_CKPT" \
    --model_type deberta_adv \
    --data_dir "$DATA_DIR" \
    --output_prefix "Natural_$(basename $ADV_CKPT .pth)"

echo "[$(date)] === 全部完成 ==="

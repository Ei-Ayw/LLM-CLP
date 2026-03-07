#!/bin/bash
set -e

export HF_HOME=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models
export HF_HUB_CACHE=/root/lanyun-fs/01_nlp_toxicity_classification/pretrained_models/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /root/lanyun-fs/01_nlp_toxicity_classification

# 等 S1 完成（检测进程）
echo "[$(date)] 等待 S1 训练完成..."
while pgrep -f "train_deberta_v3_mtl_s1" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] S1 训练已结束"

# 找到 S1 最新的 checkpoint
S1_CKPT=$(ls -t src_result/models/DebertaV3MTL_S1_Seed42_Sample300000_0307*.pth 2>/dev/null | head -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] 找不到 S1 checkpoint!"
    exit 1
fi
echo "[$(date)] 使用 S1 checkpoint: $S1_CKPT"

# 启动 S2
echo "[$(date)] 启动 S2 训练..."
torchrun --nproc_per_node=5 \
    src_script/train/train_deberta_v3_mtl_s2.py \
    --s1_checkpoint "$S1_CKPT" \
    --seed 42 --data_seed 42 \
    --sample_size 300000 \
    --epochs 4 --batch_size 48 --lr 3e-6 \
    --aux_scale 0.3 --select_by_final --layer_decay 0.8

echo "[$(date)] S2 训练完成"

# 找到 S2 checkpoint 并评估
S2_CKPT=$(ls -t src_result/models/DebertaV3MTL_S2_Seed42_Sample300000_0307*.pth 2>/dev/null | head -1)
if [ -z "$S2_CKPT" ]; then
    echo "[ERROR] 找不到 S2 checkpoint!"
    exit 1
fi
echo "[$(date)] 启动评估: $S2_CKPT"

python3 src_script/eval/eval_universal_runner.py \
    --checkpoint "$S2_CKPT" \
    --model_type deberta_mtl \
    --output_prefix "$(basename $S2_CKPT .pth)"

echo "[$(date)] 全部完成！"

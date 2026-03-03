#!/bin/bash
# 正确运行3个DataParallel基线 (不使用torch.distributed.run)
# Usage: nohup bash run_remaining_baselines.sh > remaining_baselines.log 2>&1 &

set -e
cd /root/lanyun-fs/01_nlp_toxicity_classification

export HF_HOME="$(pwd)/pretrained_models"
export HF_HUB_CACHE="$(pwd)/pretrained_models/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# 这些脚本内部用 nn.DataParallel，不需要 torch.distributed.run
COMMON_ARGS="--sample_size 300000 --batch_size 64 --max_len 256 --scheduler linear --patience 1 --early_patience 3 --seed 42 --epochs 10"

# ===== 1. Vanilla BERT =====
echo "[$(date)] >>> Training Vanilla BERT (seed=42, DataParallel on 7 GPUs)"
python3 src_script/train/train_vanilla_bert.py $COMMON_ARGS \
    2>&1 | tee src_result/logs/baseline_VanillaBERT_Seed42.log
echo "[$(date)] >>> Vanilla BERT done (rc=$?)"

# ===== 2. Vanilla RoBERTa =====
echo ""
echo "[$(date)] >>> Training Vanilla RoBERTa (seed=42, DataParallel on 7 GPUs)"
python3 src_script/train/train_vanilla_roberta.py $COMMON_ARGS \
    2>&1 | tee src_result/logs/baseline_VanillaRoBERTa_Seed42.log
echo "[$(date)] >>> Vanilla RoBERTa done (rc=$?)"

# ===== 3. BERT + CNN-BiLSTM =====
echo ""
echo "[$(date)] >>> Training BERT+CNN-BiLSTM (seed=42, DataParallel on 7 GPUs)"
python3 src_script/train/train_bert_cnn_bilstm.py $COMMON_ARGS \
    2>&1 | tee src_result/logs/baseline_BertCNNBiLSTM_Seed42.log
echo "[$(date)] >>> BERT+CNN-BiLSTM done (rc=$?)"

# ===== 4. 评估新训练的模型 =====
echo ""
echo "[$(date)] >>> Evaluating new baselines..."

for keyword in VanillaBERT VanillaRoBERTa BertCNNBiLSTM; do
    # 找最新的非0224的pth
    CKPT=$(ls -t src_result/models/${keyword}*.pth 2>/dev/null | grep -v "0224" | head -1)
    if [ -z "$CKPT" ]; then
        echo "[WARN] No new checkpoint found for $keyword"
        continue
    fi
    BASENAME=$(basename "$CKPT" .pth)
    EVAL_JSON="src_result/eval/${BASENAME}_metrics.json"
    if [ -f "$EVAL_JSON" ]; then
        echo "[Skip] $BASENAME already evaluated"
        continue
    fi

    if [[ "$keyword" == "VanillaBERT" ]]; then
        MTYPE="vanilla_bert"
    elif [[ "$keyword" == "VanillaRoBERTa" ]]; then
        MTYPE="vanilla_roberta"
    else
        MTYPE="bert_cnn"
    fi

    echo "[$(date)] Evaluating: $BASENAME (type=$MTYPE)"
    CUDA_VISIBLE_DEVICES=0 python3 src_script/eval/eval_universal_runner.py \
        --checkpoint "$CKPT" --model_type "$MTYPE" --output_prefix "$BASENAME" \
        2>&1 | tee src_result/logs/eval_${BASENAME}.log
done

echo ""
echo "[$(date)] >>> ALL DONE."

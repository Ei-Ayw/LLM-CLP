#!/bin/bash
# =============================================================================
# 全量评估脚本: 对所有训练好的 checkpoint 运行因果公平评估
# 使用 swap 和 llm 两种反事实分别评估
# =============================================================================

set -e

DATASETS=(hatexplain toxigen dynahate)
SEEDS=(42 123 2024)
CF_METHODS=(swap llm)

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${BASE_DIR}/models/saved"
LOG_DIR="${BASE_DIR}/logs/eval_runs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "全量评估开始: $(date)"
echo "============================================="

COUNT=0

# 并行评估控制
MAX_JOBS=3
job_count=0

eval_checkpoint() {
    local ckpt="$1"
    local dataset="$2"
    local model_type="$3"
    local cf_method="$4"
    local desc="$5"

    if [ ! -f "$ckpt" ]; then
        echo "  [SKIP] Checkpoint not found: $ckpt"
        return
    fi

    # 如果已经有公平性评估结果则跳过
    local ckpt_name=$(basename "$ckpt" .pth)
    local fairness_path="${BASE_DIR}/src_result/eval/${ckpt_name}_fairness.json"
    if [ -f "$fairness_path" ]; then
        echo "  [SKIP] 已有评估结果: $fairness_path"
        return
    fi

    COUNT=$((COUNT + 1))
    echo ""
    echo ">>> [$COUNT] Eval: $desc | cf=$cf_method"
    (
        python3 src_script/eval/eval_causal_fairness.py \
            --checkpoint "$ckpt" \
            --dataset "$dataset" \
            --model_type "$model_type" \
            --cf_method "$cf_method" \
            2>&1 | tee "$LOG_DIR/eval_${desc}_cf${cf_method}.log"
    ) &
    job_count=$((job_count+1))
    if [ "$job_count" -ge "$MAX_JOBS" ]; then
        wait -n   # 等待任意一个后台任务结束
        job_count=$((job_count-1))
    fi
}

# 查找所有 checkpoint 并评估
find_and_eval() {
    local pattern="$1"
    local dataset="$2"
    local model_type="$3"
    local desc_prefix="$4"

    for ckpt in $(find "$MODEL_DIR" -name "${pattern}*.pth" 2>/dev/null | sort); do
        local fname=$(basename "$ckpt" .pth)
        for cf in "${CF_METHODS[@]}"; do
            eval_checkpoint "$ckpt" "$dataset" "$model_type" "$cf" "${fname}"
        done
    done
}

# 按 model path 目录搜索
MODEL_SAVE_DIR="${BASE_DIR}/src_result/models"

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "===== Dataset: $DATASET | Seed: $SEED ====="

        # 搜索匹配的 checkpoint
        for ckpt in $(find "$MODEL_SAVE_DIR" -name "*${DATASET}*seed${SEED}*.pth" 2>/dev/null | sort); do
            fname=$(basename "$ckpt" .pth)

            # 判断 model_type
            if echo "$fname" | grep -qi "EAR"; then
                mtype="ear"
            elif echo "$fname" | grep -qi "GetFair"; then
                mtype="getfair"
            elif echo "$fname" | grep -qi "CCDF"; then
                mtype="ccdf"
            else
                mtype="ours"
            fi

            for cf in "${CF_METHODS[@]}"; do
                # 检查反事实文件是否存在
                cf_file="data/causal_fair/${DATASET}_test_cf_${cf}.parquet"
                if [ ! -f "$cf_file" ]; then
                    echo "  [SKIP] CF file not found: $cf_file"
                    continue
                fi
                eval_checkpoint "$ckpt" "$DATASET" "$mtype" "$cf" "${fname}"
            done
        done
    done
done

echo ""
echo "============================================="
echo "全量评估完成: $(date)"
echo "总评估数: $COUNT"
echo "============================================="

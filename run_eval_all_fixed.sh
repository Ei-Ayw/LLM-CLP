#!/bin/bash
# 重新评估所有已有 checkpoint (修复 cf_method 文件名)
# 不含 CCDF (正在重新训练)
set -e
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification

MODEL_DIR="src_result/models"
EVAL_SCRIPT="src_script/eval/eval_all_baselines.py"
CF_METHODS=("swap" "llm")

echo "===== 重新评估全部 (修复cf命名) ====="
echo "开始: $(date)"

eval_method() {
    local method=$1
    local ckpt=$2
    local ds=$3
    for cf in "${CF_METHODS[@]}"; do
        echo ">>> $method $ds cf=$cf ($(date '+%H:%M:%S'))"
        python3 ${EVAL_SCRIPT} --method $method --checkpoint "$ckpt" --dataset $ds --cf_method $cf 2>&1 | grep -E "(Macro-F1|CFR|FPED|保存至)" || true
    done
}

# Vanilla
echo ""; echo "=== Vanilla ==="
for f in ${MODEL_DIR}/Vanilla_*.pth; do
    ds=$(echo $f | grep -oP '(?<=Vanilla_)\w+(?=_seed)')
    eval_method vanilla "$f" "$ds"
done

# Davani
echo ""; echo "=== Davani ==="
for f in ${MODEL_DIR}/Davani_*.pth; do
    ds=$(echo $f | grep -oP '(?<=Davani_)\w+(?=_seed)')
    eval_method davani "$f" "$ds"
done

# Ramponi
echo ""; echo "=== Ramponi ==="
for f in ${MODEL_DIR}/Ramponi_*.pth; do
    ds=$(echo $f | grep -oP '(?<=Ramponi_)\w+(?=_seed)')
    eval_method ramponi "$f" "$ds"
done

# EAR
echo ""; echo "=== EAR ==="
for f in ${MODEL_DIR}/EAR_*.pth; do
    ds=$(echo $f | grep -oP '(?<=EAR_)\w+(?=_seed)')
    eval_method ear "$f" "$ds"
done

# GetFair
echo ""; echo "=== GetFair ==="
for f in ${MODEL_DIR}/GetFair_*.pth; do
    ds=$(echo $f | grep -oP '(?<=GetFair_)\w+(?=_seed)')
    eval_method getfair "$f" "$ds"
done

# Ours (llm_clp1.0_con0.0)
echo ""; echo "=== Ours ==="
declare -A OURS_CKPTS
OURS_CKPTS["hatexplain_42"]="hatexplain_llm_clp1.0_con0.0_seed42_0315_1709.pth"
OURS_CKPTS["hatexplain_123"]="hatexplain_llm_clp1.0_con0.0_seed123_0315_1823.pth"
OURS_CKPTS["hatexplain_2024"]="hatexplain_llm_clp1.0_con0.0_seed2024_0315_1935.pth"
OURS_CKPTS["toxigen_42"]="toxigen_llm_clp1.0_con0.0_seed42_0315_2020.pth"
OURS_CKPTS["toxigen_123"]="toxigen_llm_clp1.0_con0.0_seed123_0315_2051.pth"
OURS_CKPTS["toxigen_2024"]="toxigen_llm_clp1.0_con0.0_seed2024_0315_2125.pth"
OURS_CKPTS["dynahate_42"]="dynahate_llm_clp1.0_con0.0_seed42_0315_2206.pth"
OURS_CKPTS["dynahate_123"]="dynahate_llm_clp1.0_con0.0_seed123_0315_2250.pth"
OURS_CKPTS["dynahate_2024"]="dynahate_llm_clp1.0_con0.0_seed2024_0315_2332.pth"

for key in "${!OURS_CKPTS[@]}"; do
    ds=$(echo $key | cut -d_ -f1)
    eval_method ours "${MODEL_DIR}/${OURS_CKPTS[$key]}" "$ds"
done

echo ""
echo "===== 全部完成: $(date) ====="

#!/bin/bash
# 补齐缺失的 7metrics 评估: EAR, GetFair, Ours
# CCDF 需要重新训练（旧 checkpoint 格式不兼容）

set -e
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification

MODEL_DIR="src_result/models"
EVAL_SCRIPT="src_script/eval/eval_all_baselines.py"
MODEL_NAME="models/deberta-v3-base"

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)
CF_METHODS=("swap" "llm")

echo "=========================================="
echo "补齐缺失的 7metrics 评估"
echo "开始时间: $(date)"
echo "=========================================="

# ===== 1. EAR 评估 =====
echo ""
echo "===== Phase 1: EAR 评估 ====="
declare -A EAR_CKPTS
EAR_CKPTS["hatexplain_42"]="EAR_hatexplain_seed42_0315_0045.pth"
EAR_CKPTS["hatexplain_123"]="EAR_hatexplain_seed123_0315_0159.pth"
EAR_CKPTS["hatexplain_2024"]="EAR_hatexplain_seed2024_0315_0311.pth"
EAR_CKPTS["toxigen_42"]="EAR_toxigen_seed42_0315_0404.pth"
EAR_CKPTS["toxigen_123"]="EAR_toxigen_seed123_0315_0432.pth"
EAR_CKPTS["toxigen_2024"]="EAR_toxigen_seed2024_0315_0506.pth"
EAR_CKPTS["dynahate_42"]="EAR_dynahate_seed42_0315_0544.pth"
EAR_CKPTS["dynahate_123"]="EAR_dynahate_seed123_0315_0626.pth"
EAR_CKPTS["dynahate_2024"]="EAR_dynahate_seed2024_0315_0706.pth"

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="${EAR_CKPTS[${ds}_${seed}]}"
        for cf in "${CF_METHODS[@]}"; do
            echo ">>> Eval: EAR ${ds} seed${seed} cf=${cf}"
            python3 ${EVAL_SCRIPT} \
                --method ear \
                --checkpoint ${MODEL_DIR}/${ckpt} \
                --dataset ${ds} \
                --cf_method ${cf} \
                --model_name ${MODEL_NAME} || echo "[WARN] EAR ${ds} seed${seed} cf=${cf} failed"
        done
    done
done
echo "Phase 1 完成: $(date)"

# ===== 2. GetFair 评估 =====
echo ""
echo "===== Phase 2: GetFair 评估 ====="
declare -A GF_CKPTS
GF_CKPTS["hatexplain_42"]="GetFair_hatexplain_seed42_0315_1322.pth"
GF_CKPTS["hatexplain_123"]="GetFair_hatexplain_seed123_0315_1334.pth"
GF_CKPTS["hatexplain_2024"]="GetFair_hatexplain_seed2024_0315_1343.pth"
GF_CKPTS["toxigen_42"]="GetFair_toxigen_seed42_0315_1352.pth"
GF_CKPTS["toxigen_123"]="GetFair_toxigen_seed123_0315_1357.pth"
GF_CKPTS["toxigen_2024"]="GetFair_toxigen_seed2024_0315_1403.pth"
GF_CKPTS["dynahate_42"]="GetFair_dynahate_seed42_0315_1543.pth"
GF_CKPTS["dynahate_123"]="GetFair_dynahate_seed123_0315_1550.pth"
GF_CKPTS["dynahate_2024"]="GetFair_dynahate_seed2024_0315_1557.pth"

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="${GF_CKPTS[${ds}_${seed}]}"
        for cf in "${CF_METHODS[@]}"; do
            echo ">>> Eval: GetFair ${ds} seed${seed} cf=${cf}"
            python3 ${EVAL_SCRIPT} \
                --method getfair \
                --checkpoint ${MODEL_DIR}/${ckpt} \
                --dataset ${ds} \
                --cf_method ${cf} \
                --model_name ${MODEL_NAME} || echo "[WARN] GetFair ${ds} seed${seed} cf=${cf} failed"
        done
    done
done
echo "Phase 2 完成: $(date)"

# ===== 3. Ours 评估 (用最新的 checkpoint) =====
echo ""
echo "===== Phase 3: Ours 评估 ====="
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

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="${OURS_CKPTS[${ds}_${seed}]}"
        for cf in "${CF_METHODS[@]}"; do
            echo ">>> Eval: Ours ${ds} seed${seed} cf=${cf}"
            python3 ${EVAL_SCRIPT} \
                --method ours \
                --checkpoint ${MODEL_DIR}/${ckpt} \
                --dataset ${ds} \
                --cf_method ${cf} \
                --model_name ${MODEL_NAME} || echo "[WARN] Ours ${ds} seed${seed} cf=${cf} failed"
        done
    done
done
echo "Phase 3 完成: $(date)"

echo ""
echo "=========================================="
echo "全部评估完成: $(date)"
echo "=========================================="

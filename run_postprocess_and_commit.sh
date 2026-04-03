#!/bin/bash
# =============================================================================
# run_postprocess_and_commit.sh
# 在所有实验跑完之后执行:
#   1. 聚合骨干对比结果 (BERT/RoBERTa 3seeds)
#   2. 聚合 CLP 灵敏度结果
#   3. 生成综合实验报告 (txt/json)
#   4. 生成 Word 文档 (EMNLP格式)
#   5. 生成统一结果表格 (mean±std)
#   6. git add + commit + push 到 GitHub
#
# 用法: bash run_postprocess_and_commit.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$PROJECT_ROOT/src_script/eval"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/postprocess_$(date +%m%d_%H%M).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo " 后处理 + Commit: $(date)"                                    | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"

# 1. 聚合骨干对比结果
echo "" | tee -a "$LOG_FILE"
echo "[Step 1] 聚合骨干对比结果..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/aggregate_backbone_results.py" --backbone bert roberta \
    2>&1 | tee -a "$LOG_FILE"

# 2. 聚合 CLP 灵敏度
echo "" | tee -a "$LOG_FILE"
echo "[Step 2] 聚合 CLP 灵敏度结果..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/aggregate_clp_sensitivity.py" --backbone bert roberta deberta \
    2>&1 | tee -a "$LOG_FILE"

# 3. 生成综合报告 (txt/json)
echo "" | tee -a "$LOG_FILE"
echo "[Step 3] 生成综合实验报告..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/generate_report.py" --backbone bert roberta deberta \
    2>&1 | tee -a "$LOG_FILE"

# 4. 生成统一结果表格 (mean±std)
echo "" | tee -a "$LOG_FILE"
echo "[Step 4] 生成统一结果表格 (mean±std)..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/build_unified_table.py" --cf_method llm \
    2>&1 | tee -a "$LOG_FILE"

# 5. 生成 Word 文档
echo "" | tee -a "$LOG_FILE"
echo "[Step 5] 生成 Word 报告 (EMNLP格式)..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/generate_word_report.py" --cf_method llm \
    2>&1 | tee -a "$LOG_FILE"

# 6. Git commit + push 到当前仓库 (refactor/v2-rethink) + LLM-CLP 仓库
echo "" | tee -a "$LOG_FILE"
echo "[Step 6] Git commit + push..." | tee -a "$LOG_FILE"

# 添加新增文件（不覆盖原有文件，git add 只会暂存变更）
git add \
    src_model/model_backbone_baselines.py \
    src_model/model_backbone_cf.py \
    src_script/train/train_backbone_vanilla.py \
    src_script/train/train_backbone_ear.py \
    src_script/train/train_backbone_getfair.py \
    src_script/train/train_backbone_ccdf.py \
    src_script/train/train_backbone_davani.py \
    src_script/train/train_backbone_ramponi.py \
    src_script/train/train_backbone_causal_fair.py \
    src_script/eval/eval_backbone_baselines.py \
    src_script/eval/aggregate_backbone_results.py \
    src_script/eval/aggregate_clp_sensitivity.py \
    src_script/eval/generate_report.py \
    src_script/eval/generate_word_report.py \
    src_script/eval/build_unified_table.py \
    run_backbone_and_sensitivity.sh \
    run_backbone_extra_seeds.sh \
    run_postprocess_and_commit.sh \
    run_all_sequential.sh \
    2>&1 | tee -a "$LOG_FILE" || true

# 添加结果文件
git add src_result/eval/*.json src_result/eval/*.txt src_result/eval/*.docx \
    2>&1 | tee -a "$LOG_FILE" || true

git status 2>&1 | tee -a "$LOG_FILE"

COMMIT_MSG="$(cat <<'EOF'
feat: BERT/RoBERTa骨干对比(3seeds) + CLP灵敏度分析 + EMNLP综合报告

新增内容:
- BERT/RoBERTa backbone: 7方法×3数据集×3seeds 全量实验
- CLP灵敏度: λ_clp∈{0.2,0.4,0.6,0.8,1.0}×3backbone×3数据集×3seeds
- 统一结果表格: mean±std格式 (DeBERTa 3seeds + BERT/RoBERTa 3seeds)
- EMNLP Word报告: 含结果分析、公平性分析、方法对比

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

git commit -m "$COMMIT_MSG" 2>&1 | tee -a "$LOG_FILE"

# 推送到当前仓库 origin (refactor/v2-rethink)
echo "[Push] 推送到 origin/refactor/v2-rethink ..." | tee -a "$LOG_FILE"
git push origin HEAD 2>&1 | tee -a "$LOG_FILE"

# 推送到 LLM-CLP 仓库
echo "[Push] 推送到 LLM-CLP 仓库 ..." | tee -a "$LOG_FILE"
# 添加 llm-clp remote（如果已存在则忽略）
git remote add llm-clp git@github.com:Ei-Ayw/LLM-CLP.git 2>/dev/null || true
# 推送当前分支到 LLM-CLP 的 main 分支
git push llm-clp HEAD:main 2>&1 | tee -a "$LOG_FILE"
echo "[Push] LLM-CLP 推送完成" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部完成: $(date)"                                           | tee -a "$LOG_FILE"
echo " 日志: $LOG_FILE"                                             | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

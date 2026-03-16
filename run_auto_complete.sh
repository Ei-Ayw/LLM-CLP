#!/bin/bash
# 自动化流程：等待实验完成 → 公平性评估 → 生成报告 → Git commit
set -e

echo "=========================================="
echo "自动化实验完成流程"
echo "=========================================="

# 1. 等待 CDA fix 重跑完成
echo "[1/4] 等待 CDA fix 重跑完成..."
while pgrep -f "train_causal_fair.py" > /dev/null; do
    sleep 300  # 每 5 分钟检查一次
done
echo "✓ CDA fix 重跑完成: $(date)"

# 2. 运行公平性评估
echo ""
echo "[2/4] 运行公平性评估..."
bash run_all_eval.sh
echo "✓ 公平性评估完成: $(date)"

# 3. 生成实验报告
echo ""
echo "[3/4] 生成实验报告..."
python3 generate_report.py
echo "✓ 报告生成完成"

# 4. Git commit
echo ""
echo "[4/4] Commit 到 GitHub..."
git add docs/EXPERIMENT_REPORT.md
git add src_result/eval/*.json
git add src_result/logs/*_results.json
git commit -m "feat: 完整实验结果报告

- 完成所有对比实验 (Baseline, CDA-Swap, EAR, GetFair, CCDF, Ours)
- 完成消融实验 (CF方法 × CLP)
- 修复 CDA bug: CF 样本现在正确参与 CE loss 计算
- 添加公平性评估指标 (CFR, FPED, FNED)
- 生成完整实验报告

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

echo "✓ Git commit 完成"
echo ""
echo "=========================================="
echo "全部流程完成: $(date)"
echo "=========================================="

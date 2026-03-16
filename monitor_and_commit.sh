#!/bin/bash
# 监控公平性评估完成并自动生成报告

while true; do
    count=$(ls src_result/eval/*fairness.json 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] 公平性评估进度: $count/144"

    if [ $count -ge 144 ]; then
        echo "✅ 公平性评估完成!"
        echo "[$(date +%H:%M:%S)] 生成报告..."
        python3 generate_report.py

        echo "[$(date +%H:%M:%S)] Git commit..."
        git add docs/EXPERIMENT_REPORT.md src_result/eval/*.json src_result/logs/*_results.json
        git commit -m "feat: 完整实验结果报告

- 完成所有对比实验 (Baseline, CDA-Swap, EAR, GetFair, CCDF, Ours)
- 完成消融实验 (CF方法 × CLP)
- 修复 CDA bug: CF 样本正确参与 CE loss
- 完成公平性评估 (CFR, FPED, FNED)
- 生成完整实验报告

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
        echo "✅ 全部完成!"
        break
    fi
    sleep 300
done

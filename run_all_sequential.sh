#!/bin/bash
# =============================================================================
# run_all_sequential.sh
# 全自动流水线:
#   阶段1: 等待 run_backbone_and_sensitivity.sh 完成 (PID=444014)
#   阶段2: 补跑 BERT/RoBERTa seed=123,2024
#   阶段3: 后处理 + Word报告 + commit + push
#
# 用法: nohup bash run_all_sequential.sh > logs/sequential_$(date +%m%d_%H%M).log 2>&1 &
# =============================================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sequential_$(date +%m%d_%H%M).log"

tee_log() { echo "$1" | tee -a "$LOG_FILE"; }

tee_log "============================================================"
tee_log " 全自动流水线启动: $(date)"
tee_log "============================================================"

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────
# 阶段1: 等待 PID 444014 (当前正在跑的实验) 完成
# ─────────────────────────────────────────────────────────────────
WATCH_PID=444014
tee_log ""
tee_log "[Stage 1] 等待实验进程 PID=$WATCH_PID 完成..."

while kill -0 "$WATCH_PID" 2>/dev/null; do
    SENSITIVITY_DONE=$(grep -c '>>> Train.*sensitivity' "$LOG_DIR/backbone_sensitivity_full_0402_2027.log" 2>/dev/null || echo 0)
    tee_log "  $(date '+%H:%M:%S') PID=$WATCH_PID 仍在运行, 灵敏度实验已完成: $SENSITIVITY_DONE/90"
    sleep 300  # 每5分钟检查一次
done

tee_log "[Stage 1] PID=$WATCH_PID 已结束: $(date)"

# ─────────────────────────────────────────────────────────────────
# 阶段2: 补跑 BERT/RoBERTa seed=123,2024
# ─────────────────────────────────────────────────────────────────
tee_log ""
tee_log "[Stage 2] 开始补跑 BERT/RoBERTa seed=123,2024: $(date)"
bash "$PROJECT_ROOT/run_backbone_extra_seeds.sh" bert roberta 2>&1 | tee -a "$LOG_FILE"
tee_log "[Stage 2] 补跑完成: $(date)"

# ─────────────────────────────────────────────────────────────────
# 阶段3: 后处理 + Word报告 + commit + push
# ─────────────────────────────────────────────────────────────────
tee_log ""
tee_log "[Stage 3] 后处理 + 生成报告 + commit + push: $(date)"
bash "$PROJECT_ROOT/run_postprocess_and_commit.sh" 2>&1 | tee -a "$LOG_FILE"
tee_log "[Stage 3] 完成: $(date)"

tee_log ""
tee_log "============================================================"
tee_log " 全部流水线完成: $(date)"
tee_log " 日志: $LOG_FILE"
tee_log "============================================================"

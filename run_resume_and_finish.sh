#!/bin/bash
# =============================================================================
# run_resume_and_finish.sh (v2 - 灵敏度分析只在DeBERTa上跑，已完成)
# 现在只需要:
#   [Part A补] BERT/RoBERTa 骨干对比 seed=123,2024
#   [Post]     后处理 + Word报告 + commit + push
# =============================================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/resume_v2_$(date +%m%d_%H%M).log"
TRAIN_DIR="$PROJECT_ROOT/src_script/train"
EVAL_DIR="$PROJECT_ROOT/src_script/eval"

tee_log() { echo "$1" | tee -a "$LOG_FILE"; }

tee_log "============================================================"
tee_log " 续跑(v2): 骨干对比 seed=123,2024 + 后处理: $(date)"
tee_log " 注: 灵敏度分析只在DeBERTa-v3上跑，已完成，BERT/RoBERTa灵敏度跳过"
tee_log "============================================================"

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────
# 补跑 BERT/RoBERTa 骨干对比 seed=123,2024（边跑边删权重节省空间）
# ─────────────────────────────────────────────────────────────────
tee_log ""
tee_log "[Part A补] BERT/RoBERTa 骨干对比 seed=123,2024: $(date)"

METHOD_ENTRIES=(
    "vanilla:train_backbone_vanilla.py:Vanilla:"
    "ear:train_backbone_ear.py:EAR:--lambda_ear 0.1"
    "getfair:train_backbone_getfair.py:GetFair:--lambda_gf 0.5"
    "ccdf:train_backbone_ccdf.py:CCDF:--lambda_kl 1.0 --tde_alpha 0.5 --bias_epochs 5"
    "davani:train_backbone_davani.py:Davani:--lambda_lp 1.0 --cf_method swap"
    "ramponi:train_backbone_ramponi.py:Ramponi:--lambda_adv 0.1"
    "ours:train_backbone_causal_fair.py:Ours:--cf_method llm --lambda_clp 1.0 --lambda_con 0.5"
)

EXTRA_SEEDS=(123 2024)
DATASETS=(hatexplain toxigen dynahate)

for BACKBONE in bert roberta; do
    for SEED in "${EXTRA_SEEDS[@]}"; do
        tee_log ""
        tee_log "  === $BACKBONE seed=$SEED ==="
        for entry in "${METHOD_ENTRIES[@]}"; do
            METHOD=$(echo "$entry" | cut -d: -f1)
            SCRIPT=$(echo "$entry" | cut -d: -f2)
            PREFIX=$(echo "$entry" | cut -d: -f3)
            EXTRA=$(echo "$entry" | cut -d: -f4-)

            for DS in "${DATASETS[@]}"; do
                tee_log "  >>> Train: $PREFIX $BACKBONE $DS seed$SEED"

                python3 "$TRAIN_DIR/$SCRIPT" \
                    --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                    --epochs 3 --batch_size 32 --lr 2e-5 $EXTRA \
                    2>&1 | tee -a "$LOG_FILE"

                # Find checkpoint
                if [ "$METHOD" = "ours" ]; then
                    CKPT=$(ls -t "$PROJECT_ROOT/src_result/models"/Ours_${BACKBONE}_${DS}_llm_clp1.0_con0.5_seed${SEED}_*.pth 2>/dev/null | head -1)
                elif [ "$METHOD" = "ccdf" ]; then
                    CKPT=$(ls -t "$PROJECT_ROOT/src_result/models"/CCDF_${BACKBONE}_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
                else
                    CKPT=$(ls -t "$PROJECT_ROOT/src_result/models"/${PREFIX}_${BACKBONE}_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
                fi

                if [ -n "${CKPT:-}" ]; then
                    for CF in swap llm; do
                        tee_log "    eval $METHOD $BACKBONE $DS cf=$CF"
                        python3 "$EVAL_DIR/eval_backbone_baselines.py" \
                            --method "$METHOD" --backbone "$BACKBONE" \
                            --checkpoint "$CKPT" --dataset "$DS" \
                            --cf_method "$CF" 2>&1 | tee -a "$LOG_FILE"
                    done
                    rm -f "$CKPT"
                else
                    tee_log "  [WARN] Checkpoint not found for $PREFIX $BACKBONE $DS seed$SEED"
                fi
            done
        done
    done
done

tee_log "[Part A补] 完成: $(date)"

# ─────────────────────────────────────────────────────────────────
# 后处理 + Word报告 + commit + push
# ─────────────────────────────────────────────────────────────────
tee_log ""
tee_log "[Post] 后处理 + 报告 + push: $(date)"
bash "$PROJECT_ROOT/run_postprocess_and_commit.sh" 2>&1 | tee -a "$LOG_FILE"

tee_log ""
tee_log "============================================================"
tee_log " 全部完成: $(date)"
tee_log "============================================================"

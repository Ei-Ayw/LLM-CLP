#!/bin/bash
# =============================================================================
# run_backbone_and_sensitivity.sh
# 一次性跑完:
#   (A) BERT / RoBERTa 骨干对比实验 — 7 个方法 × 3 数据集 × 3 seed × 2 CF 方式
#   (B) CausalFair CLP 灵敏度分析  — λ_clp ∈ {0.2,0.4,0.6,0.8,1.0}
#                                     × 3 数据集 × 3 seed × 3 backbone × 2 CF 方式
#   (C) 聚合结果 JSON
#   (D) 生成综合实验报告
#
# 用法:
#   bash run_backbone_and_sensitivity.sh [BACKBONE_LIST]
#   例:
#     bash run_backbone_and_sensitivity.sh bert roberta
#     bash run_backbone_and_sensitivity.sh bert roberta deberta  (全部)
#
# 注意: 训练后自动用 eval_backbone_baselines.py 评估 7 指标
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------
BACKBONES=("${@:-bert roberta}")          # 默认 bert + roberta
if [ $# -eq 0 ]; then
    BACKBONES=("bert" "roberta")
else
    BACKBONES=("$@")
fi

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)
BACKBONE_SEEDS=(42)          # Part A 骨干对比只跑 seed=42，节省时间
CF_METHODS=("swap" "llm")
LAMBDA_CLPS=("0.2" "0.4" "0.6" "0.8" "1.0")
EPOCHS=3                     # 3 epoch + 早停，与 DeBERTa 实验对齐
BATCH_SIZE=32                # 双卡 batch_size=32

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/backbone_sensitivity_$(date +%m%d_%H%M).log"

TRAIN_DIR="$PROJECT_ROOT/src_script/train"
EVAL_DIR="$PROJECT_ROOT/src_script/eval"
MODEL_DIR="$PROJECT_ROOT/src_result/models"
RESULT_DIR="$PROJECT_ROOT/src_result/eval"

echo "============================================================" | tee -a "$LOG_FILE"
echo " Backbone + Sensitivity 全量实验: $(date)"              | tee -a "$LOG_FILE"
echo " Backbones : ${BACKBONES[*]}"                           | tee -a "$LOG_FILE"
echo " Datasets  : ${DATASETS[*]}"                            | tee -a "$LOG_FILE"
echo " Seeds     : ${SEEDS[*]}"                               | tee -a "$LOG_FILE"
echo " λ_clp vals: ${LAMBDA_CLPS[*]}"                        | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"


# =============================================================================
# 辅助函数
# =============================================================================
find_latest_model() {
    # find_latest_model <prefix_glob>
    # 例: find_latest_model "Vanilla_bert_hatexplain_seed42_*"
    local pat="$1"
    ls -t "$MODEL_DIR"/${pat}.pth 2>/dev/null | head -1
}

run_eval() {
    local method="$1"     # e.g. vanilla
    local backbone="$2"   # e.g. bert
    local dataset="$3"
    local cf_method="$4"
    local ckpt="$5"
    local extra_tag="${6:-}"

    echo "  >>> Eval: $method $backbone $dataset cf=$cf_method" | tee -a "$LOG_FILE"
    python3 "$EVAL_DIR/eval_backbone_baselines.py" \
        --method     "$method" \
        --backbone   "$backbone" \
        --checkpoint "$ckpt" \
        --dataset    "$dataset" \
        --cf_method  "$cf_method" \
        2>&1 | tee -a "$LOG_FILE"
}


# =============================================================================
# PART A — 骨干对比���验 (Backbone Comparison)
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " PART A: 骨干对比实验"                                        | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"


for BACKBONE in "${BACKBONES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "  ====== Backbone: $BACKBONE ======" | tee -a "$LOG_FILE"

    # ------------------------------------------------------------------
    # A1. Vanilla
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A1. Vanilla ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: Vanilla $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_vanilla.py" \
                --backbone  "$BACKBONE" \
                --dataset   "$DS" \
                --seed      "$SEED" \
                --epochs    $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "Vanilla_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: Vanilla_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "vanilla" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A2. EAR
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A2. EAR ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: EAR $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ear.py" \
                --backbone   "$BACKBONE" \
                --dataset    "$DS" \
                --seed       "$SEED" \
                --lambda_ear 0.1 \
                --epochs     $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "EAR_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: EAR_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "ear" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A3. GetFair
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A3. GetFair ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: GetFair $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_getfair.py" \
                --backbone   "$BACKBONE" \
                --dataset    "$DS" \
                --seed       "$SEED" \
                --lambda_gf  0.5 \
                --epochs     $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "GetFair_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: GetFair_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "getfair" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A4. CCDF
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A4. CCDF ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: CCDF $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ccdf.py" \
                --backbone   "$BACKBONE" \
                --dataset    "$DS" \
                --seed       "$SEED" \
                --lambda_kl  1.0 --tde_alpha 0.5 --bias_epochs 5 \
                --epochs     $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "CCDF_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: CCDF_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "ccdf" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A5. Davani
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A5. Davani ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: Davani $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_davani.py" \
                --backbone   "$BACKBONE" \
                --dataset    "$DS" \
                --seed       "$SEED" \
                --lambda_lp  1.0 --cf_method swap \
                --epochs     $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "Davani_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: Davani_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "davani" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A6. Ramponi
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A6. Ramponi ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            echo "  >>> Train: Ramponi $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ramponi.py" \
                --backbone    "$BACKBONE" \
                --dataset     "$DS" \
                --seed        "$SEED" \
                --lambda_adv  0.1 \
                --epochs      $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"

            CKPT=$(find_latest_model "Ramponi_${BACKBONE}_${DS}_seed${SEED}_*")
            if [ -z "$CKPT" ]; then
                echo "  [WARN] Model not found: Ramponi_${BACKBONE}_${DS}_seed${SEED}" | tee -a "$LOG_FILE"
                continue
            fi
            for CF in "${CF_METHODS[@]}"; do
                run_eval "ramponi" "$BACKBONE" "$DS" "$CF" "$CKPT"
            done
        done
    done

    # ------------------------------------------------------------------
    # A7. CausalFair (Ours) — λ_clp=1.0, λ_con=0.5 (default settings)
    # ------------------------------------------------------------------
    echo "" | tee -a "$LOG_FILE"
    echo "  ----- A7. CausalFair/Ours ($BACKBONE) -----" | tee -a "$LOG_FILE"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${BACKBONE_SEEDS[@]}"; do
            for CF_TRAIN in "llm"; do   # primary CF method for training
                echo "  >>> Train: Ours $BACKBONE $DS seed$SEED cf=$CF_TRAIN" | tee -a "$LOG_FILE"
                python3 "$TRAIN_DIR/train_backbone_causal_fair.py" \
                    --backbone    "$BACKBONE" \
                    --dataset     "$DS" \
                    --seed        "$SEED" \
                    --cf_method   "$CF_TRAIN" \
                    --lambda_clp  1.0 \
                    --lambda_con  0.5 \
                    --epochs      $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                    2>&1 | tee -a "$LOG_FILE"

                CKPT=$(find_latest_model "Ours_${BACKBONE}_${DS}_${CF_TRAIN}_clp1.0_con0.5_seed${SEED}_*")
                if [ -z "$CKPT" ]; then
                    echo "  [WARN] Model not found: Ours_${BACKBONE}_${DS}_${CF_TRAIN}_seed${SEED}" | tee -a "$LOG_FILE"
                    continue
                fi
                for CF in "${CF_METHODS[@]}"; do
                    run_eval "ours" "$BACKBONE" "$DS" "$CF" "$CKPT"
                done
            done
        done
    done

done  # end BACKBONE loop (Part A)


# =============================================================================
# PART B — CLP 灵敏度分析 (所有 backbone × 数据集 × seed)
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " PART B: CLP 灵敏度分析 (λ_clp ∈ {0.2,0.4,0.6,0.8,1.0})"   | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Sensitivity uses llm CF (primary evaluation CF); backbone list from Part A
for BACKBONE in "${BACKBONES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for LAMBDA_CLP in "${LAMBDA_CLPS[@]}"; do
                echo "" | tee -a "$LOG_FILE"
                echo "  >>> Train: Ours $BACKBONE $DS seed$SEED λ_clp=$LAMBDA_CLP (sensitivity)" | tee -a "$LOG_FILE"
                python3 "$TRAIN_DIR/train_backbone_causal_fair.py" \
                    --backbone    "$BACKBONE" \
                    --dataset     "$DS" \
                    --seed        "$SEED" \
                    --cf_method   "llm" \
                    --lambda_clp  "$LAMBDA_CLP" \
                    --lambda_con  0.5 \
                    --epochs      $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                    2>&1 | tee -a "$LOG_FILE"

                # λ_clp=1.0 was already trained in Part A (with llm); skip duplicate eval
                # but we still run eval for ALL clp values to collect sensitivity data
                CKPT=$(find_latest_model "Ours_${BACKBONE}_${DS}_llm_clp${LAMBDA_CLP}_con0.5_seed${SEED}_*")
                if [ -z "$CKPT" ]; then
                    echo "  [WARN] Model not found: Ours_${BACKBONE}_${DS}_llm_clp${LAMBDA_CLP}_seed${SEED}" | tee -a "$LOG_FILE"
                    continue
                fi
                # Evaluate with both CF methods for completeness
                for CF in "${CF_METHODS[@]}"; do
                    run_eval "ours" "$BACKBONE" "$DS" "$CF" "$CKPT"
                done
            done
        done
    done
done


# =============================================================================
# PART C — 聚合结果
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " PART C: 聚合结果 JSON"                                       | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "  >>> 聚合骨干对比结果..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/aggregate_backbone_results.py" \
    --backbone "${BACKBONES[@]}" \
    2>&1 | tee -a "$LOG_FILE"

echo "  >>> 聚合 CLP 灵敏度结果..." | tee -a "$LOG_FILE"
python3 "$EVAL_DIR/aggregate_clp_sensitivity.py" \
    --backbone "${BACKBONES[@]}" \
    2>&1 | tee -a "$LOG_FILE"


# =============================================================================
# PART D — 生成综合实验报告
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " PART D: 生成综合实验报告"                                     | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

python3 "$EVAL_DIR/generate_report.py" \
    --backbone "${BACKBONES[@]}" \
    2>&1 | tee -a "$LOG_FILE"


echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部完成: $(date)"                                            | tee -a "$LOG_FILE"
echo " 日志: $LOG_FILE"                                              | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

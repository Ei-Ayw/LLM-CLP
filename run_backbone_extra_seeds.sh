#!/bin/bash
# =============================================================================
# run_backbone_extra_seeds.sh
# 补跑 BERT / RoBERTa 骨干对比实验的 seed=123 和 seed=2024
# （Part A 已经跑了 seed=42，这里只补 seed=123, 2024）
#
# 用法:
#   bash run_backbone_extra_seeds.sh [bert] [roberta]
# =============================================================================
set -euo pipefail

if [ $# -eq 0 ]; then
    BACKBONES=("bert" "roberta")
else
    BACKBONES=("$@")
fi

DATASETS=("hatexplain" "toxigen" "dynahate")
EXTRA_SEEDS=(123 2024)          # 只补跑这两个 seed
CF_METHODS=("swap" "llm")
EPOCHS=3
BATCH_SIZE=32

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/backbone_extra_seeds_$(date +%m%d_%H%M).log"

TRAIN_DIR="$PROJECT_ROOT/src_script/train"
EVAL_DIR="$PROJECT_ROOT/src_script/eval"
MODEL_DIR="$PROJECT_ROOT/src_result/models"

echo "============================================================" | tee -a "$LOG_FILE"
echo " 补跑骨干对比 seed=123,2024: $(date)"                        | tee -a "$LOG_FILE"
echo " Backbones : ${BACKBONES[*]}"                                 | tee -a "$LOG_FILE"
echo " Extra Seeds: ${EXTRA_SEEDS[*]}"                              | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

find_latest_model() {
    local pat="$1"
    ls -t "$MODEL_DIR"/${pat}.pth 2>/dev/null | head -1
}

run_eval() {
    local method="$1"
    local backbone="$2"
    local dataset="$3"
    local cf_method="$4"
    local ckpt="$5"
    echo "  >>> Eval: $method $backbone $dataset cf=$cf_method" | tee -a "$LOG_FILE"
    python3 "$EVAL_DIR/eval_backbone_baselines.py" \
        --method     "$method" \
        --backbone   "$backbone" \
        --checkpoint "$ckpt" \
        --dataset    "$dataset" \
        --cf_method  "$cf_method" \
        2>&1 | tee -a "$LOG_FILE"
}

for BACKBONE in "${BACKBONES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "====== Backbone: $BACKBONE ======" | tee -a "$LOG_FILE"

    for SEED in "${EXTRA_SEEDS[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "  ------ seed=$SEED ------" | tee -a "$LOG_FILE"

        # --- Vanilla ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: Vanilla $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_vanilla.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "Vanilla_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] Vanilla not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "vanilla" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- EAR ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: EAR $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ear.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --lambda_ear 0.1 \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "EAR_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] EAR not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "ear" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- GetFair ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: GetFair $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_getfair.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --lambda_gf 0.5 \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "GetFair_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] GetFair not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "getfair" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- CCDF ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: CCDF $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ccdf.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --lambda_kl 1.0 --tde_alpha 0.5 --bias_epochs 5 \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "CCDF_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] CCDF not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "ccdf" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- Davani ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: Davani $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_davani.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --lambda_lp 1.0 --cf_method swap \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "Davani_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] Davani not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "davani" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- Ramponi ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: Ramponi $BACKBONE $DS seed$SEED" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_ramponi.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --lambda_adv 0.1 \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "Ramponi_${BACKBONE}_${DS}_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] Ramponi not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "ramponi" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

        # --- Ours/CausalFair ---
        for DS in "${DATASETS[@]}"; do
            echo "  >>> Train: Ours $BACKBONE $DS seed$SEED cf=llm" | tee -a "$LOG_FILE"
            python3 "$TRAIN_DIR/train_backbone_causal_fair.py" \
                --backbone "$BACKBONE" --dataset "$DS" --seed "$SEED" \
                --cf_method llm --lambda_clp 1.0 --lambda_con 0.5 \
                --epochs $EPOCHS --batch_size $BATCH_SIZE --lr 2e-5 \
                2>&1 | tee -a "$LOG_FILE"
            CKPT=$(find_latest_model "Ours_${BACKBONE}_${DS}_llm_clp1.0_con0.5_seed${SEED}_*")
            [ -z "$CKPT" ] && echo "  [WARN] Ours not found" | tee -a "$LOG_FILE" && continue
            for CF in "${CF_METHODS[@]}"; do run_eval "ours" "$BACKBONE" "$DS" "$CF" "$CKPT"; done
        done

    done  # end SEED loop
done  # end BACKBONE loop

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 补跑完成: $(date)"                                           | tee -a "$LOG_FILE"
echo " 日志: $LOG_FILE"                                             | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

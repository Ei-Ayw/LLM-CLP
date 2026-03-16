#!/bin/bash
# =============================================================================
# 统一训练 + 评估所有 Baseline 方法
# 使用 eval_all_baselines.py 统一 7 指标评估
# =============================================================================
set -e

LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/all_baselines_$(date +%m%d_%H%M).log"

DATASETS=("hatexplain" "toxigen" "dynahate")
SEEDS=(42 123 2024)
CF_METHODS=("swap" "llm")

echo "========================================" | tee -a $LOG_FILE
echo "开始全部 Baseline 训练+评估: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# =============================================================================
# 1. Vanilla (无去偏)
# =============================================================================
echo -e "\n===== [1/5] Vanilla =====" | tee -a $LOG_FILE
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Vanilla $DS seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_vanilla.py \
            --dataset $DS --seed $SEED \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE
    done
done

# =============================================================================
# 2. CCDF (TDE 去偏)
# =============================================================================
echo -e "\n===== [2/5] CCDF =====" | tee -a $LOG_FILE
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: CCDF $DS seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_ccdf.py \
            --dataset $DS --seed $SEED \
            --lambda_kl 1.0 --tde_alpha 0.5 --bias_epochs 5 \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE
    done
done

# =============================================================================
# 3. Davani (Logit Pairing)
# =============================================================================
echo -e "\n===== [3/5] Davani =====" | tee -a $LOG_FILE
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Davani $DS seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_davani.py \
            --dataset $DS --seed $SEED \
            --lambda_lp 1.0 --cf_method swap \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE
    done
done

# =============================================================================
# 4. Ramponi (对抗去偏)
# =============================================================================
echo -e "\n===== [4/5] Ramponi =====" | tee -a $LOG_FILE
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ">>> Train: Ramponi $DS seed$SEED" | tee -a $LOG_FILE
        python3 src_script/train/train_baseline_ramponi.py \
            --dataset $DS --seed $SEED \
            --lambda_adv 0.1 \
            --epochs 6 --batch_size 16 --lr 2e-5 \
            2>&1 | tee -a $LOG_FILE
    done
done

# =============================================================================
# 5. 统一评估 (7 指标)
# =============================================================================
echo -e "\n===== [5/5] 统一评估 =====" | tee -a $LOG_FILE

METHODS=("vanilla" "ccdf" "davani" "ramponi")
METHOD_PREFIXES=("Vanilla" "CCDF" "Davani" "Ramponi")

for idx in "${!METHODS[@]}"; do
    METHOD=${METHODS[$idx]}
    PREFIX=${METHOD_PREFIXES[$idx]}

    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # 找到最新的模型文件
            MODEL_FILE=$(ls -t src_result/models/${PREFIX}_${DS}_seed${SEED}_*.pth 2>/dev/null | head -1)
            if [ -z "$MODEL_FILE" ]; then
                echo "[SKIP] 未找到: ${PREFIX}_${DS}_seed${SEED}" | tee -a $LOG_FILE
                continue
            fi

            for CF in "${CF_METHODS[@]}"; do
                echo ">>> Eval: $METHOD $DS seed$SEED cf=$CF" | tee -a $LOG_FILE
                python3 src_script/eval/eval_all_baselines.py \
                    --method $METHOD \
                    --checkpoint "$MODEL_FILE" \
                    --dataset $DS \
                    --cf_method $CF \
                    2>&1 | tee -a $LOG_FILE
            done
        done
    done
done

echo -e "\n========================================" | tee -a $LOG_FILE
echo "全部完成: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

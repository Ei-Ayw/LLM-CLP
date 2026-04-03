#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
cd /root/lanyun-fs/01nlpchongou/01_nlp_toxicity_classification
MODEL_PATH="models/deberta-v3-base"
API_KEYS="ad443e864e1046838e00b2072c798320.9k8PEtdmWiQV8Yw2,5f2e493b563348e28d9748fe0020f760.Y2yxIZICZPk1XQBn"

echo "========================================================"
echo " Phase 0: ToxiGen LLM 反事实生成"
echo " $(date)"
echo "========================================================"
python src_script/counterfactual/cf_generator_llm.py --dataset toxigen --split train --api zhipu --api_key "$API_KEYS" --max_workers 30
python src_script/counterfactual/cf_generator_llm.py --dataset toxigen --split test --api zhipu --api_key "$API_KEYS" --max_workers 30
echo ">>> Phase 0 done: $(date)"

echo "========================================================"
echo " Phase 1: E5 超参调优 (HateXplain, seed=42)"
echo " $(date)"
echo "========================================================"
for CON in 0.1 0.2 0.3; do
    echo "--- E5: clp=1.0, con=$CON ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con $CON --seed 42
done
for CLP in 0.5; do
    for CON in 0.1 0.2; do
        echo "--- E5: clp=$CLP, con=$CON ---"
        python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp $CLP --lambda_con $CON --seed 42
    done
done
echo ">>> Phase 1 done: $(date)"

echo "========================================================"
echo " Phase 2: HateXplain 多 Seed (123, 2024)"
echo " $(date)"
echo "========================================================"
for SEED in 123 2024; do
    echo "--- E1 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method none --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 0.0 --lambda_con 0.0 --seed $SEED
    echo "--- E2 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method swap --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED
    echo "--- E3 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED
    echo "--- E4 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 0.0 --lambda_con 0.5 --seed $SEED
    echo "--- E5 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset hatexplain --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.5 --seed $SEED
done
echo ">>> Phase 2 done: $(date)"

echo "========================================================"
echo " Phase 3: ToxiGen 全实验 (3 seeds)"
echo " $(date)"
echo "========================================================"
for SEED in 42 123 2024; do
    echo "--- ToxiGen E1 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset toxigen --cf_method none --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 0.0 --lambda_con 0.0 --seed $SEED
    echo "--- ToxiGen E2 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset toxigen --cf_method swap --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED
    echo "--- ToxiGen E3 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset toxigen --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.0 --seed $SEED
    echo "--- ToxiGen E4 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset toxigen --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 0.0 --lambda_con 0.5 --seed $SEED
    echo "--- ToxiGen E5 seed=$SEED ---"
    python src_script/train/train_causal_fair.py --dataset toxigen --cf_method llm --model_name "$MODEL_PATH" --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 --lambda_clp 1.0 --lambda_con 0.5 --seed $SEED
done
echo ">>> Phase 3 done: $(date)"

echo "========================================================"
echo " Phase 4: 全量因果公平评估"
echo " $(date)"
echo "========================================================"
for CKPT in src_result/models/hatexplain_*.pth; do
    echo "评估(HX): $CKPT"
    python src_script/eval/eval_causal_fairness.py --checkpoint "$CKPT" --dataset hatexplain --cf_method llm || true
done
for CKPT in src_result/models/toxigen_*.pth; do
    echo "评估(TG): $CKPT"
    python src_script/eval/eval_causal_fairness.py --checkpoint "$CKPT" --dataset toxigen --cf_method llm || true
done
echo "========================================================"
echo " ALL DONE! $(date)"
echo "========================================================"

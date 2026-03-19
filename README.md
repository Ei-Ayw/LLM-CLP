# LLM-CLP

Counterfactual Logit Pairing for Causally Fair Toxicity Classification

## 概述

毒性检测模型常对不同身份群体产生偏见预测。LLM-CLP 通过两个核心组件解决这一问题：

1. 利用 LLM（GLM-4-Flash）生成语义自然的反事实样本，替代朴素词替换
2. 通过 Counterfactual Logit Pairing (CLP) 损失最小化原始文本与反事实文本的输出分布差异

```
L = L_CE(f(x), y) + λ · D_KL(f(x) ∥ f(x'))
```

在 HateXplain 上将 CFR 降低 66.9%，F1 仅下降 1.9%。

## 项目结构

```
├── docs/
│   └── EMNLP_2026_Paper_Draft.docx        # 论文初稿
├── src_model/
│   ├── model_deberta_cf.py                 # 主模型：DeBERTa-v3 + CLP
│   └── model_vanilla_deberta_v3.py         # Vanilla baseline
├── src_script/
│   ├── train/
│   │   ├── train_causal_fair.py            # 主训练入口（LLM+CLP）
│   │   ├── train_baseline_vanilla.py       # Vanilla fine-tuning
│   │   ├── train_baseline_ear.py           # EAR (Han et al., 2022)
│   │   ├── train_baseline_getfair.py       # GetFair (Chhabra et al., 2024)
│   │   ├── train_baseline_ccdf.py          # CCDF (Qian et al., 2023)
│   │   ├── train_baseline_davani.py        # LogitPairing (Davani et al., 2021)
│   │   └── train_baseline_ramponi.py       # AdvDebias (Ramponi & Tonelli, 2022)
│   ├── eval/
│   │   └── eval_causal_fairness.py         # 评估：CFR / CTFG / FPED / FNED
│   ├── counterfactual/
│   │   ├── cf_generator_llm.py             # LLM 反事实生成（GLM-4-Flash）
│   │   ├── cf_generator_swap.py            # 朴素词替换 baseline
│   │   └── cf_validator.py                 # 生成质量校验
│   ├── data/
│   │   └── data_loader_cf.py               # 反事实配对 DataLoader
│   └── utils/
│       ├── loss_contrastive.py             # CLP + SupCon 损失
│       ├── train_utils.py                  # EarlyStopping 等
│       └── path_config.py                  # 路径配置
└── requirements.txt
```

## 数据集

| 数据集 | 来源 | 说明 |
|--------|------|------|
| HateXplain | Mathew et al., 2021 | 20,148 条社交媒体仇恨言论，含身份标签 |
| ToxiGen | Hartvigsen et al., 2022 | 机器生成，覆盖 13 个少数群体 |
| DynaHate | Vidgen et al., 2021 | 动态对抗生成，4 轮迭代 |

## Baselines

| 方法 | 论文 | 训练脚本 |
|------|------|----------|
| Vanilla | - | `train_baseline_vanilla.py` |
| EAR | Han et al., 2022 | `train_baseline_ear.py` |
| GetFair | Chhabra et al., 2024 | `train_baseline_getfair.py` |
| CCDF | Qian et al., 2023 | `train_baseline_ccdf.py` |
| LogitPairing | Davani et al., 2021 | `train_baseline_davani.py` |
| AdvDebias | Ramponi & Tonelli, 2022 | `train_baseline_ramponi.py` |

## 快速开始

```bash
pip install -r requirements.txt

# 训练 LLM+CLP（主方法）
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name models/deberta-v3-base \
    --epochs 5 --batch_size 48 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 --seed 42

# 训练 baseline
python src_script/train/train_baseline_vanilla.py --dataset hatexplain --seed 42

# 评估因果公平性
python src_script/eval/eval_causal_fairness.py \
    --checkpoint <model_path> \
    --dataset hatexplain --cf_method llm
```

## 环境

- Python 3.8+
- PyTorch + Transformers
- GPU: NVIDIA RTX 3090 24GB
- Backbone: `microsoft/deberta-v3-base` (184M)
- 反事实生成: 智谱 GLM-4-Flash API

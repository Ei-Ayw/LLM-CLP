# LLM-CLP

Counterfactual Logit Pairing for Causally Fair Toxicity Classification

## 概述

毒性检测模型常对不同身份群体产生偏见预测。LLM-CLP 通过两个核心组件解决这一问题：

1. 利用 LLM（GLM-4-Flash）生成语义自然的反事实样本，替代朴素词替换
2. 通过 Counterfactual Logit Pairing (CLP) 损失最小化原始文本与反事实文本的输出分布差异

```
L_CLP = MSE(z, z_cf)
L = L_CE + λ_clp · L_CLP
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

### 数据获取与预处理

受限于数据使用协议与文件大小（GitHub 容量限制），本项目仓库不直接包含处理好的原始及反事实数据集文件。你在运行代码前需要自行拉取并生成数据：

1. **获取原始数据**
   - **HateXplain**: 遵循 [HateXplain 官方仓库](https://github.com/hate-alert/HateXplain) 获取数据集。
   - **ToxiGen**: 推荐使用 HuggingFace Datasets 官方 API 拉取 `load_dataset("skg/toxigen-data")`。
   - **DynaHate**: 参见 [DynaHate 官方指南](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) 获取四轮对抗数据。
   - *说明*：获取后请统一处理为 `.parquet` 格式字典（包含 `text` 和 `label` 字段），拆分为 `train/val/test` 并存放在工程的 `data/causal_fair/` 目录下（如 `hatexplain_train.parquet`）。

2. **生成 LLM 反事实数据 (Counterfactual Generation)**
   原始训练集准备好后，首先调用本仓库提供的反事实生成脚本进行数据扩增和身份词改写：
   ```bash
   python src_script/counterfactual/cf_generator_llm.py \
       --dataset hatexplain \
       --split train \
       --api zhipu \
       --api_key "YOUR_API_KEY_HERE"
   ```
   *运行完毕后，程序会自动在同级目录生成如 `hatexplain_train_cf_llm.parquet` 的反事实成对数据集文件，之后即可正常传入主训练脚本开始训练。*

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

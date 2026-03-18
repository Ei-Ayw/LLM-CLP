# LLM-CLP: 基于大模型反事实数据增强的因果公平毒性检测框架

## 简介

LLM-CLP 是一个针对文本毒性分类中身份偏见问题的去偏框架。核心思路：利用大模型（GLM-4-Flash）生成语义自然的反事实样本，结合 Counterfactual Logit Pairing（CLP）损失进行因果公平训练，使模型对不同身份群体的预测保持一致。

在 HateXplain 数据集上，LLM-CLP 将反事实翻转率（CFR）降低了 64%，F1 仅下降不到 1%。

## 核心方法

### 训练目标

```
L_total = L_CE + λ_clp × L_CLP
```

- L_CE：原始样本的交叉熵分类损失
- L_CLP：原始文本与反事实文本之间的 logit 配对损失（MSE），强制模型对身份替换后的文本给出相同预测

### LLM 反事实生成

```
原始文本:    "Muslims are destroying this country"
        ↓ GLM-4-Flash 改写
反事实文本:  "Christians are ruining this nation"
        （文化适配、情感保持）

vs. 朴素替换: "Christians are destroying this country"
        （机械替换、文化不一致）
```

生成流程：身份词检测 → 选择替换目标 → LLM 改写（保持句法、情感、毒性等级，仅替换身份相关词） → 质量校验

### 模型架构

```
输入文本 → DeBERTa-v3-base → [CLS] pooling → 二分类器
```

- Backbone: `microsoft/deberta-v3-base` (184M)
- 序列长度: 128
- AMP (FP16) + cosine LR

## 主要结果

### HateXplain

| 方法 | Macro-F1 | AUC | CFR ↓ | CTFG ↓ |
|------|:--------:|:---:|:-----:|:------:|
| Baseline | .7973 | .8882 | .1568 | .1446 |
| Swap+CLP | .7995 | .8877 | .1244 | .1077 |
| **LLM+CLP (Ours)** | **.7883** | **.8774** | **.0563** | **.0429** |

### ToxiGen

| 方法 | Macro-F1 | AUC | CFR ↓ | CTFG ↓ |
|------|:--------:|:---:|:-----:|:------:|
| Baseline | .8495 | .9318 | .0424 | .0429 |
| Swap+CLP | .8595 | .9375 | .0256 | .0248 |
| **LLM+CLP (Ours)** | **.8589** | **.9304** | **.0256** | **.0203** |

## 项目结构

```
├── src_model/                          # 模型定义
│   ├── model_deberta_cf.py             # 主模型：DeBERTa + 分类器 + 投影头
│   └── ...                             # 其他 baseline 模型
├── src_script/
│   ├── train/                          # 训练脚本
│   │   └── train_causal_fair.py        # 主训练入口 (E1-E5)
│   ├── eval/                           # 评估脚本
│   │   └── eval_causal_fairness.py     # CFR/CTFG/分组评估
│   ├── data/                           # 数��加载
│   │   └── data_loader_cf.py           # 反事实配对 DataLoader
│   ├── counterfactual/                 # 反事实生成
│   │   ├── cf_generator_llm.py         # LLM 反事实生成器 (智谱 GLM)
│   │   ├── cf_generator_swap.py        # 朴素词替换 baseline
│   │   └── cf_validator.py             # 质量校验
│   └── utils/                          # 工具函数
│       ├── loss_contrastive.py         # CLP + SupCon 损失
│       ├── train_utils.py              # EarlyStopping 等
│       └── path_config.py              # 路径管理
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练（LLM+CLP，推荐配置）
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name models/deberta-v3-base \
    --epochs 5 --batch_size 48 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 --seed 42

# 评估因果公平性
python src_script/eval/eval_causal_fairness.py \
    --checkpoint src_result/models/<checkpoint>.pth \
    --dataset hatexplain --cf_method llm
```

## 环境要求

- Python 3.8+
- PyTorch + Transformers
- GPU: NVIDIA RTX 3090 24GB
- 预训练模型: `microsoft/deberta-v3-base`
- 反事实生成 LLM: 智谱 GLM-4-Flash

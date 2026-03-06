# Toxicity Classification with Fairness-Aware Multi-Task Learning

基于 DeBERTa-V3 的毒性评论分类系统，结合多任务学习（MTL）和公平性优化，在 Jigsaw Unintended Bias in Toxicity Classification 数据集上进行实验。

## 项目结构

```
├── src_model/                    # 模型定义
│   ├── model_deberta_v3_mtl.py   # 核心：DeBERTa-V3 + MTL (毒性/子类型/身份)
│   ├── model_bert_cnn_bilstm.py  # Baseline: BERT+CNN+BiLSTM
│   ├── model_vanilla_bert.py     # Baseline: BERT
│   ├── model_vanilla_roberta.py  # Baseline: RoBERTa
│   └── model_vanilla_deberta_v3.py # Baseline: Vanilla DeBERTa-V3
├── src_script/
│   ├── train/                    # 训练脚本
│   │   ├── train_deberta_v3_mtl_s1.py      # MTL 第一阶段 (预训练)
│   │   ├── train_deberta_v3_mtl_s2.py      # MTL 第二阶段 (微调)
│   │   ├── train_deberta_v3_fair_s2.py     # Fair S2 (公平性优化)
│   │   ├── train_vanilla_deberta_v3.py     # Vanilla DeBERTa
│   │   ├── train_vanilla_bert.py           # Vanilla BERT
│   │   ├── train_vanilla_roberta.py        # Vanilla RoBERTa
│   │   ├── train_bert_cnn_bilstm.py        # BERT+CNN+BiLSTM
│   │   └── train_classical_tfidf_lr.py     # TF-IDF + LR
│   ├── eval/                     # 评估脚本
│   │   ├── eval_universal_runner.py        # 通用评估 (Jigsaw Official Metric)
│   │   ├── eval_per_subgroup.py            # 子群细粒度分析
│   │   └── eval_bias_analysis.py           # 偏见分析
│   ├── data/                     # 数据处理
│   │   ├── exp_data_preprocess.py          # 数据预处理
│   │   └── exp_data_loader.py              # Dataset 定义
│   ├── utils/                    # 工具函数
│   └── viz/                      # 可视化
├── src_result/                   # 实验产出
│   ├── models/                   # 模型权重 (17个, 12GB)
│   ├── eval/                     # 评估报告 (JSON + 阈值扫描图)
│   ├── logs/                     # 训练 loss 曲线 (JSON + PNG)
│   └── viz/                      # 可视化图表
├── data/                         # 数据集
├── data_eda/                     # 数据探索分析
├── docs/                         # 论文草稿与方案
└── pretrained_models/            # HuggingFace 预训练权重
```

## 数据集

| 划分 | 样本数 | 说明 |
|------|--------|------|
| 训练集 | 223,219 | 有毒 111,610 / 正常 111,609 (50:50 平衡) |
| 验证集 | 27,902 | 训练时 checkpoint 选择 |
| 测试集 | 27,903 | 最终评估 (独立于训练过程) |

来源：Jigsaw Unintended Bias in Toxicity Classification (Kaggle)

## 评估指标体系

### Jigsaw Official Final Metric (Borkan et al., 2019)

```
Final = 0.25 × Overall_AUC + 0.75 × BiasScore
BiasScore = (PM₋₅(Subgroup_AUCs) + PM₋₅(BPSN_AUCs) + PM₋₅(BNSP_AUCs)) / 3
```

- **PM₋₅**: Power Mean (p=-5)，强调最差子群表现
- **Subgroup AUC**: 仅在提及特定身份群体的样本上计算 AUC
- **BPSN AUC**: Background Positive + Subgroup Negative 混合集上的 AUC
- **BNSP AUC**: Background Negative + Subgroup Positive 混合集上的 AUC

评估覆盖 9 个身份子群：male, female, black, white, muslim, jewish, christian, homosexual_gay_or_lesbian, psychiatric_or_mental_illness

---

## 实验结果

> 所有模型使用相同训练集 (223,219 条)、验证集 (27,902 条)、测试集 (27,903 条)。
> 评估统一使用 `eval_universal_runner.py`，指标完全可比。

### 一、主要结果对比

| 模型 | Final | BiasScore | AUC | F1 | PM(Sub) | PM(BPSN) | PM(BNSP) | Worst Bias |
|------|:-----:|:---------:|:---:|:--:|:-------:|:--------:|:--------:|:----------:|
| **Vanilla DeBERTa-V3** (best seed) | **0.9365** | 0.9247 | 0.9718 | 0.9166 | 0.8968 | 0.9742 | 0.9029 | 0.8581 |
| **Fair S2 V3** (MTL+CLP) | 0.9358 | 0.9237 | 0.9721 | 0.9160 | 0.8951 | 0.9746 | 0.9016 | 0.8571 |
| **Fair S2 V2** (MTL+CLP) | 0.9358 | 0.9238 | 0.9717 | 0.9157 | 0.8948 | 0.9735 | 0.9029 | 0.8604 |
| MTL S2 NoFocal | 0.9352 | 0.9233 | 0.9706 | 0.9150 | 0.8962 | 0.9739 | 0.9000 | 0.8609 |
| Vanilla DeBERTa-V3 (Seed2024) | 0.9354 | 0.9234 | 0.9714 | 0.9155 | 0.8945 | 0.9732 | 0.9027 | 0.8534 |
| Vanilla DeBERTa-V3 (Seed123) | 0.9351 | 0.9233 | 0.9705 | 0.9156 | 0.8955 | 0.9725 | 0.9018 | 0.8575 |
| MTL S2 (Seed123) | 0.9344 | 0.9227 | 0.9697 | 0.9144 | 0.8949 | 0.9712 | 0.9019 | 0.8575 |
| Vanilla RoBERTa | 0.9343 | 0.9221 | 0.9710 | 0.9140 | 0.8943 | 0.9731 | 0.8988 | 0.8602 |
| MTL S2 NoPooling | 0.9338 | 0.9217 | 0.9699 | 0.9140 | 0.8929 | 0.9725 | 0.8998 | 0.8561 |
| BERT+CNN+BiLSTM | 0.9334 | 0.9212 | 0.9700 | 0.9130 | 0.8933 | 0.9696 | 0.9006 | 0.8502 |
| Vanilla BERT | 0.9325 | 0.9202 | 0.9693 | 0.9139 | 0.8919 | 0.9681 | 0.9007 | 0.8459 |
| MTL S2 (Seed42) | 0.9321 | 0.9197 | 0.9692 | 0.9128 | 0.8909 | 0.9712 | 0.8971 | 0.8531 |
| MTL S2 NoReweight | 0.9321 | 0.9197 | 0.9692 | 0.9141 | 0.8906 | 0.9689 | 0.8996 | 0.8451 |
| MTL S2 (Seed2024) | 0.9317 | 0.9192 | 0.9691 | 0.9125 | 0.8903 | 0.9716 | 0.8957 | 0.8528 |
| TF-IDF + LR | 0.8315 | 0.7951 | 0.9408 | 0.8787 | 0.7400 | 0.9495 | 0.6957 | 0.5815 |

### 二、多 Seed 统计 (均值 ± 标准差)

| 模型 | Final | AUC | F1 | BiasScore | PM(Sub) | PM(BPSN) | PM(BNSP) |
|------|:-----:|:---:|:--:|:---------:|:-------:|:--------:|:--------:|
| **Vanilla DeBERTa-V3** (×3) | **0.9357 ± 0.0006** | 0.9712 ± 0.0006 | 0.9159 ± 0.0005 | 0.9238 ± 0.0006 | 0.8956 ± 0.0010 | 0.9733 ± 0.0007 | 0.9025 ± 0.0005 |
| MTL S2 (×3) | 0.9327 ± 0.0012 | 0.9693 ± 0.0003 | 0.9133 ± 0.0008 | 0.9205 ± 0.0015 | 0.8920 ± 0.0020 | 0.9713 ± 0.0002 | 0.8982 ± 0.0026 |

### 三、消融实验 (基于 MTL S2 Seed42)

| 配置 | Final | AUC | BiasScore | PM(Sub) | PM(BPSN) | PM(BNSP) |
|------|:-----:|:---:|:---------:|:-------:|:--------:|:--------:|
| MTL S2 Full (完整模型) | 0.9321 | 0.9692 | 0.9197 | 0.8909 | 0.9712 | 0.8971 |
| − Focal Loss (用 BCE) | 0.9352 (+0.0031) | 0.9706 | 0.9233 | 0.8962 | 0.9739 | 0.9000 |
| − Attention Pooling | 0.9338 (+0.0017) | 0.9699 | 0.9217 | 0.8929 | 0.9725 | 0.8998 |
| − Identity Reweighting | 0.9321 (+0.0000) | 0.9692 | 0.9197 | 0.8906 | 0.9689 | 0.8996 |

### 四、关键发现

1. **Vanilla DeBERTa-V3 ≥ MTL S2**：简单微调的 DeBERTa-V3 (Final=0.9357±0.0006) 显著优于多任务学习版本 (Final=0.9327±0.0012)，差异 Δ=0.0030 超过两者各自的标准差。
2. **Fair S2 与 Vanilla 持平**：公平性优化 (CLP) 未能突破 Vanilla DeBERTa 的上限，Fair S2 V2/V3 的 Final=0.9358 与 Vanilla best=0.9365 在统计上无显著差异。
3. **PM(BNSP) 是核心瓶颈**：所有模型的 PM(BNSP) 均在 0.895-0.903 范围，远低于 PM(BPSN) 的 0.968-0.975，是限制 Final 提升的主要因素。
4. **Focal Loss 反而有害**：去掉 Focal Loss (用 BCE) 后 Final 提升 +0.0031，是消融中最大的正向影响。
5. **模型间方差小**：多 seed 实验标准差 < 0.002，说明结果稳定可复现。

---

## 模型权重清单

| 模型 | 文件名 | 用途 |
|------|--------|------|
| MTL S1 ×3 seeds | `DebertaV3MTL_S1_Seed{42,123,2024}_*` | S2 的预训练权重 (中间产物) |
| MTL S2 ×3 seeds | `DebertaV3MTL_S2_Seed{42,123,2024}_*` | 主模型多 seed |
| MTL S2 消融 ×3 | `DebertaV3MTL_S2_No{Focal,Pooling,Reweight}_*` | 消融实验 |
| Fair S2 V2/V3 | `DebertaV3Fair_S2_Fair_*_{0305_1333,0305_1905}` | 公平性优化 |
| Vanilla DeBERTa ×3 | `VanillaDeBERTa_Seed{42,123,2024}_*` | Baseline (当前最佳) |
| Vanilla BERT | `VanillaBERT_*` | Baseline |
| Vanilla RoBERTa | `VanillaRoBERTa_*` | Baseline |
| BERT+CNN+BiLSTM | `BertCNNBiLSTM_*` | Baseline |

## 环境

- Python 3.8 + PyTorch + Transformers
- DDP 多卡训练 (5× RTX 3090 24GB)
- 预训练模型：`microsoft/deberta-v3-base`

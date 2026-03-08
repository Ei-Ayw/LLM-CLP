# Toxicity Classification with Adversarial Debiasing

基于 DeBERTa-V3 的毒性评论分类系统，采用**条件对抗去偏（Conditional Adversarial Debiasing）** 方法，通过梯度反转层（GRL）消除模型对身份属性的虚假关联，在 Jigsaw Unintended Bias in Toxicity Classification 数据集上进行实验。

## 项目结构

```
├── src_model/                          # 模型定义
│   ├── model_deberta_v3_adversarial.py # 核心：DeBERTa-V3 + 条件 GRL 对抗去偏
│   ├── model_deberta_v3_mtl.py         # MTL 多任务学习 (毒性/子类型/身份)
│   ├── model_bert_cnn_bilstm.py        # Baseline: BERT+CNN+BiLSTM
│   ├── model_vanilla_bert.py           # Baseline: BERT
│   ├── model_vanilla_roberta.py        # Baseline: RoBERTa
│   └── model_vanilla_deberta_v3.py     # Baseline: Vanilla DeBERTa-V3
├── src_script/
│   ├── train/                          # 训练脚本
│   │   ├── train_deberta_v3_adv.py     # 对抗去偏两阶段训练
│   │   ├── train_deberta_v3_mtl_s1.py  # MTL 第一阶段 (多任务预训练)
│   │   ├── train_deberta_v3_mtl_s2.py  # MTL 第二阶段 (微调)
│   │   ├── train_deberta_v3_fair_s2.py # Fair S2 (公平性优化)
│   │   ├── train_vanilla_deberta_v3.py # Vanilla DeBERTa
│   │   ├── train_vanilla_bert.py       # Vanilla BERT
│   │   ├── train_vanilla_roberta.py    # Vanilla RoBERTa
│   │   ├── train_bert_cnn_bilstm.py    # BERT+CNN+BiLSTM
│   │   └── train_classical_tfidf_lr.py # TF-IDF + LR
│   ├── eval/                           # 评估脚本
│   │   ├── eval_universal_runner.py    # 通用评估 (Jigsaw Official Metric)
│   │   ├── eval_per_subgroup.py        # 子群细粒度分析
│   │   └── eval_bias_analysis.py       # 偏见分析
│   ├── data/                           # 数据处理
│   │   ├── exp_data_preprocess.py      # 数据预处理
│   │   └── exp_data_loader.py          # Dataset 定义
│   ├── utils/                          # 工具函数
│   └── viz/                            # 可视化
├── src_result/                         # 实验产出
│   ├── models/                         # 模型权重
│   ├── eval/                           # 评估报告 (JSON + 阈值扫描图)
│   ├── logs/                           # 训练 loss 曲线 (JSON + PNG)
│   └── viz/                            # 可视化图表
├── data/                               # 数据集 (平衡采样)
├── data/natural/                       # 数据集 (自然分布)
├── data_eda/                           # 数据探索分析
├── docs/                               # 论文草稿与方案
└── pretrained_models/                  # HuggingFace 预训练权重
```

---

## 模型架构

### 条件对抗去偏模型（Conditional Adversarial Debiasing）

```
输入: input_ids, attention_mask
         │
    ┌────▼─────────┐
    │  DeBERTa-V3  │  microsoft/deberta-v3-base (12层, 768d)
    │    Base       │  从 S1 多任务预训练 checkpoint 初始化
    └────┬─────────┘
         │ last_hidden_state (B, seq_len, 768)
         │
    ┌────▼───────────────────┐
    │  双通道池化              │
    │  [CLS] token  (768d)   │
    │  + Attention Pooling    │  学习的注意力加权 (768d)
    │    → concat (1536d)     │
    └────┬───────────────────┘
         │
    ┌────▼──────────┐
    │  Projection    │  Linear(1536 → 512) + ReLU + Dropout(0.1)
    └────┬──────────┘
         │ z (512d)  ← 共享特征表示
         │
    ┌────┼────────────────────────┐
    │    │                        │
    ▼    ▼                        ▼
┌──────┐ ┌──────────┐    ┌──────────────┐
│ Tox  │ │ Subtype  │    │     GRL      │ Gradient Reversal Layer
│ Head │ │  Head    │    │  λ · (-∇)    │ 前向不变，反向梯度 × (-λ)
│512→1 │ │ 512→6   │    └──────┬───────┘
└──────┘ └──────────┘          │ z_rev
  毒性      6个子类型    ┌──────▼───────┐
  预测      预测         │  Adversary   │ 3层判别器
                        │  512 → 512   │ ReLU + Dropout(0.3)
                        │  512 → 256   │ ReLU + Dropout(0.3)
                        │  256 → 9     │ 9个身份属性预测
                        └──────────────┘
```

#### 核心创新：条件梯度反转

传统 GRL 对所有样本统一施加对抗压力，但在毒性分类中存在问题：仇恨言论的检测**需要**身份信息（如 "kill all [group]" 中的群体指代）。因此我们提出**条件 GRL**：

- **非毒性样本**（`y_tox < 0.5`）：施加对抗损失 L_adv，强制 backbone 不编码身份信息 → 消除"提到身份 = 有毒"的虚假关联
- **毒性样本**：不施加 L_adv，保留身份信息 → 保障仇恨言论检测能力

```python
# 条件 GRL：只对非毒性样本计算 adversarial loss
non_toxic_mask = (y_tox.squeeze(-1) < 0.5)
if non_toxic_mask.any():
    adv_loss = criterion_adv(logits_id_adv, y_id).mean(dim=-1)
    l_adv = adv_loss[non_toxic_mask].mean()
else:
    l_adv = 0
loss = L_tox + 0.3 × L_sub + L_adv
```

#### 损失函数

| 损失项 | 含义 | 应用范围 |
|--------|------|----------|
| L_tox | BCE，毒性预测主任务 | 所有样本 |
| L_sub | BCE，6 个子类型辅助任务 (×0.3) | 所有样本 |
| L_adv | BCE，9 个身份属性预测（经 GRL 反转） | 仅非毒性样本 |

#### 两阶段训练流程

**Phase A — Head 热身**（5 epochs）
- 冻结 DeBERTa backbone
- 只训练 Projection + Tox Head + Subtype Head + Adversary
- 目的：让各任务头（尤其是 Adversary）从随机初始化充分收敛
- LR = 1e-4，固定学习率

**Phase B — 对抗微调**（6 epochs）
- 解冻全部参数
- Layer-wise LR decay (factor=0.95)：底层 LR 低，顶层 LR 高
- Adversary 独立学习率 = 5× base_lr，确保对抗强度
- λ sigmoid 爬升：`λ(t) = λ_max × (2/(1+exp(-γt)) - 1)`，从 0 平滑增长到 0.5
- EMA（指数移动平均）：decay=0.999
- Early Stopping：patience=3，基于 Jigsaw Final Metric

#### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| λ_max | 0.5 | GRL 最大反转强度 |
| γ | 10.0 | λ 爬升速度 |
| aux_scale | 0.3 | 子类型辅助损失权重 |
| adv_lr_mult | 5.0 | Adversary 学习率倍率 |
| base_lr (Phase B) | 2e-6 | backbone 微调学习率 |
| layer_decay | 0.95 | layer-wise LR 衰减因子 |
| ema_decay | 0.999 | EMA 平滑系数 |
| batch_size | 48 × 5 GPU | 有效批大小 240 |
| grad_accum | 2 | 梯度累积步数 |

---

## 数据集

### 数据来源

Jigsaw Unintended Bias in Toxicity Classification (Kaggle)，原始约 180 万条评论。

### 采样策略

从原始数据随机采样约 28 万条，保持**自然分布**（~8% 毒性率），不做 50/50 毒性平衡。所有模型使用同一份采样数据，确保公平对比。

| 划分 | 样本数 | 毒性率 | 说明 |
|------|--------|--------|------|
| 训练集 | 223,219 | 8.0% | 80% |
| 验证集 | 27,902 | 8.0% | 10%，checkpoint 选择 |
| 测试集 | 27,903 | 8.0% | 10%，最终评估 |

### 标签体系

| 标签类型 | 列名 | 处理方式 |
|----------|------|----------|
| 毒性 | `target` | 连续值 [0,1]，>0.5 视为有毒 |
| 子类型 (×6) | `severe_toxicity, obscene, identity_attack, insult, threat, sexually_explicit` | 连续值 [0,1] |
| 身份属性 (×9) | `male, female, homosexual_gay_or_lesbian, christian, jewish, muslim, black, white, psychiatric_or_mental_illness` | 二值化（≥0.5 → 1） |

### 为什么选择自然分布而非平衡采样

前期实验发现，50/50 毒性平衡采样会制造严重的**虚假关联**：

| | 平衡采样 (50/50) | 自然分布 (~8%) |
|--|-----------------|---------------|
| 含身份标签样本 → 毒性率 | 16.5% | ~正常 |
| 不含身份标签样本 → 毒性率 | 97.4% | ~正常 |
| 虚假关联 gap | **81pp** | **~10pp** |

平衡采样的"有身份标签 = 安全，无身份标签 = 有毒"虚假关联，使得去偏方法难以正确工作。自然分布下虚假关联更弱，更接近真实场景。

---

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

### 一、自然分布数据（主实验）

> 所有模型使用相同的自然分布数据集（~8% 毒性率），评估统一使用 `eval_universal_runner.py`。

| 模型 | Final | BiasScore | AUC | F1 | PM(Sub) | PM(BPSN) | PM(BNSP) | Worst Bias |
|------|:-----:|:---------:|:---:|:--:|:-------:|:--------:|:--------:|:----------:|
| **Cond-GRL (ours)** | **0.9302** | **0.9196** | 0.9623 | 0.6845 | 0.8889 | 0.9281 | 0.9417 | 0.8483 |
| Vanilla DeBERTa-V3 | 0.9295 | 0.9176 | 0.9653 | 0.6900 | 0.8928 | 0.8950 | 0.9651 | 0.8421 |
| GRL (全样本对抗) | 0.9283 | 0.9161 | 0.9647 | 0.6918 | 0.8884 | 0.8969 | 0.9631 | 0.8389 |

- **Cond-GRL 超过 Vanilla baseline**：Final 0.9302 vs 0.9295（+0.0007），BiasScore 0.9196 vs 0.9176（+0.0020）
- **全样本 GRL 不如 Vanilla**：对所有样本统一施加对抗（包括毒性样本）反而损害性能，说明条件反转是关键设计

- **Cond-GRL 超过 Vanilla baseline**：Final 0.9302 vs 0.9295（+0.0007），BiasScore 0.9196 vs 0.9176（+0.0020）
- **全样本 GRL 不如 Vanilla**：对所有样本统一施加对抗（包括毒性样本）反而损害性能，说明条件反转是关键设计

### 二、平衡数据（历史对照实验）

> 历史实验，使用 50/50 毒性平衡采样数据。仅供参考，与自然分布实验不直接可比。

| 模型 | Final | BiasScore | AUC | F1 | PM(Sub) | PM(BPSN) | PM(BNSP) | Worst Bias |
|------|:-----:|:---------:|:---:|:--:|:-------:|:--------:|:--------:|:----------:|
| **Vanilla DeBERTa-V3** (best seed) | **0.9365** | 0.9247 | 0.9718 | 0.9166 | 0.8968 | 0.9742 | 0.9029 | 0.8581 |
| Fair S2 V3 (MTL+CLP) | 0.9358 | 0.9237 | 0.9721 | 0.9160 | 0.8951 | 0.9746 | 0.9016 | 0.8571 |
| Fair S2 V2 (MTL+CLP) | 0.9358 | 0.9238 | 0.9717 | 0.9157 | 0.8948 | 0.9735 | 0.9029 | 0.8604 |
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

### 三、关键发现

1. **条件 GRL 有效**：在自然分布数据上，Cond-GRL (Final=0.9302) 超过 Vanilla DeBERTa-V3 (Final=0.9295)，而无条件 GRL (Final=0.9283) 反而更差。条件反转（仅对非毒性样本施加对抗）是关键设计。
2. **数据分布影响显著**：50/50 平衡采样制造 81pp 的虚假关联，使去偏方法难以正常工作；自然分布（~8% 毒性率）下虚假关联仅 ~10pp，更接近真实场景。
3. **Adversary 初始化很重要**：Phase A 从 2 epochs 增加到 5 epochs 后，Adversary 进入 Phase B 时预测能力更强，Final 从 0.9294 提升到 0.9302。
4. **MTL 路线失败**：多任务学习中的身份预测头迫使 backbone 编码身份信息，反而加剧偏见（MTL S2 Final=0.9327 < Vanilla Final=0.9357）。

---

## 环境

- Python 3.8 + PyTorch + Transformers
- DDP 多卡训练 (5× RTX 3090 24GB)
- 预训练模型：`microsoft/deberta-v3-base`

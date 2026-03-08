# HCMA: Hierarchical Conditional Metric-Aligned Debiasing
## 面向毒性评论分类的分层条件对抗去偏框架

---

## 1. 研究动机

### 1.1 问题定义

在线内容审核系统中，基于预训练语言模型的毒性分类器存在系统性偏见：当评论中提及特定身份群体（如宗教、种族、性别等）时，模型倾向于给出更高的毒性预测分数，即使评论本身并不含有毒性内容。这种**虚假关联（spurious correlation）** 会导致涉及少数群体的正常讨论被错误标记为有毒内容，造成系统性的言论压制。

### 1.2 现有方法的局限

**多任务学习（MTL）路线的失败**：直接在共享 backbone 上添加身份预测头，反而迫使模型更强地编码身份信息到公共表示中，加剧了捷径学习（shortcut learning），实验显示 MTL 方法 Final Metric 比 Vanilla baseline 低 0.003（0.9327 vs 0.9357）。

**简单对抗去偏的不足**：传统 GRL（Gradient Reversal Layer）对所有样本统一施加对抗压力，忽略了一个关键事实：对于真实的仇恨言论（如 "kill all [group]"），身份信息是检测毒性所**必需**的语义特征。无条件对抗会同时抹除有用信号和有害捷径。

**训练-评测目标错配**：现有方法普遍使用 BCE 作为训练目标，而 Jigsaw 官方评测指标（Final Metric）主要由 AUC 和 worst-group bias score 驱动，两者的优化方向并不完全一致。

### 1.3 核心洞察

我们提出三个关键洞察：

1. **身份信息应分层处理**：模型需要知道"文本在谈论某个身份群体"（语义理解），但不应该因为"具体是哪个群体"而改变毒性预测（公平性）。
2. **去偏力度应与毒性程度负相关**：对于非毒性的身份提及样本，应强力去偏；对于真实的仇恨言论，应保留身份信息。
3. **训练目标应直接对齐评测指标**：通过可微分的 AUC surrogate 直接优化 Jigsaw Final Metric。

---

## 2. 方法

### 2.1 整体架构

```
输入: comment_text
         │
    ┌────▼─────────────┐
    │   DeBERTa-V3     │  Backbone (12 layers, 768d)
    │     Base         │  从多任务预训练 S1 初始化
    └────┬─────────────┘
         │ last_hidden_state (B, L, 768)
         │
    ┌────▼─────────────────────┐
    │   双通道池化               │
    │   [CLS] token    (768d)  │
    │   + Attention Pooling    │  可学习注意力加权 (768d)
    │     → concat     (1536d) │
    └────┬─────────────────────┘
         │
    ┌────▼─────────────┐
    │   Projection     │  Linear(1536→512) + GELU + Dropout(0.2)
    └────┬─────────────┘
         │ z ∈ ℝ^512  (共享特征表示)
         │
    ┌────┼──────────┬────────────────┬──────────────────────┐
    │    │          │                │                      │
    ▼    ▼          ▼                ▼                      ▼
  ┌────┐ ┌────┐  ┌──────┐    ┌──────────┐          ┌──────────────┐
  │Tox │ │Sub │  │Exist │    │  Coarse  │          │     GRL      │
  │Head│ │Head│  │Head  │    │   Head   │          │  λ·(-∇)      │
  │1   │ │6   │  │1     │    │   5      │          └──────┬───────┘
  └────┘ └────┘  └──────┘    └──────────┘                 │
  主任务  辅助      Level 1     Level 2              ┌──────▼───────┐
  毒性   子类型    身份存在性   粗粒度类别            │  Adversary   │
  预测   预测     (保留)      (保留)                │  Specific    │
                                                   │ 512→512→256→9│
                                                   └──────────────┘
                                                     Level 3
                                                   具体身份 (对抗压制)
```

### 2.2 分层条件对抗（Hierarchical Conditional Adversary）

**核心创新**：将身份信息拆分为三个语义层级，仅对最容易形成数据集捷径的"具体身份"层级施加对抗压制。

#### Level 1 — 身份存在性（Existence）
- **输出**：Binary（文本是否提及任何身份群体）
- **训练方式**：正常梯度（不经过 GRL），BCE loss
- **目的**：保留"文本涉及身份话题"的语义理解能力

#### Level 2 — 粗粒度类别（Coarse Group）
- **输出**：5 维 multi-label（性别、性取向、宗教、种族、残障）
- **训练方式**：正常梯度（不经过 GRL），BCE loss
- **目的**：保留对身份讨论场景的粗粒度理解

**粗粒度分组映射**：

| Coarse Group | Specific Identities |
|--------------|-------------------|
| Gender | male, female |
| Sexual Orientation | homosexual_gay_or_lesbian |
| Religion | christian, jewish, muslim |
| Race | black, white |
| Disability | psychiatric_or_mental_illness |

#### Level 3 — 具体身份（Specific Identity）— 对抗压制
- **输出**：9 维 multi-label（9 个具体身份属性）
- **训练方式**：经过 GRL 梯度反转
- **网络结构**：3 层判别器 512→512→256→9（ReLU + Dropout(0.3)）
- **目的**：迫使 backbone 无法区分"是 muslim 还是 christian"、"是 black 还是 white"，从而消除具体身份与毒性预测之间的虚假关联

**设计依据**：数据集中的偏见捷径主要存在于 Level 3。例如，训练数据中 muslim 相关评论的毒性标注率可能高于 christian，导致模型学到"muslim → toxic"的虚假关联。通过 Level 3 的对抗压制，模型只能知道"这是一条关于宗教的评论"（Level 2），但无法区分具体是哪个宗教，从而在宗教类别内部实现公平。

### 2.3 Soft Gate（软门控）

替代之前的 hard threshold（`y_tox < 0.5`），使用基于连续标签的 sigmoid 软门控：

$$g_i = \sigma\left(\frac{\tau_{\text{tox}} - y_{\text{tox},i}}{T_{\text{tox}}}\right) \cdot \sigma\left(\frac{\tau_{\text{ia}} - y_{\text{ia},i}}{T_{\text{ia}}}\right)$$

其中：
- $y_{\text{tox},i}$ 为 toxicity 连续标注值（crowd fraction）
- $y_{\text{ia},i}$ 为 identity_attack 连续标注值
- $\tau_{\text{tox}} = 0.30$，$T_{\text{tox}} = 0.08$
- $\tau_{\text{ia}} = 0.20$，$T_{\text{ia}} = 0.05$

**物理含义**：
- 当样本明显无毒且非身份攻击时，$g_i \approx 1$，对抗压力全开
- 当样本有毒或含身份攻击时，$g_i \approx 0$，保留身份信息
- 边界样本（toxicity ∈ [0.2, 0.4]）获得平滑过渡的中间权重

**优势**：避免 hard threshold 在边界处的信息丢失，保留 crowd annotator 的标注分歧信息。

### 2.4 Metric-Aligned AUC Surrogate Loss

训练目标直接对齐 Jigsaw 官方 Final Metric：

$$\text{Final} = 0.25 \times \text{Overall\_AUC} + 0.75 \times \text{BiasScore}$$

实现方式：

**Overall AUC Surrogate**：采用 pairwise logistic loss：
$$\mathcal{L}_{\text{AUC}} = \frac{1}{|P|} \sum_{(i,j) \in P} \log(1 + e^{-(s_i - s_j)})$$
其中 $P$ 为正负样本对集合，$s_i, s_j$ 分别为正样本和负样本的 logit。每个 batch 随机采样 256 对。

**Slice AUC Surrogate**：对 9 个身份子群分别计算：
- **Subgroup AUC**：仅在子群内部采样正负对
- **BPSN AUC**：从（背景正样本, 子群负样本）采样
- **BNSP AUC**：从（背景负样本, 子群正样本）采样

**Soft Worst-Group 聚合**：用 Log-Sum-Exp 逼近 Power Mean (p=-5)：
$$\mathcal{L}_{\text{bias}} = \frac{1}{\beta} \log \sum_k e^{\beta \cdot \mathcal{L}_k}$$
其中 $\beta = 8.0$，$\mathcal{L}_k$ 为各切片 AUC loss。$\beta$ 越大越接近 worst-group（max），逼近 PM₋₅ 的强调最差子群效果。

**最终 Metric Loss**：
$$\mathcal{L}_{\text{metric}} = 0.25 \times \mathcal{L}_{\text{AUC,overall}} + 0.75 \times \mathcal{L}_{\text{bias}}$$

### 2.5 Slice-Aware Sampler

为保证每个 mini-batch 中有足够的身份提及样本（rare groups 在随机采样下可能完全缺失），设计分层采样策略：

| 池子 | 占比 | 描述 |
|------|-----|------|
| Background | 50% | 无身份提及的非毒性样本（主任务基座） |
| Non-toxic Identity | 25% | 非毒性的身份提及样本（去偏核心样本） |
| Toxic Identity | 15% | 毒性的身份提及样本（仇恨言论检测） |
| Toxic Background | 10% | 无身份提及的毒性样本 |

**DDP 兼容**：每个 GPU rank 使用独立种子的采样器，保证跨 GPU 不重叠。

### 2.6 总损失函数

$$\mathcal{L}_{\text{total}} = (1-w_m) \cdot \mathcal{L}_{\text{soft}} + w_m \cdot \mathcal{L}_{\text{metric}} + \alpha \cdot \mathcal{L}_{\text{sub}} + \gamma_h \cdot (\mathcal{L}_{\text{exist}} + \mathcal{L}_{\text{coarse}}) + \lambda(t) \cdot \mathcal{L}_{\text{adv}}$$

其中：
$$\mathcal{L}_{\text{adv}} = \frac{1}{N} \sum_{i=1}^{N} g_i \cdot \text{BCE}(f_{\text{spec}}(\text{GRL}(z_i)), y_{\text{spec},i})$$

| 损失项 | 权重 | 说明 |
|--------|------|------|
| $\mathcal{L}_{\text{soft}}$ | $1 - w_m = 0.70$ | 加权 BCE 毒性分类损失 |
| $\mathcal{L}_{\text{metric}}$ | $w_m = 0.30$ | AUC surrogate (对齐 Final Metric) |
| $\mathcal{L}_{\text{sub}}$ | $\alpha = 0.15$ | 6 个子类型辅助 BCE |
| $\mathcal{L}_{\text{exist}} + \mathcal{L}_{\text{coarse}}$ | $\gamma_h = 0.05$ | 分层身份存在性 + 粗粒度 (保留) |
| $\mathcal{L}_{\text{adv}}$ | $\lambda(t)$ | Soft-gated 对抗损失 (具体身份压制) |

---

## 3. 训练策略

### 3.1 两阶段训练

#### Phase A — Head 热身（5 epochs）
- **冻结** DeBERTa backbone
- **训练** Projection + 所有任务头 + Adversary
- **目的**：让各任务头（尤其是 3 层 Adversary）从随机初始化充分收敛
- **学习率**：固定 1e-4
- **关键发现**：Adversary 初始化质量直接决定 Phase B 对抗的有效性。Phase A 从 2 epochs 增加到 5 epochs 后，Phase B 的 Final Metric 从 0.9294 提升到 0.9302

#### Phase B — 对抗微调（6 epochs）
- **解冻** 全部参数
- **Layer-wise LR decay**：
  - Embedding 层：$\text{lr} \times 0.95^2$
  - Encoder 底层 (0-5)：$\text{lr} \times 0.95^2$
  - Encoder 顶层 (6-11)：$\text{lr} \times 0.95$
  - 任务头 + Projection：$\text{lr}$
  - Adversary：$\text{lr} \times 3.0$（独立更高学习率）
- **λ Sigmoid Ramp-up**：
  $$\lambda(t) = \lambda_{\max} \cdot \left(\frac{2}{1 + e^{-\gamma t}} - 1\right)$$
  其中 $\lambda_{\max} = 0.3$，$\gamma = 10.0$
- **EMA**：Exponential Moving Average，decay = 0.999
- **Cosine LR Schedule**：warmup ratio = 0.1
- **Early Stopping**：patience = 3，基于 val loss

### 3.2 关键超参数

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| Backbone | DeBERTa-V3-Base (86M) | 迁移成本低，5×3090 显存可控 |
| max_len | 256 | 覆盖 >95% 评论长度 |
| batch_size | 48 × 5 GPU × 2 accum = 480 | 有效批大小 480 |
| Phase B base_lr | 2e-6 | 微调阶段低学习率 |
| λ_max | 0.3 | 比原始 0.5 降低，因引入 L_metric 后信号更多 |
| adv_lr_mult | 3.0 | 比原始 5.0 降低，避免 Adversary 过强搅乱表示 |
| aux_scale | 0.15 | 比原始 0.3 降低，减少子类型对主任务干扰 |
| w_metric | 0.30 | AUC surrogate 占总损失 30% |
| Soft gate τ_tox | 0.30 | toxicity 低于 0.3 时全力去偏 |
| Soft gate T_tox | 0.08 | sigmoid 温度，控制过渡平滑度 |
| AUC pair samples | 256/batch | 计算效率与梯度稳定的平衡 |
| Worst-group β | 8.0 | 逼近 PM₋₅ 的 worst-group 效果 |

---

## 4. 数据策略

### 4.1 数据集

CivilCommentsIdentities 子集：从 Jigsaw Unintended Bias in Toxicity Classification 原始 180 万条评论中，筛选所有经过身份属性标注的样本，共 405,130 条。

| 划分 | 样本数 | 毒性率 | 有身份提及 (≥0.5) |
|------|--------|--------|-------------------|
| Train | 324,104 | 11.4% | 40.4% |
| Val | 40,513 | 11.4% | 40.2% |
| Test | 40,513 | 11.4% | 40.7% |

### 4.2 标签处理

- **毒性标签**（target）：保留原始 crowd fraction 连续值 [0,1] 作为 soft label 用于 BCE 训练，≥0.5 二值化用于 AUC 评估
- **子类型**（6 维）：保留连续值
- **身份属性**（9 维）：保留连续值用于 soft gate，≥0.5 二值化用于身份头训练和子群评估

### 4.3 为什么用身份标注子集而非全量数据

| 比较维度 | 全量 180 万 | 身份标注 40.5 万 |
|----------|-----------|----------------|
| 身份标签覆盖率 | 22.4% 有标注 | 100% 有标注 |
| L_adv 有效比例 | ~22% 样本可计算 | 100% 样本可计算 |
| Subgroup 评估可靠性 | 子群样本少 | 子群样本充足 |
| 训练时间 | ~20h | ~4.5h |

对于以 worst-group bias 为核心评测指标的任务，身份标签的**完整性**比数据**总量**更重要。

---

## 5. 评估指标

### Jigsaw Official Final Metric (Borkan et al., 2019)

$$\text{Final} = 0.25 \times \text{Overall\_AUC} + 0.75 \times \text{BiasScore}$$

$$\text{BiasScore} = \frac{1}{3}\left(\text{PM}_{-5}(\text{Subgroup AUCs}) + \text{PM}_{-5}(\text{BPSN AUCs}) + \text{PM}_{-5}(\text{BNSP AUCs})\right)$$

覆盖 9 个身份子群：male, female, black, white, muslim, jewish, christian, homosexual_gay_or_lesbian, psychiatric_or_mental_illness

---

## 6. 与现有方法的本质区别

| 方法 | 对身份信息的处理 | 门控方式 | 训练目标 |
|------|----------------|---------|---------|
| 传统 GRL (Ganin 2016) | 统一抹除所有身份 | 无 | BCE |
| Conditional GRL (ours v1) | 非毒样本抹除，毒样本保留 | Hard (y<0.5) | BCE |
| MTL + Identity Head | 强化编码身份 | 无 | BCE + Identity BCE |
| CLP (Google) | Counterfactual 一致性 | 无 | BCE + Logit Pairing |
| **HCMA (ours)** | **分层：保留 coarse，压制 specific** | **Soft gate** | **BCE + AUC surrogate** |

HCMA 的核心差异在于：不是"要不要去偏"的二选一，而是**在哪个语义粒度上去偏**。

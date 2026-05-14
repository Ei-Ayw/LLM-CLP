Open-source LLM + Open-set Controlled Counterfactual Rewriting + Semantic Gate + CLP

**日期**：2026-05-14  
**目标会议**：EMNLP  
**当前决策**：基本重做反事实生成相关部分；保留不直接依赖反事实生成的数据、baseline、训练框架和评估框架。

---

## 0. 一句话版本

当前稿件最需要重构的不是 CLP 公式，也不是 DeBERTa 训练框架，而是 **counterfactual generation protocol**。

旧版主方法：

> GLM-4-Flash API + identity keyword detection + 16 fixed bidirectional mappings + prompt-only counterfactual generation + CLP

新版主方法建议改成：

> Open-source local LLM + open-set controlled identity rewriting + semantic/toxicity validity gate + CLP

核心目标是同时解决四个审稿风险：

1. **闭源商业 API 不可复现**：GLM-4-Flash 改成本地开源 LLM，例如 Qwen3 / Qwen2.5。
2. **16 对固定映射限制 LLM 能力**：主方法不再依赖固定 target identity mapping。
3. **反事实样本语义有效性不足**：加入 semantic/toxicity gate，并让 `delta_sem` 真正进入 CLP。
4. **评估协议不公平风险**：新 test counterfactuals 一旦生成，所有方法都必须在同一套新 CF 上重算 CFR/CTFG。

---

## 1. 当前稿件的核心风险诊断

### 1.1 闭源 API 复现风险

当前论文主方法使用 GLM-4-Flash API 生成反事实样本。这个问题很容易被顶会审稿人攻击：

- API 模型可能版本漂移；
- 生成结果无法完全复现；
- 别人无法在无 API 权限时复现；
- 论文主贡献依赖闭源黑盒能力。

**处理方案**：

将主方法生成器改为本地开源模型，例如：

- `Qwen3-14B` via Ollama / vLLM；
- `Qwen2.5-14B-Instruct` via vLLM / HuggingFace；
- 显存不足时可用 `Qwen2.5-7B-Instruct` 或量化版本。

论文中必须明确写：

- model name；
- model version / checkpoint；
- inference backend；
- quantization setting；
- temperature；
- top-p；
- max tokens；
- prompt version；
- random seed；
- generated data release plan。

---

### 1.2 16 fixed bidirectional mappings 风险

当前流程是：

1. 检测 identity-related keywords；
2. 从 16 对 bidirectional swap pairs 中选择 target；
3. 让 LLM 改写。

这个设计会让审稿人质疑：

> 你的 LLM-generated counterfactuals 到底是 LLM 生成，还是 lexical swap + LLM polish？

**处理方案**：

不要把 16 对固定映射作为主方法。改成：

> LLM first detects explicit or implicit identity references, then proposes a contextually plausible alternative identity, and finally performs a minimal identity-only rewrite.

关键区别：

| 方法 | 谁选择 target identity | 是否允许大幅改写句子 | 建议定位 |
|---|---|---:|---|
| Fixed-16 | 人工固定映射表 | 否 | 旧方法 / ablation |
| Open-set controlled | LLM 选择上下文合理 identity | 否，只允许 minimal identity edit | 新主方法 |
| Free-form | LLM 自由选择并自由改写 | 可能 | 诊断实验，不建议做主方法 |

最重要的一点：

> 新方法不是 free-form，而是 **open-set target identity selection + controlled minimal rewrite**。

也就是：

> 身份选择开放，句子改写受控。

---

### 1.3 语义门控缺失风险

当前论文已经承认：CLP 只有在 `x'` 是 label-preserving identity edit 时才合理。如果 LLM paraphrase、generalize 或改变毒性强度，CLP 会变成 noisy agreement constraint。

旧稿中 semantic gate 只是 future work，这对顶会很危险。

**处理方案**：

实现真正的 semantic/toxicity gate，并在 CLP 中使用：

\[
\delta_j^{final} = \delta_j^{gen} \cdot \delta_j^{sem}
\]

\[
\mathcal{L}_{CLP}
=
\frac{1}{|S|}
\sum_{j \in S}
\|z_j - z'_j\|_2^2,
\quad
S = \{j: \delta_j^{gen}=1, \delta_j^{sem}=1\}
\]

只有通过 gate 的 pair 才进入 CLP。

---

### 1.4 DynaHate 风险

当前结果中 DynaHate 上 Swap+CLP 的 CFR 可能优于 LLM+CLP。这个不是小问题，因为它会削弱“LLM counterfactuals 一定更好”的叙事。

**处理方案**：

不要再宣称 LLM-CLP 在所有情况下绝对优于 swap。改成更稳的主张：

> LLM-CLP improves counterfactual stability under a validated generated-pair protocol, especially when identity references are explicit or recoverable by the LLM. On adversarial or implicit datasets such as DynaHate, gains depend on coverage and pair validity.

如果新方法在 DynaHate 仍然没有显著领先，可以解释为：

- DynaHate 更多隐性/对抗式表达；
- identity-bearing span 不总是显式存在；
- open-set detection 能提升 coverage，但 CLP 的收益仍受有效 pair 数量限制；
- 这说明方法是 targeted intervention，而不是 universal fairness solution。

---

## 2. 新版主方法设计

## 2.1 方法名建议

论文主方法仍可叫 **LLM-CLP**，但展开方式建议改成：

> LLM-CLP: Open-source LLM-guided Controlled Counterfactual Rewriting with Counterfactual Logit Pairing

或者：

> LLM-CLP combines open-set controlled identity rewriting with counterfactual logit pairing.

不要再把 GLM-4-Flash 写进 abstract 或主方法名。

---

## 2.2 新 pipeline

新版 pipeline 建议分为四个阶段：

1. **Identity reference detection**
2. **Open-set target identity planning**
3. **Minimal identity-only rewriting**
4. **Semantic/toxicity validity gate**

---

### Stage A：Identity reference detection

目标：识别句子中显式或隐式的 demographic / protected-group reference。

不仅检测显式身份词，例如：

- Muslim
- Christian
- women
- gay people
- immigrants
- Black people

也要允许检测隐性身份引用，例如：

- “those people from the Middle East”
- “people crossing the border”
- “welfare queens”
- “inner city people”

输出 schema：

```json
{
  "has_identity_reference": true,
  "source_identity": "Middle Eastern immigrants",
  "identity_type": "ethnicity/national-origin/migration-status",
  "identity_span": "Middle Eastern immigrants",
  "implicit": false,
  "reason": "The phrase refers to a demographic group."
}
```

如果无法识别身份，输出：

```json
{
  "has_identity_reference": false,
  "source_identity": "",
  "identity_type": "",
  "identity_span": "",
  "implicit": false,
  "reason": "No demographic or protected-group reference is present."
}
```

---

### Stage B：Open-set target identity planning

目标：由 LLM 选择一个上下文合理的替代身份，而不是从 16 对固定映射里选。

约束：

1. target identity 应尽量与 source identity 属于同一 broad demographic dimension；
2. target identity 必须适合原句语境；
3. target identity 不应引入新的社会含义或额外毒性；
4. 如果无法选择合理替代项，则标记 invalid。

例子：

原句：

> Those Middle Eastern immigrants are stealing our jobs.

不推荐：

> Those Christian immigrants are stealing our jobs.

原因：Middle Eastern 是地区/族裔/国籍相关，而 Christian 是宗教，维度不一致。

推荐：

> Those Eastern European immigrants are stealing our jobs.

原因：替换后仍保持“移民群体 + 抢工作”的上下文结构。

---

### Stage C：Minimal identity-only rewriting

目标：只改变 identity-bearing span，不允许大幅 paraphrase。

要求：

- preserve toxicity level；
- preserve sentiment；
- preserve syntax；
- preserve topic；
- preserve speaker intent；
- preserve all non-identity content；
- do not add/remove unrelated insults；
- do not change label-relevant toxic content。

输出 schema：

```json
{
  "valid_generation": true,
  "source_identity": "Middle Eastern immigrants",
  "target_identity": "Eastern European immigrants",
  "changed_span_original": "Middle Eastern immigrants",
  "changed_span_counterfactual": "Eastern European immigrants",
  "counterfactual": "Those Eastern European immigrants are stealing our jobs.",
  "edit_type": "minimal_identity_substitution",
  "reason": "Only the demographic span was changed."
}
```

如果无法生成：

```json
{
  "valid_generation": false,
  "source_identity": "Middle Eastern immigrants",
  "target_identity": "",
  "changed_span_original": "",
  "changed_span_counterfactual": "",
  "counterfactual": "",
  "edit_type": "none",
  "reason": "No contextually plausible label-preserving identity substitution was found."
}
```

---

### Stage D：Semantic / toxicity validity gate

目标：过滤掉不适合进入 CLP 的 pair。

Gate 应至少包含三层：

1. **规则过滤**
2. **自动指标过滤**
3. **LLM-as-judge 验证**

建议最终生成字段：

```json
{
  "delta_gen": 1,
  "delta_sem": 1,
  "judge_valid": true,
  "semantic_similarity": 0.93,
  "toxicity_drift": 0.04,
  "normalized_edit_distance": 0.18,
  "identity_change_success": true
}
```

只有 `delta_gen = 1` 且 `delta_sem = 1` 的样本进入 CLP。

---

## 3. 代码修改方案

## 3.1 文件级修改清单

| 文件 | 修改内容 |
|---|---|
| `src_script/counterfactual/cf_generator_llm.py` | 支持 Ollama / vLLM / HuggingFace 后端；新增 open-set controlled generation |
| `src_script/counterfactual/cf_validator.py` | 新增 semantic/toxicity gate、LLM-as-judge、toxicity drift filter |
| `src_script/counterfactual/prompts.py` | 统一维护 detection / planning / rewrite / judge prompts |
| `src_script/counterfactual/schema.py` | 定义 JSON schema、解析器、失败 fallback |
| `src_script/train/train_causal_fair.py` | 读取 `delta_sem`；CLP 只作用于通过 gate 的 pair |
| `src_script/eval/eval_causal_fairness.py` | 所有模型统一在新 test CF 上评估 CFR/CTFG |
| `src_script/eval/eval_cf_quality.py` | 新增 coverage、acceptance、edit distance、semantic similarity、tox drift、identity diversity |
| `generate_emnlp_report.py` | 更新论文表格 |
| `EXPERIMENT_REPORT.md` | 记录所有实验配置、模型、prompt hash、数据 hash、随机种子 |

---

## 3.2 命令行参数建议

生成 train CF：

```bash
python src_script/counterfactual/cf_generator_llm.py \
  --dataset dynahate \
  --split train \
  --backend ollama \
  --model qwen3:14b \
  --mode open_set_controlled \
  --temperature 0.2 \
  --top_p 0.9 \
  --max_tokens 256 \
  --num_workers 4 \
  --output data/cf/qwen3_open_set/dynahate_train.parquet \
  --cache_dir cache/qwen3_open_set
```

生成 test CF：

```bash
python src_script/counterfactual/cf_generator_llm.py \
  --dataset dynahate \
  --split test \
  --backend ollama \
  --model qwen3:14b \
  --mode open_set_controlled \
  --temperature 0.2 \
  --top_p 0.9 \
  --max_tokens 256 \
  --num_workers 4 \
  --output data/cf/qwen3_open_set/dynahate_test.parquet \
  --cache_dir cache/qwen3_open_set
```

运行 validator：

```bash
python src_script/counterfactual/cf_validator.py \
  --input data/cf/qwen3_open_set/dynahate_train.parquet \
  --output data/cf/qwen3_open_set/dynahate_train_validated.parquet \
  --judge_backend ollama \
  --judge_model qwen3:14b \
  --semantic_threshold 0.85 \
  --toxicity_drift_threshold 0.10 \
  --edit_distance_threshold 0.35
```

训练主方法：

```bash
python src_script/train/train_causal_fair.py \
  --dataset dynahate \
  --cf_file data/cf/qwen3_open_set/dynahate_train_validated.parquet \
  --use_semantic_gate true \
  --lambda_clp 1.0 \
  --seed 42
```

统一评估：

```bash
python src_script/eval/eval_causal_fairness.py \
  --dataset dynahate \
  --test_cf_file data/cf/qwen3_open_set/dynahate_test_validated.parquet \
  --model_dir checkpoints/dynahate/llm_clp_qwen3_openset_gate_seed42 \
  --metrics acc macro_f1 auc cfr ctfg fped fned
```

---

## 3.3 数据 schema 建议

最终 parquet/jsonl 中每条样本建议包含：

```json
{
  "id": "...",
  "dataset": "dynahate",
  "split": "train",
  "text": "...",
  "label": 1,

  "has_identity_reference": true,
  "source_identity": "...",
  "source_identity_type": "...",
  "source_span": "...",
  "implicit_identity": false,

  "target_identity": "...",
  "target_identity_type": "...",
  "counterfactual": "...",
  "changed_span_original": "...",
  "changed_span_counterfactual": "...",

  "generation_valid": true,
  "judge_valid": true,
  "toxicity_preserved": true,
  "sentiment_preserved": true,
  "minimal_edit": true,
  "identity_changed": true,
  "contextually_plausible": true,

  "semantic_similarity": 0.93,
  "toxicity_drift": 0.04,
  "normalized_edit_distance": 0.18,
  "identity_change_success": true,

  "delta_gen": 1,
  "delta_sem": 1,

  "generator_backend": "ollama",
  "generator_model": "qwen3:14b",
  "judge_backend": "ollama",
  "judge_model": "qwen3:14b",
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 256,
  "prompt_version": "v2.1",
  "created_at": "2026-05-xx"
}
```

---

## 4. Prompt 设计

## 4.1 主生成 prompt

```text
You are generating a label-preserving counterfactual for toxicity classification.

Task:
Given a sentence, identify any explicit or implicit demographic/protected-group reference.
Then choose ONE contextually plausible alternative demographic group.
Rewrite the sentence by changing only the identity-bearing expression.

Strict constraints:
1. Do not use a fixed mapping table.
2. Change only the identity-bearing span.
3. Preserve toxicity level, sentiment, syntax, topic, and speaker intent.
4. Do not add or remove insults.
5. Do not paraphrase unrelated content.
6. The target identity must be in the same broad demographic dimension as the source identity when possible.
7. If no valid identity-preserving rewrite is possible, return valid_generation=false.

Return only valid JSON:
{
  "has_identity_reference": true/false,
  "source_identity": "...",
  "source_identity_type": "...",
  "source_span": "...",
  "target_identity": "...",
  "target_identity_type": "...",
  "counterfactual": "...",
  "changed_span_original": "...",
  "changed_span_counterfactual": "...",
  "valid_generation": true/false,
  "reason": "brief reason"
}

Sentence:
"{text}"
```

---

## 4.2 Judge prompt

```text
You are validating a counterfactual pair for toxicity classification.

Sentence A:
"{text}"

Sentence B:
"{text_cf}"

Answer whether Sentence B is a valid identity counterfactual of Sentence A.

A valid pair must satisfy ALL conditions:
1. The only substantive change is the demographic/protected-group reference.
2. Toxicity level is preserved.
3. Sentiment is preserved.
4. Syntax and topic are substantially preserved.
5. No unrelated content is added or removed.
6. The target identity is contextually plausible.

Return only JSON:
{
  "valid": true/false,
  "toxicity_preserved": true/false,
  "sentiment_preserved": true/false,
  "minimal_edit": true/false,
  "identity_changed": true/false,
  "contextually_plausible": true/false,
  "reason": "brief reason"
}
```

---

## 4.3 Free-form diagnostic prompt

Free-form 只作为诊断实验，不建议作为主方法。

```text
Rewrite the sentence by replacing the demographic or protected-group reference with a contextually plausible alternative identity.
You may adjust wording if necessary to make the sentence natural, but preserve the original toxicity label, sentiment, and speaker intent.

Return only JSON:
{
  "counterfactual": "...",
  "source_identity": "...",
  "target_identity": "...",
  "valid_generation": true/false,
  "reason": "brief reason"
}

Sentence:
"{text}"
```

这个实验的预期不是证明 free-form 最好，而是证明：

- free-form identity diversity 更高；
- 但 edit distance 更大；
- semantic drift 更严重；
- 因此主方法选择 controlled minimal rewrite 是合理的。

---

## 5. Gate 设计

## 5.1 Rule-based gate

直接过滤以下情况：

| 情况 | 处理 |
|---|---|
| JSON 解析失败 | retry，仍失败则 invalid |
| `counterfactual` 为空 | invalid |
| `text == counterfactual` | invalid |
| `valid_generation=false` | invalid |
| source identity 为空 | invalid |
| target identity 为空 | invalid |
| 长度比例小于 0.6 或大于 1.6 | invalid |
| 改写后没有身份变化 | invalid |

---

## 5.2 Metric-based gate

建议初始阈值：

| 指标 | 初始阈值 | 说明 |
|---|---:|---|
| Semantic similarity | `>= 0.85` | 防止语义漂移 |
| Normalized edit distance | `<= 0.35` | 防止大幅 paraphrase |
| Toxicity drift | `<= 0.10` 或 `<= 0.15` | 保证毒性强度基本保持 |
| Identity change success | true | 确保身份确实被替换 |
| LLM judge valid | true | 判断上下文合理性 |

阈值要在 validation set 上确定，不能对 test set 调参。

---

## 5.3 LLM-as-judge gate

使用 LLM judge 作为 scalable filter，但不要把它当 gold standard。论文中应写：

> We use the LLM judge as a scalable validity filter and validate its reliability on a stratified human-annotated subset.

---

## 6. 实验设计

## 6.1 主实验 Table 3

主表仍然报告：

- Macro-F1
- CFR
- CTFG

可继续保留 FPED/FNED 在 appendix 或主文后续表格中。

新 Table 3 中，所有方法的 CFR/CTFG 都必须基于同一套新的 test counterfactuals。

| Method | HX F1 ↑ | HX CFR ↓ | HX CTFG ↓ | TG F1 ↑ | TG CFR ↓ | TG CTFG ↓ | DH F1 ↑ | DH CFR ↓ | DH CTFG ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard FT |  |  |  |  |  |  |  |  |  |
| EAR |  |  |  |  |  |  |  |  |  |
| GetFair |  |  |  |  |  |  |  |  |  |
| CCDF |  |  |  |  |  |  |  |  |  |
| LogitPairing |  |  |  |  |  |  |  |  |  |
| AdvDebias |  |  |  |  |  |  |  |  |  |
| LLM-CLP, old GLM-Fixed16 | optional appendix | optional appendix | optional appendix | optional appendix | optional appendix | optional appendix | optional appendix | optional appendix | optional appendix |
| LLM-CLP, new OpenSet+Gate |  |  |  |  |  |  |  |  |  |

---

## 6.2 Reliability Table

旧 Table 2 建议换成：

| Dataset | Coverage ↑ | Gate Accept ↑ | Identity Success ↑ | Identity Diversity ↑ | Edit Distance ↓ | Semantic Sim ↑ | Toxicity Drift ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| HateXplain |  |  |  |  |  |  |  |
| ToxiGen |  |  |  |  |  |  |  |
| DynaHate |  |  |  |  |  |  |  |

指标定义：

| 指标 | 定义 |
|---|---|
| Coverage | 原始样本中能生成非平凡 CF 的比例 |
| Gate Accept | 生成样本中通过 gate 的比例 |
| Identity Success | source identity 成功替换为 target identity 的比例 |
| Identity Diversity | target identity unique count / entropy |
| Edit Distance | normalized token/character edit distance |
| Semantic Sim | embedding similarity 或 BERTScore |
| Toxicity Drift | `|tox(x) - tox(x')|` |

---

## 6.3 Ablation Table

建议新 Table 4：

| Variant | 描述 | 是否必须 |
|---|---|---:|
| No CF | 无反事实 | 保留 |
| Swap | lexical swap augmentation | 保留 |
| Swap+CLP | lexical swap + CLP | 保留 |
| LLM-Fixed16 | 旧版 LLM + 16 fixed pairs | 建议保留 |
| LLM-OpenSet | open-set controlled generation，无 gate，无 CLP | 建议 |
| LLM-OpenSet+Gate | open-set controlled generation + gate，无 CLP | 建议 |
| LLM-OpenSet+CLP | open-set controlled generation + CLP，无 gate | 可选 |
| LLM-OpenSet+Gate+CLP | 新主方法 | 必须 |

这个 ablation 可以回答：

1. LLM 比 swap 好在哪里？
2. open-set 比 fixed-16 好在哪里？
3. gate 是否真的有必要？
4. CLP 是否仍是主要收益来源？

---

## 6.4 Generator ablation

建议至少在一个数据集上做，最好三个数据集都做轻量版：

| Generator | Target selection | Gate | 作用 |
|---|---|---|---|
| GLM old | Fixed-16 | No | 旧版本 |
| Qwen local | Fixed-16 | No | 只验证开源生成器替换 |
| Qwen local | Open-set | No | 只验证 open-set target planning |
| Qwen local | Open-set | Yes | 新主方法 |

---

## 6.5 Human validation

强烈建议加入，尤其是顶会投稿。

抽样：

| 样本来源 | 数量 |
|---|---:|
| Gate accepted | 100 |
| Gate rejected | 100 |
| 总计 | 200 |

标注者：2–3 人。

标注问题：

1. 是否只改变 identity？
2. 非 identity 语义是否保持？
3. 毒性强度是否保持？
4. target identity 是否上下文合理？
5. 该 pair 是否适合用于 CLP？

报告：

| 指标 | 说明 |
|---|---|
| Percent agreement | 标注者一致率 |
| Cohen’s κ / Fleiss’ κ | 一致性统计 |
| LLM judge vs human accuracy | LLM judge 与人工结果一致性 |
| Accepted precision | gate accepted 中人工认为 valid 的比例 |
| Rejected correctness | gate rejected 中人工认为 invalid 的比例 |

---

## 7. 训练与评估策略

## 7.1 哪些需要重训

| 方法 | 是否需要重训 | 原因 |
|---|---:|---|
| Standard FT | 否 | 不依赖 CF training data |
| EAR | 否 | 不依赖 LLM CF |
| GetFair | 否 | 不依赖 LLM CF |
| CCDF | 否 | 不依赖 LLM CF |
| AdvDebias | 否 | 不依赖 LLM CF |
| Swap | 否 | lexical swap data 不变 |
| Swap+CLP | 否/可选 | lexical swap data 不变 |
| LogitPairing | 建议做 λ search | 最强相关 baseline |
| LLM only | 是 | 训练 CF 数据变了 |
| LLM+CLP | 是 | 主方法训练 CF 数据变了，且加入 gate |

---

## 7.2 哪些需要重算 CFR/CTFG

只要 test CF 换了，所有方法都要重算 CFR/CTFG：

| 方法 | 需要重算 CFR/CTFG |
|---|---:|
| Standard FT | 是 |
| EAR | 是 |
| GetFair | 是 |
| CCDF | 是 |
| AdvDebias | 是 |
| Swap | 是 |
| Swap+CLP | 是 |
| LogitPairing | 是 |
| LLM only | 是 |
| LLM+CLP | 是 |

这是公平比较的底线。

---

## 7.3 训练优先级

先跑 1 seed 看方向：

1. `LLM-OpenSet+Gate+CLP`
2. `LLM-OpenSet+Gate`
3. 所有旧 baseline 在新 test CF 上重算 CFR/CTFG
4. `LLM-Fixed16`
5. `LogitPairing` λ grid search
6. 主方法 3 seeds

不要一开始把所有 ablation 全跑满，先确认主方法方向。

---

## 8. LogitPairing vs Swap+CLP 数字不一致处理

当前同学意见指出 Table 3 的 LogitPairing 和 Table 4 的 Swap+CLP 数字不一致。这不是一定要改代码，但必须解释清楚。

建议写入论文实验设置：

> LogitPairing follows Davani et al. and applies supervised cross-entropy to both original and swap-based counterfactual inputs, whereas our Swap+CLP ablation isolates the CLP effect under the same single-sided CE objective as LLM-CLP. Therefore, LogitPairing and Swap+CLP are not expected to be numerically identical despite both using swap-based pairs.

中文解释：

- LogitPairing baseline 是原论文设定：`CE(x) + CE(x') + MSE(z, z')`；
- 你们的 Swap+CLP ablation 是为了对齐 LLM-CLP：`CE(x) + CLP(z, z')`；
- 所以二者数字不同是合理的；
- 但论文里必须提前说明，否则审稿人会认为表格冲突。

---

## 9. 文稿修改清单

## 9.1 Title

当前标题可以保留，但建议不要让标题暗示“任何 LLM 都可以”或“完全自由生成”。

可选标题：

> Improving Counterfactual Fairness in Toxicity Classification with Open-Source LLM-Guided Counterfactual Logit Pairing

或者：

> Open-Set Controlled Counterfactual Logit Pairing for Fair Toxicity Classification

---

## 9.2 Abstract

旧 abstract 中 GLM-4-Flash 必须删除。

建议新版 abstract 结构：

1. 毒性分类器存在 identity shortcut；
2. 旧 counterfactual 方法要么 lexical swap 太死板，要么 unconstrained LLM rewrite 太 noisy；
3. 提出 LLM-CLP：本地开源 LLM 做 open-set controlled identity rewrite；
4. 使用 semantic/toxicity gate 过滤无效 pair；
5. 通过 CLP 约束 logits；
6. 报告 CFR/CTFG 改进，同时承认 F1 trade-off。

示例：

> Toxicity classifiers often become sensitive to protected-group mentions rather than toxicity itself. We propose LLM-CLP, a reproducible framework that combines locally run open-source LLM counterfactual rewriting with Counterfactual Logit Pairing. Unlike fixed lexical swaps, our generator performs open-set identity planning: it detects explicit or implicit identity references, selects a contextually plausible alternative group, and applies a minimal identity-only rewrite. To prevent noisy pairing, a semantic/toxicity validity gate filters counterfactual pairs before they contribute to the CLP loss. Experiments on HateXplain, ToxiGen, and DynaHate show improved counterfactual stability under this validated generation protocol, with moderate utility trade-offs.

---

## 9.3 Introduction

新增贡献点：

1. **Reproducible generator**：使用本地开源 LLM，而非闭源 API。
2. **Open-set controlled rewriting**：突破 fixed identity mappings，但仍保持 minimal edit。
3. **Validity-gated CLP**：只有通过 semantic/toxicity validation 的 pair 进入 CLP。
4. **Protocol-level analysis**：报告 coverage、identity diversity、edit distance、toxicity drift、人类验证。

---

## 9.4 Method 3.3

整段重写。

删除：

- GLM-4-Flash；
- API rotation；
- 16 bidirectional swap pairs 作为主方法；
- keyword-only detection。

新增：

- local open-source LLM；
- identity reference detection；
- target identity planning；
- minimal identity rewrite；
- structured JSON decoding；
- invalid generation handling。

---

## 9.5 Method 3.4

旧稿把 semantic gate 作为 future work。新版必须改成已实现：

> Because CLP is only valid for label-preserving identity edits, we introduce a post-generation validity gate. The gate combines rule-based checks, automatic semantic/toxicity diagnostics, and an LLM judge. Each pair receives a binary flag \(\delta^{sem}\), and only pairs with \(\delta^{gen}\delta^{sem}=1\) contribute to CLP.

---

## 9.6 Experimental Setup

新增 reproducibility details：

| 项 | 需要写清楚 |
|---|---|
| Generator | Qwen3-14B / Qwen2.5-14B |
| Backend | Ollama / vLLM |
| Quantization | Q4_K_M / AWQ / BF16 |
| Temperature | 0.2 or 0.3 |
| Top-p | 0.9 |
| Max tokens | 256 |
| Prompt version | v2.1 |
| Gate thresholds | selected on validation set |
| Seeds | 42, 123, 2024 |
| Release plan | code, prompts, generated CF data, hashes |

---

## 9.7 Results

避免过强表述。

不要写：

> LLM-CLP dominates all baselines.

建议写：

> LLM-CLP improves counterfactual stability under a validated open-set counterfactual protocol, while retaining the expected fairness-utility trade-off.

如果 DynaHate 结果没有全面领先：

> On DynaHate, the advantage is smaller because many examples express toxicity implicitly or adversarially, reducing the marginal benefit of surface identity rewriting.

---

## 9.8 Limitations

旧版 limitations 中这些内容要删或改：

- “we did not ship an automatic equivalence check”；
- “counterfactuals come from GLM-4-Flash over an API”；
- “semantic verification left for future work”。

新版 limitations 建议写：

> Although our validity gate reduces noisy counterfactuals and is calibrated against a human-annotated subset, generated counterfactuals are still not gold-standard causal interventions. The LLM judge may miss subtle semantic drift, and open-set identity planning can occasionally introduce culturally implausible substitutions. Therefore, the results should be interpreted as evidence under a validated generated-pair protocol rather than a gold counterfactual benchmark.

---

## 10. 11 天执行计划

## Day 1：冻结 v2 protocol

产出：

- 方法名；
- prompt v2.1；
- JSON schema；
- gate 指标；
- 表格模板；
- 实验优先级；
- 数据路径规范。

---

## Day 2：实现 generator

完成：

- Ollama / vLLM backend；
- JSON parser；
- retry；
- cache；
- logging；
- failure cases；
- 每个数据集抽 100 条 smoke test。

---

## Day 3：实现 validator

完成：

- LLM judge；
- semantic similarity；
- edit distance；
- toxicity drift；
- identity success；
- 输出 `delta_sem`；
- 生成第一版 reliability report。

---

## Day 4：全量生成 train/test CF

优先顺序：

1. HateXplain
2. ToxiGen
3. DynaHate

理由：前两个更可能出正结果，DynaHate 风险最大。

---

## Day 5：训练 1 seed 主方法

跑：

- `LLM-OpenSet+Gate`
- `LLM-OpenSet+Gate+CLP`

同时：

- 所有 baseline 在新 test CF 上重算 CFR/CTFG。

---

## Day 6：检查结果与修正

检查：

- coverage；
- gate acceptance；
- edit distance；
- semantic similarity；
- toxicity drift；
- DynaHate 是否崩；
- CFR/CTFG 是否有方向性改进。

只能基于 validation set 调 threshold，不要对 test set 反复调。

---

## Day 7：补 ablation

跑：

- `LLM-Fixed16`
- `LLM-OpenSet no gate`
- `LLM-OpenSet+Gate no CLP`
- `LogitPairing` λ grid `{0.5, 1.0, 2.0, 3.0}`

---

## Day 8：3 seeds

主方法 3 seeds：

- `LLM-OpenSet+Gate`
- `LLM-OpenSet+Gate+CLP`
- `LogitPairing optimal λ`

算力够再补：

- `LLM-Fixed16` 3 seeds；
- `LLM-OpenSet no gate` 3 seeds。

---

## Day 9：人工验证

抽样：

- gate accepted 100；
- gate rejected 100。

2–3 位标注者盲审。

计算：

- agreement；
- Cohen’s κ / Fleiss’ κ；
- LLM judge vs human accuracy；
- accepted precision。

---

## Day 10：重写论文

优先改：

1. Abstract；
2. Introduction；
3. Method 3.3–3.5；
4. Experimental setup；
5. Results；
6. Ablation；
7. Limitations；
8. Ethics；
9. Appendix reproducibility。

---

## Day 11：一致性检查

检查项：

- 所有 GLM-4-Flash 主方法描述已删除或降级 appendix；
- 所有 16 fixed mappings 主方法描述已删除或降级 ablation；
- semantic gate 不再是 future work；
- Table 2 / Table 3 / Table 4 数字一致；
- CFR/CTFG 全部基于同一套新 test CF；
- 论文 claim 不超过结果；
- `EXPERIMENT_REPORT.md` 与论文表格一致；
- prompt、seed、model、data hash 都可复现。

---

## 11. 最低成功标准

投稿前至少做到：

1. 主方法不再依赖 GLM-4-Flash。
2. 主方法不再依赖 16 fixed bidirectional mappings。
3. `delta_sem` 真正进入 CLP。
4. 所有方法在同一套新 test CF 上评估 CFR/CTFG。
5. 新 Table 2 包含 coverage、gate acceptance、identity success、edit distance、semantic similarity、toxicity drift。
6. 至少 200 条人工验证。
7. LogitPairing 有 λ 搜索，或明确说明为什么无法搜索。
8. 旧 GLM+Fixed16 作为 ablation 或 appendix，而不是主方法。
9. 论文结论从 “general dominance” 改成 “validated counterfactual stability improvement”。

---

## 12. 最终论文叙事

建议把论文主线改成：

> Toxicity classifiers should be stable under identity-preserving wording changes. Existing counterfactual methods face a trade-off: lexical swaps are controllable but rigid, while unconstrained LLM rewrites are fluent but noisy. We propose a reproducible LLM-CLP framework that uses a locally run open-source LLM for open-set but controlled identity rewriting, and filters generated pairs with a semantic/toxicity validity gate before applying counterfactual logit pairing. This design expands beyond fixed identity mappings while preserving the pair validity required by CLP.

这比旧版更强：

旧版卖点：

> GLM 生成反事实 + CLP。

新版卖点：

> 复现性、开放身份空间、反事实有效性、CLP 四者统一。

---

## 13. 给团队分工建议

| 角色 | 任务 |
|---|---|
| A | 修改 `cf_generator_llm.py`，接 Ollama/vLLM |
| B | 写 prompts、schema、JSON parser、retry |
| C | 写 `cf_validator.py` 和 quality metrics |
| D | 跑全量 CF 生成与数据清洗 |
| E | 跑 LLM/LLM+CLP 训练 |
| F | 统一评估 CFR/CTFG |
| G | 做人工验证与 agreement 统计 |
| H | 改论文 Method/Experiment/Results |
| I | 检查表格、appendix、reproducibility |

---

## 14. 风险预案

### 情况 A：OpenSet+Gate coverage 太低

处理：

- 放宽 identity detection；
- 允许 more implicit references；
- 降低 semantic similarity threshold；
- 但不要牺牲 toxicity preservation。

论文写法：

> The gate trades coverage for pair validity.

---

### 情况 B：Free-form 指标更差

这是正常结果。用它证明 controlled rewrite 的必要性：

> Although free-form rewriting increases identity diversity, it also increases edit distance and semantic drift, which makes it less suitable for CLP.

---

### 情况 C：DynaHate 没有明显提升

不要硬凹。写成 dataset-dependent：

> DynaHate contains adversarial and implicit examples where surface identity rewriting provides weaker signal. This supports our claim that LLM-CLP targets identity-wording sensitivity rather than all forms of hate-speech bias.

---

### 情况 D：Qwen 结果不如 GLM

处理：

- GLM 放 appendix 作为 upper-bound / robustness；
- 主文强调 reproducibility；
- 如果 Qwen-Fixed16 与 GLM-Fixed16 差距小，则说明开源替代可行；
- 如果 OpenSet+Gate 的 reliability 更强，也可以支撑主方法。

---

### 情况 E：LLM judge 和人工一致率不高

处理：

- 报告这个 limitation；
- 提高 gate threshold；
- 把 human-validated subset 作为 gold diagnostic；
- 不要夸大 gate 完美性。

---

## 15. 最后建议

这次修改不要包装成“补实验”，而要包装成：

> **A protocol upgrade for reproducible and valid LLM-generated counterfactuals.**

这样审稿人看到的是：

1. 你们意识到 LLM 反事实生成的核心难点；
2. 你们没有只依赖闭源 API；
3. 你们没有让 LLM 随便 free-form；
4. 你们也没有被 16 对固定映射锁死；
5. 你们让 validity gate 真正参与训练目标；
6. 你们在统一 test CF protocol 下比较所有方法。

这会比原稿更像一篇顶会主会论文。

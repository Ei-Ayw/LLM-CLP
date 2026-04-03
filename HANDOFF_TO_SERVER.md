# 交接文档: NLP 毒性分类 - LLM 反事实增强 + 因果公平

> 本文档供服务器端 Claude Code 继续执行训练和评估流程

---

## 1. 项目背景

NLP 毒性分类研究项目，目标 CCFA 顶会 (ACL/EMNLP/AAAI)。
核心创新: **LLM 反事实数据增强 + 因果公平训练**，而非传统简单换词 (CDA-Swap)。

### 训练目标函数
```
L_total = L_CE + λ_clp × L_CLP + λ_con × L_SupCon
```
- L_CE: 交叉熵分类损失
- L_CLP: Counterfactual Logit Pairing (原文-反事实 logit 对齐)
- L_SupCon: Supervised Contrastive Loss (原文-反事实 特征拉近)

---

## 2. 已完成的工作

### 数据准备 (全部完成)
| 文件 | 说明 | 行数 |
|------|------|------|
| `data/causal_fair/hatexplain_train.parquet` | HateXplain 训练集 | 15,383 |
| `data/causal_fair/hatexplain_val.parquet` | HateXplain 验证集 | 1,922 |
| `data/causal_fair/hatexplain_test.parquet` | HateXplain 测试集 | 1,924 |
| `data/causal_fair/toxigen_train.parquet` | ToxiGen 训练集 | 7,168 |
| `data/causal_fair/toxigen_val.parquet` | ToxiGen 验证集 | 896 |
| `data/causal_fair/toxigen_test.parquet` | ToxiGen 测试集 | 896 |
| `data/causal_fair/hatecheck_test.parquet` | HateCheck 诊断集 | 3,728 |
| `data/causal_fair/hatexplain_train_cf_swap.parquet` | 传统换词反事实(train) | 22,571 |
| `data/causal_fair/hatexplain_test_cf_swap.parquet` | 传统换词反事实(test) | 2,796 |
| `data/causal_fair/hatexplain_train_cf_llm.parquet` | **LLM反事实(train)** | 12,046 (验证后) |
| `data/causal_fair/hatexplain_test_cf_llm.parquet` | **LLM反事实(test)** | 2,640 (验证后) |

### 模型下载 (已完成)
- DeBERTa-v3-base 已下载到 `models/deberta-v3-base/`
- 包含 `model.safetensors`(702MB) + tokenizer 文件

### E1 (Baseline) 训练已完成
- 模���权重: `src_result/models/hatexplain_none_clp0.0_con0.0_seed42_0312_0948.pth`
- 训练日志: `src_result/logs/hatexplain_none_clp0.0_con0.0_seed42_0312_0948_results.json`
- **Test 结果: F1=0.7929 | AUC=0.8878 | Acc=0.8041**

---

## 3. 待完成的工作

### Step 1: 运行 E2-E5 训练实验

**注意事项**:
- `--model_name` 需改为服务器上模型路径 (或用 `microsoft/deberta-v3-base` 在线下载)
- 3090 24GB 显存可以用 `--batch_size 16 --grad_accum 2` (有效 batch=32)
- 如果显存充裕可以尝试 `--batch_size 32 --grad_accum 1`

```bash
cd <项目根目录>
export TOKENIZERS_PARALLELISM=false

# 服务器上模型路径，按实际修改 (或用 microsoft/deberta-v3-base 在线下载)
MODEL_PATH="models/deberta-v3-base"

# E2: 传��换词反事实 + CLP (对照组)
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method swap \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

# E3: LLM 反事实 + CLP
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --seed 42

# E4: LLM 反事实 + SupCon
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 0.0 --lambda_con 0.5 \
    --seed 42

# E5: LLM 反事实 + CLP + SupCon (完整方法，期望最佳)
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 5 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.5 \
    --seed 42
```

### Step 2: 因果公平评估

```bash
for CKPT in src_result/models/hatexplain_*.pth; do
    echo "评估: $CKPT"
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" \
        --dataset hatexplain \
        --cf_method llm
done
```

---

## 4. 关键代码文件说明

| 文件 | 作用 |
|------|------|
| `src_model/model_deberta_cf.py` | DeBERTa + 分类头 + 对比学习投影头 |
| `src_script/train/train_causal_fair.py` | 主训练脚本 (含 5 实验的 CLI) |
| `src_script/data/data_loader_cf.py` | 数据加载器 (原文+反事实配对) |
| `src_script/utils/loss_contrastive.py` | CLP + SupCon 损失函数 |
| `src_script/eval/eval_causal_fairness.py` | 因果公平评估 (CFR/FPED/FNED) |
| `src_script/counterfactual/cf_generator_llm.py` | LLM 反事实生成器 (已完成) |
| `src_script/counterfactual/cf_generator_swap.py` | 传统换词生成器 (已完成) |
| `src_script/counterfactual/cf_validator.py` | 反事实质量验证 (已完成) |

### 训练脚本 CLI 参数一览
```
--dataset         hatexplain|toxigen (默认 hatexplain)
--cf_method       none|swap|llm (反事实方法)
--model_name      预训练模型路径 (默认 models/deberta-v3-base)
--max_len         最大序列长度 (默认 128)
--batch_size      批次大小 (3090 建议 16)
--grad_accum      梯度累积 (3090 建议 2, 有效batch=32)
--epochs          训练轮次 (默认 5)
--lr              学习率 (默认 2e-5)
--lambda_clp      CLP 损失权重 (0.0-1.0)
--lambda_con      SupCon 损失权重 (0.0-0.5)
--temperature     对比学习温度 (默认 0.07)
--seed            随机种子 (默认 42)
--patience        早停耐心 (默认 3)
--data_dir        数据目录 (默认 <BASE_DIR>/data/causal_fair)
```

---

## 5. 依赖安装

```bash
pip install torch transformers sentencepiece accelerate \
    pandas numpy pyarrow scikit-learn tqdm matplotlib seaborn \
    python-dotenv safetensors
```

注意: 服务器不需要 `zai-sdk` (反事实已生成完毕)

---

## 6. 期望结果对比

| 实验 | 方法 | 期望 F1 | 期望 CFR↓ | 说明 |
|------|------|---------|-----------|------|
| E1 | Baseline | **0.7929** (已有) | 高 | 无反事实，无公平约束 |
| E2 | Swap+CLP | ~0.78-0.79 | 中 | 传统换词对照组 |
| E3 | LLM+CLP | ~0.79-0.80 | 中低 | LLM反事实应优于swap |
| E4 | LLM+SupCon | ~0.79-0.80 | 中低 | 对比学习路线 |
| E5 | LLM+CLP+SupCon | ~0.79-0.81 | **最低** | 完整方法，核心卖点 |

核心论点: **E5 在保持分类性能的同时，显著降低 CFR (反事实翻转率)**

---

## 7. 数据传输清单

需要从本地传到���务器的文件/目录:
```
data/causal_fair/               # 所有 parquet 数据文件 (~8.9MB)
models/deberta-v3-base/         # 预训练模型 (~712MB), 也可服务器在线下载
src_model/model_deberta_cf.py   # 模型定义
src_script/train/train_causal_fair.py
src_script/data/data_loader_cf.py
src_script/utils/loss_contrastive.py
src_script/utils/path_config.py
src_script/utils/train_utils.py
src_script/eval/eval_causal_fairness.py
requirements.txt
```

**建议直接 git push 然后服务器 git pull，只需额外传 `data/causal_fair/` 和 `models/` 目录。**

---

## 8. 已知问题 & 注意事项

1. **模型路径**: `--model_name` 目前写死为本地路径，需要在服务器上改成服务器路径或使用 `microsoft/deberta-v3-base`
2. **E1 结果**: 本地已有 E1 baseline 结果 (F1=0.7929)，服务器可以跳过 E1 直接跑 E2-E5。如果想对比，也可以重新跑 E1
3. **safetensors**: `model_deberta_cf.py` 已修复为不限制加载格式 (`from_pretrained(model_path)`)
4. **数据列名**:
   - HateXplain: `post_id, text, label, binary_label, target_groups, coarse_groups, has_identity, rationale`
   - 反事实: `post_id, original_text, cf_text, source_group, target_group, method`
5. **Git 分支**: `refactor/v2-rethink`

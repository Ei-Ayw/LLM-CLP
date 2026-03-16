# HANDOFF V2: NLP 毒性分类 - 因果公平实验

> 新机器上让 Claude 读这个文件: "读 HANDOFF_V2.md，继续执行"

---

## 1. 项目状态

分支: `refactor/v2-rethink`
模型: `models/deberta-v3-base` (DeBERTa-v3-base, 需要下载或从旧机器拷贝)

---

## 2. 已完成的实验

### HateXplain (3 seeds: 42/123/2024)

| 实验 | 方法 | F1 (mean±std) | AUC (mean±std) | CFR↓ (mean) |
|------|------|:---:|:---:|:---:|
| E1 | Baseline | 0.7973±0.0066 | 0.8882±0.0010 | 0.1462 |
| E2 | Swap+CLP | 0.7995±0.0034 | 0.8877±0.0008 | 0.1244 |
| E3 | LLM+CLP | 0.7883±0.0030 | 0.8774±0.0010 | **0.0571** |
| E4 | LLM+SupCon | 0.7915±0.0042 | 0.8853±0.0033 | 0.1033 |
| E5 | LLM+CLP+SupCon(0.5) | 0.7875±0.0021 | 0.8748±0.0005 | 0.0641 |

E5 超参调优 (seed=42 only):

| clp | con | F1 | CFR↓ |
|-----|-----|---:|-----:|
| 1.0 | 0.1 | 0.7901 | 0.0583 |
| 1.0 | 0.2 | 0.7902 | 0.0591 |
| 1.0 | 0.3 | 0.7893 | 0.0580 |
| 0.5 | 0.1 | 0.7898 | 0.0644 |
| 0.5 | 0.2 | 0.7922 | 0.0659 |

### ToxiGen (3 seeds: 42/123/2024)

| 实验 | 方法 | F1 (mean) | AUC (mean) |
|------|------|:---------:|:----------:|
| E1 | Baseline | 0.8495 | 0.9317 |
| E2 | Swap+CLP | 0.8595 | 0.9376 |
| E3 | LLM+CLP | 0.8589 | 0.9304 |
| E4 | LLM+SupCon | 0.8486 | 0.9277 |
| E5 | LLM+CLP+SupCon | 0.8517 | 0.9288 |

---

## 3. 核心问题诊断

### 为什么 fairness 方法 F1 低于 Baseline?
- HateXplain 数据严重依赖捷径: 有身份词毒性率=0.756, 无身份词=0.198
- 纯捷径规则 (有身份词=有毒) 就能拿 F1=0.74, Baseline 只比它高 0.053
- CLP/SupCon 消除捷径后, CE 收敛变慢, 5 epoch 不够
- **不是过拟合, 是欠拟合** (epoch 5 时 Val F1 仍在上升)

### 数据不平衡
- Train: 15,383 条 (Toxic 59.4% / Non-toxic 40.6%)
- 交叉不平衡: 有身份词+有毒 8,245 vs 有身份词+无毒 2,655 vs 无身份词+有毒 887
- 反事实覆盖率: 81.7% (有身份词样本)

---

## 4. 已实施的代码改进 (未跑实验)

文件: `src_script/train/train_causal_fair.py` 已修改:

### 4a. Focal Loss 替代普通 CE
- 新增 `FocalCrossEntropyLoss` 类 (gamma=2.0)
- 自动计算 class_weight 并传入
- CLI: `--focal_gamma 2.0` (设为 0 退化为 weighted CE)

### 4b. 两阶段训练 (Warmup CE → Fairness)
- 前 N epoch 只用 Focal CE, 让分类先收敛
- 之后加入 CLP + SupCon
- CLI: `--warmup_epochs 2`

### 4c. 增加训练轮次
- epochs 默认 5→10, patience 默认 3→5

---

## 5. 待执行: 改进版实验

### Step 1: 快速验证 (E3v2 + E5v2, seed=42)

```bash
export TOKENIZERS_PARALLELISM=false
MODEL_PATH="models/deberta-v3-base"

# E3v2: LLM+CLP + Focal + Warmup
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 10 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 \
    --focal_gamma 2.0 --warmup_epochs 2 --seed 42

# E5v2: LLM+CLP+SupCon(0.2) + Focal + Warmup
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name "$MODEL_PATH" \
    --epochs 10 --batch_size 16 --grad_accum 2 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.2 \
    --focal_gamma 2.0 --warmup_epochs 2 --seed 42
```

期望: F1 提升到 ~0.80+ (接近 Baseline), CFR 保持 ~0.05

### Step 2: 如果 Step 1 有效, 跑多 seed + ToxiGen

对 E1-E5 全部加上 `--focal_gamma 2.0 --warmup_epochs 2 --epochs 10`, 3 seeds × 2 datasets

### Step 3: 因果公平评估

```bash
for CKPT in src_result/models/hatexplain_*.pth; do
    python src_script/eval/eval_causal_fairness.py \
        --checkpoint "$CKPT" --dataset hatexplain --cf_method llm
done
```

注意: `eval_causal_fairness.py` 的 `--model_name` 默认已改为 `models/deberta-v3-base`

---

## 6. 还需要做的补充实验 (发 CCF-A)

1. **分子集评估**: 在"有身份词"和"无身份词"子集上分别报 F1
2. **HateCheck 诊断评估**: 数据已有 `data/causal_fair/hatecheck_test.parquet` (3,728条)
3. **跨数据集泛化**: HateXplain 训练 → ToxiGen 测试
4. **外部 baseline 对比**: FairBatch / Adversarial Debiasing 等

---

## 7. 关键文件

| 文件 | 说明 |
|------|------|
| `src_script/train/train_causal_fair.py` | 主训练脚本 (已改: Focal+Warmup) |
| `src_model/model_deberta_cf.py` | DeBERTa + 分类头 + 投影头 |
| `src_script/data/data_loader_cf.py` | 反事实配对数据加载器 |
| `src_script/utils/loss_contrastive.py` | CLP + SupCon 损失 |
| `src_script/eval/eval_causal_fairness.py` | 因果公平评估 (CFR/CTFG/FPED/FNED) |
| `data/causal_fair/` | 所有 parquet 数据 (HateXplain + ToxiGen + HateCheck) |
| `models/deberta-v3-base/` | 预训练模型 (~712MB) |

---

## 8. 依赖

```bash
pip install torch transformers sentencepiece accelerate \
    pandas numpy pyarrow scikit-learn tqdm matplotlib seaborn \
    python-dotenv safetensors zhipuai
```

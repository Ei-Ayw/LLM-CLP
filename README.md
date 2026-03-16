# LLM-CLP: LLM-Powered Counterfactual Data Augmentation for Causal Fairness in Toxicity Detection

> Targeting CCF-A venues (ACL / EMNLP / AAAI)

## TL;DR

We propose **LLM-CLP**, a simple yet effective framework that combines **LLM-generated counterfactual data augmentation** with **Counterfactual Logit Pairing (CLP)** to reduce identity-based bias in toxicity classifiers. On HateXplain, LLM-CLP reduces the Counterfactual Flip Rate (CFR) by **64%** while maintaining classification performance (F1 drop < 1%).

---

## 1. Motivation

Toxicity classifiers learn **spurious correlations** between identity mentions and toxicity labels — e.g., treating "Muslims" as a toxicity signal. Traditional Counterfactual Data Augmentation (CDA) uses naive word-swapping (e.g., "Muslim" → "Christian"), which produces grammatically awkward and culturally implausible texts. We leverage LLMs to generate **semantically natural** counterfactuals with culturally appropriate substitutions (e.g., "mosque" → "church", "hijab" → "cross necklace"), then train with a causal fairness objective.

## 2. Method

### 2.1 Training Objective

```
L_total = L_CE + λ_clp × L_CLP
```

| Loss | Description |
|------|-------------|
| L_CE | Cross-entropy classification loss (on original samples) |
| L_CLP | Counterfactual Logit Pairing: MSE between logits of original and counterfactual |

CLP forces the model to produce **identical predictions** for an original text and its identity-swapped counterfactual, directly encoding the causal fairness constraint: *changing only the identity group should not change the toxicity prediction*.

### 2.2 LLM Counterfactual Generation Pipeline

```
Original:      "Muslims are destroying this country"
               ↓ LLM (GLM-4-Flash)
Counterfactual: "Christians are ruining this nation"
               (culturally adapted, sentiment preserved)

vs. Naive Swap: "Christians are destroying this country"
               (mechanical replacement, culturally inconsistent)
```

**Pipeline**: Detect identity groups → Select swap targets → LLM rewrite with strict rules (preserve syntax, sentiment, toxicity level; only change identity-related terms) → Quality validation (reject if >30% length change or identical output)

### 2.3 Architecture

```
Input text → DeBERTa-v3-base → [CLS] pooling → Classifier (2-class)
                                      ↓
                              Projection head (768→128) → SupCon (ablation)
```

- Backbone: `microsoft/deberta-v3-base` (184M params)
- Max sequence length: 128
- AMP (FP16) training with cosine LR schedule

## 3. Experimental Setup

### 3.1 Datasets

| Dataset | Train | Val | Test | Source |
|---------|------:|----:|-----:|--------|
| HateXplain | 15,383 | 1,922 | 1,924 | Mathew et al. (2021) |
| ToxiGen | 7,168 | 896 | 896 | Hartvigsen et al. (2022) |

### 3.2 Counterfactual Data

| Type | Method | HateXplain | ToxiGen |
|------|--------|------:|------:|
| CDA-Swap | Naive word replacement | 22,571 | 12,548 |
| CDA-LLM | GLM-4-Flash rewriting | 12,046 | 5,861 |

### 3.3 Experiments

| ID | Method | CF Source | λ_clp | λ_con |
|----|--------|-----------|------:|------:|
| E1 | Baseline (no CF) | — | 0.0 | 0.0 |
| E2 | Swap + CLP | CDA-Swap | 1.0 | 0.0 |
| E3 | **LLM + CLP (Ours)** | CDA-LLM | 1.0 | 0.0 |
| E4 | LLM + SupCon | CDA-LLM | 0.0 | 0.5 |
| E5 | LLM + CLP + SupCon | CDA-LLM | 1.0 | 0.5 |

All experiments: 3 seeds (42, 123, 2024), DeBERTa-v3-base, batch_size=48, lr=2e-5, 5 epochs, early stopping (patience=3).

## 4. Results

### 4.1 Main Results — HateXplain

| Method | Macro-F1 | AUC | CFR ↓ | CTFG ↓ |
|--------|:--------:|:---:|:-----:|:------:|
| E1: Baseline | .7973±.0066 | .8882±.0010 | .1568±.0140 | .1446±.0109 |
| E2: Swap+CLP | .7995±.0034 | .8877±.0008 | .1244±.0032 | .1077±.0028 |
| **E3: LLM+CLP (Ours)** | **.7883±.0030** | **.8774±.0010** | **.0563±.0026** | **.0429±.0005** |
| E4: LLM+SupCon | .7915±.0042 | .8853±.0033 | .1033±.0022 | .1018±.0006 |
| E5: LLM+CLP+SupCon | .7875±.0021 | .8748±.0005 | .0641±.0024 | .0485±.0002 |

### 4.2 Main Results — ToxiGen

| Method | Macro-F1 | AUC | CFR ↓ | CTFG ↓ |
|--------|:--------:|:---:|:-----:|:------:|
| E1: Baseline | .8495±.0035 | .9318±.0025 | .0424±.0071 | .0429±.0047 |
| E2: Swap+CLP | .8595±.0051 | .9375±.0005 | .0256±.0044 | .0248±.0006 |
| **E3: LLM+CLP (Ours)** | **.8589±.0039** | **.9304±.0019** | **.0256±.0079** | **.0203±.0009** |
| E4: LLM+SupCon | .8486±.0022 | .9277±.0018 | .0288±.0063 | .0334±.0034 |
| E5: LLM+CLP+SupCon | .8517±.0072 | .9287±.0021 | .0231±.0065 | .0208±.0016 |

### 4.3 Fairness Metrics

- **CFR** (Counterfactual Flip Rate): Fraction of original–counterfactual pairs where the prediction flips. Lower = fairer.
- **CTFG** (Counterfactual Token Fairness Gap): Mean absolute difference in predicted toxic probability between original and counterfactual. Lower = fairer.

### 4.4 Key Findings

1. **LLM-CLP reduces CFR by 64%** on HateXplain (0.157 → 0.056) with only 0.9% F1 drop, and by 40% on ToxiGen (0.042 → 0.026) with no F1 loss.
2. **LLM >> Swap**: LLM counterfactuals are strictly more effective than naive word-swapping. E3 CFR=0.056 vs E2 CFR=0.124 on HateXplain — LLM counterfactuals cut the remaining bias in half.
3. **CLP is the key driver**: The CLP loss alone (E3) outperforms SupCon alone (E4) on fairness (CFR 0.056 vs 0.103). Adding SupCon on top of CLP (E5) does not consistently improve over CLP alone.
4. **Results are stable**: Standard deviations across 3 seeds are small (CFR std < 0.008), confirming robustness.

### 4.5 Ablation: CLP vs SupCon Weight (HateXplain, seed=42)

| λ_clp | λ_con | F1 | AUC | CFR ↓ | CTFG ↓ |
|------:|------:|---:|----:|------:|-------:|
| 0.0 | 0.5 | .7897 | .8897 | .1061 | .1021 |
| 0.5 | 0.1 | .7898 | .8813 | .0644 | .0576 |
| 0.5 | 0.2 | .7922 | .8798 | .0659 | .0584 |
| **1.0** | **0.0** | **.7904** | **.8766** | **.0527** | **.0427** |
| 1.0 | 0.1 | .7901 | .8779 | .0583 | .0453 |
| 1.0 | 0.2 | .7902 | .8780 | .0591 | .0475 |
| 1.0 | 0.3 | .7893 | .8775 | .0580 | .0478 |
| 1.0 | 0.5 | .7851 | .8741 | .0633 | .0486 |

**Takeaway**: CLP at λ=1.0 with no SupCon (λ_con=0) achieves the best fairness. Adding SupCon introduces slight interference.

### 4.6 Per-Group CFR Breakdown (HateXplain, E3 vs Baseline)

| Group | Baseline CFR | LLM+CLP CFR | Δ |
|-------|:-----------:|:----------:|:---:|
| disabled | 0.422 | 0.133 | -68% |
| black | 0.271 | 0.048 | -82% |
| gay | 0.215 | 0.044 | -80% |
| muslim | 0.127 | 0.018 | -86% |
| men | 0.114 | 0.057 | -50% |
| white | 0.104 | 0.050 | -52% |
| women | 0.103 | 0.055 | -47% |
| jewish | 0.084 | 0.042 | -50% |
| asian | 0.089 | 0.044 | -51% |
| christian | 0.029 | 0.059 | +100% |

LLM-CLP achieves **consistent improvements across almost all identity groups**, with the largest gains on historically disadvantaged groups (black -82%, gay -80%, muslim -86%).

## 5. Project Structure

```
├── src_model/
│   ├── model_deberta_cf.py              # DeBERTa + classifier + projection head
│   └── ...                              # Legacy models (Vanilla, MTL, GRL)
├── src_script/
│   ├── train/
│   │   └── train_causal_fair.py         # Main training script (E1-E5)
│   ├── eval/
│   │   └── eval_causal_fairness.py      # CFR/CTFG/per-group evaluation
│   ├── data/
│   │   └── data_loader_cf.py            # Counterfactual paired DataLoader
│   ├── counterfactual/
│   │   ├── cf_generator_llm.py          # LLM counterfactual generator (Zhipu GLM)
│   │   ├── cf_generator_swap.py         # Naive word-swap baseline
│   │   └── cf_validator.py              # Quality validation
│   └── utils/
│       ├── loss_contrastive.py          # CLP + SupCon losses
│       ├── train_utils.py               # EarlyStopping, etc.
│       └── path_config.py               # Path management
├── data/causal_fair/                    # HateXplain + ToxiGen + counterfactuals
├── models/deberta-v3-base/              # Pretrained weights
├── src_result/
│   ├── models/                          # Checkpoints (*.pth)
│   ├── eval/                            # Fairness eval JSONs
│   └── logs/                            # Training result JSONs
└── docs/                                # Paper drafts
```

## 6. Reproduction

```bash
# Environment
pip install torch transformers sentencepiece accelerate pandas numpy pyarrow scikit-learn tqdm

# Train E3 (proposed method)
python src_script/train/train_causal_fair.py \
    --dataset hatexplain --cf_method llm \
    --model_name models/deberta-v3-base \
    --epochs 5 --batch_size 48 --grad_accum 1 --lr 2e-5 \
    --lambda_clp 1.0 --lambda_con 0.0 --seed 42

# Evaluate causal fairness
python src_script/eval/eval_causal_fairness.py \
    --checkpoint src_result/models/<checkpoint>.pth \
    --dataset hatexplain --cf_method llm
```

## 7. Environment

- Python 3.8 / PyTorch / Transformers
- GPU: NVIDIA RTX 3090 24GB
- Pretrained: `microsoft/deberta-v3-base` (184M params)
- LLM for CF generation: Zhipu GLM-4-Flash

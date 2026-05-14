# Codebase Audit

> Generated during refactoring. Provides a map of the entire codebase.

## Directory Structure

```
LLM-CLP/
├── src/llm_clp/           # New refactored package (canonical)
│   ├── data/              # Data loading (dataset, collators)
│   ├── counterfactual/     # CF generation (prompts, schema, validator)
│   ├── models/            # Model definitions + losses
│   ├── training/          # Training scripts
│   ├── evaluation/       # Metrics (CFR, CTFG, FPED, FNED)
│   └── utils/             # seed, logging, I/O
│
├── src_script/            # Legacy scripts (preserved for reproducibility)
│   ├── counterfactual/    # CF generators (llm, swap, validator)
│   ├── data/              # Data loaders + download
│   ├── eval/              # Evaluators
│   ├── train/             # 28+ training scripts
│   ├── utils/             # Utilities
│   └── viz/               # Visualization
│
├── src_model/             # Original model definitions
├── src_result/eval/       # 779 evaluation JSON result files
├── src_supplement/        # Supplementary experiments
├── tests/                 # pytest tests
├── docs/                  # Documentation
├── legacy/                # Unported legacy scripts
└── configs/              # YAML configs (to be created)
```

## Entry Points

### Counterfactual Generation

| Script | Backend | Purpose |
|--------|---------|---------|
| `src_script/counterfactual/cf_generator_llm.py` | Zhipu GLM / OpenAI | Main CF generator |
| `src_script/counterfactual/cf_generator_swap.py` | Lexical | Baseline swap |
| `src_script/counterfactual/cf_validator.py` | N/A | Quality filter |
| `src/llm_clp/counterfactual/generator.py` | (new) | Unified CF generator |

### Training

| Script | Method |
|--------|--------|
| `src_script/train/train_causal_fair.py` | LLM-CLP (main) |
| `src_script/train/train_baseline_davani.py` | LogitPairing (Davani) |
| `src_script/train/train_baseline_ear.py` | EAR |
| `src_script/train/train_baseline_getfair.py` | GetFair |
| `src_script/train/train_baseline_ccdf.py` | CCDF |
| `src_script/train/train_baseline_ramponi.py` | AdvDebias |
| `src/llm_clp/training/train_causal_fair.py` | (new) LLM-CLP |

### Evaluation

| Script | Metrics |
|--------|---------|
| `src_script/eval/eval_causal_fairness.py` | CFR, CTFG, FPED, FNED, Macro-F1, AUC |
| `src_script/eval/eval_backbone_baselines.py` | 7-metric evaluator |
| `src/llm_clp/evaluation/metrics.py` | (new) canonical metrics |

## Duplicate Code Found

### Gradient Reversal Layer — 3 implementations

| File | Class/Function |
|------|---------------|
| `src_model/model_deberta_v3_adversarial.py:9` | `GradientReversalFunction` + `GradientReversalLayer` |
| `src_model/model_backbone_baselines.py:249` | `GradientReversalFunction` |
| `src_script/train/train_baseline_ramponi.py:50` | Inline duplicate |

**Resolution**: Consolidated into `src/llm_clp/models/losses.py`.

### Weighted Toxicity Loss — 2 implementations

| File | Lines |
|------|-------|
| `src_script/train/train_deberta_v3_adv.py` | 92-110 |
| `src_script/train/train_deberta_v3_hcma.py` | 349-361 |

### DeBERTa-v3 Pooling — shared

`AttentionPooling` defined in `src_model/model_deberta_v3_mtl.py:5-19` and imported in adversarial/hcma models.

## Hardcoded Values

### API Keys (SECURITY)
- `src_script/counterfactual/cf_generator_llm.py:309` — Zhipu API key (FIXED: now uses env var)

### Hyperparameters (acceptable, all CLI-settable)
- `lambda_clp=1.0`, `lambda_con=0.5`, `temperature=0.07` in `train_causal_fair.py`
- `IDENTITY_GROUP_WEIGHTS` in `train_deberta_v3_adv.py:81`
- `soft_gate` params in `train_deberta_v3_hcma.py:297-304`

## Data Files Expected

| Dataset | Location | Format |
|---------|----------|--------|
| HateXplain | `data/causal_fair/hatexplain_*.parquet` | parquet |
| ToxiGen | `data/causal_fair/toxigen_*.parquet` | parquet |
| DynaHate | `data/causal_fair/dynahate_*.parquet` | parquet |
| CF (LLM) | `data/causal_fair/{dataset}_train_cf_llm.parquet` | parquet |
| CF (Swap) | `data/causal_fair/{dataset}_train_cf_swap.parquet` | parquet |

## Papers Referenced

- LLM-CLP (ours)
- Davani et al., 2021 — Counterfactual Logit Pairing
- Kennedy et al., 2022 — EAR (Entropy-based Attention Regularization)
- Qian et al., 2024 — GetFair
- Lu et al., 2024 — CCDF
- Zhang et al., 2018 — Adversarial Debiasing
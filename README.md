# LLM-CLP: Counterfactual Logit Pairing with LLM-Generated Counterfactuals

**EMNLP 2026 Submission**

A research library for identity-fair toxicity classification. We generate counterfactual texts by replacing identity references using a Large Language Model, then train classifiers with Counterfactual Logit Pairing (CLP) to achieve identity-invariant predictions.

## What This Repository Contains

- **Core method**: LLM-CLP — counterfactual data augmentation + logit pairing regularization
- **Models**: DeBERTa-v3-base with CLP + supervised contrastive loss
- **Datasets**: HateXplain, ToxiGen, DynaHate
- **Metrics**: CFR (Counterfactual Flip Rate), CTFG, FPED, FNED
- **Baselines**: Vanilla, EAR, GetFair, CCDF, AdvDebias, Davani LogitPairing

## Installation

```bash
git clone https://github.com/Ei-Ayw/LLM-CLP.git
cd LLM-CLP
pip install -e .

# Core dependencies
pip install torch transformers pandas numpy scikit-learn tqdm
# For counterfactual generation
pip install zai-sdk  # or: pip install zhipuai
# For local Qwen generation (optional)
pip install transformers accelerate bitsandbytes
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
ZHIPU_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

## Step1: Data Preparation

```bash
python -m src.llm_clp.data.prepare \
  --data data/causal_fair \
  --dataset hatexplain
```

Datasets are expected at `data/causal_fair/{dataset}_{split}.parquet`.

## Counterfactual Generation

```bash
# LLM-based counterfactual generation (requires API key)
python -m src.llm_clp.counterfactual.generate \
    --dataset hatexplain --split train --backend zhipu

# Swap-based baseline (no API needed)
python -m src.llm_clp.counterfactual.generate \
    --dataset hatexplain --split train --backend swap

# Local Qwen (no API needed)
python -m src.llm_clp.counterfactual.generate \
    --dataset hatexplain --split train --backend qwen \
    --model_name Qwen/Qwen2.5-7B-Instruct
```

## Step2: Training

```bash
python -m src.llm_clp.train.run \
  --data data/causal_fair \
  --model microsoft/deberta-v3-base \
  --method ours \
  --dataset hatexplain \
  --cf_path data/causal_fair/hatexplain_train_cf_llm.parquet \
  --epochs 5 --seed 42
```

## Step3: Evaluation

```bash
python -m src.llm_clp.eval.run \
  --data data/causal_fair \
  --model microsoft/deberta-v3-base \
  --method ours \
  --ckpt outputs/hatexplain_llmclp_s42.pth \
  --dataset hatexplain
```

Outputs: Macro-F1, AUC-ROC, CFR, CTFG, FPED, FNED.

## Step4: Visualization

```bash
python -m src.llm_clp.viz.run --type performance
python -m src.llm_clp.viz.run --type tsne --checkpoint outputs/model.pth
```

## Reproducing Paper Tables

All main results are in `EXPERIMENT_REPORT.md`. Evaluation JSON files are in `src_result/eval/`.

## Repository Structure

```
src/llm_clp/
├── data/            # Data loading (CausalFairDataset)
├── counterfactual/   # CF generation (prompts, schema, validator)
├── models/          # Model + losses (CLP, SupCon)
├── train/           # Training script (train_causal_fair.py)
├── eval/            # Metrics (CFR, CTFG, FPED, FNED)
└── utils/           # seed, logging, I/O

src_result/eval/    # Evaluation result JSON files
tests/              # Unit tests (pytest)
docs/               # Documentation
configs/            # YAML experiment configs

scripts/            # CLI entry points
legacy/             # Legacy unported scripts
```

## Running Tests

```bash
pytest tests/ -v
```

## Notes on Safety

This repository works with **toxic text** (hate speech datasets). Generated counterfactuals may also contain offensive content. Use only for research purposes.

## Citation

```bibtex
@article{llm_clp,
  title={LLM-CLP: Counterfactual Logit Pairing with LLM-Generated Counterfactuals},
  author={},
  journal={EMNLP 2026},
  year={2026}
}
```
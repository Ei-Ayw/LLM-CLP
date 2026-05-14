# Legacy Scripts

This directory contains original scripts preserved for reproducibility.

**Do not modify** — these scripts are the exact code that produced the paper results.

## Contents

### Baseline Training Scripts
- `train_baseline_vanilla.py` — Vanilla fine-tuning baseline
- `train_baseline_ear.py` — EAR (Entropy-based Attention Regularization)
- `train_baseline_getfair.py` — GetFair debiasing
- `train_baseline_ccdf.py` — CCDF (Counterfactual-guided Debiasing)
- `train_baseline_ramponi.py` — Adversarial Debiasing
- `train_baseline_davani.py` — Davani LogitPairing

### Evaluators
- `eval_backbone_baselines.py` — 7-metric evaluator for baseline models
- `eval_all_baselines.py` — All-baseline evaluator

### Visualization
- `viz_performance_summary.py` — Performance visualization
- `viz_feature_t_sne.py` — t-SNE feature visualization

## Migration Status

See `docs/MIGRATION_MAP.md` for the current migration status of all scripts.
New experiments should use the refactored code in `src/llm_clp/`.

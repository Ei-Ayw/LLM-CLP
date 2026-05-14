# Migration Map

> Tracks migration of legacy scripts to the new `src/llm_clp/` package.
> Status: `migrated` | `pending` | `legacy`

| Old Path | New Path | Status | Notes |
|----------|----------|--------|-------|
| `src_script/counterfactual/cf_generator_llm.py` | `src/llm_clp/counterfactual/generator.py` | migrated | CF generation with unified backend |
| `src_script/counterfactual/cf_generator_swap.py` | `src/llm_clp/counterfactual/prompts.py` | migrated | Identity swap logic extracted to prompts.py |
| `src_script/counterfactual/cf_validator.py` | `src/llm_clp/counterfactual/validator.py` | pending | Quality filtering |
| `src_script/utils/loss_contrastive.py` | `src/llm_clp/models/losses.py` | migrated | CLP + SupConLoss |
| `src_script/data/data_loader_cf.py` | `src/llm_clp/data/dataset.py` | migrated | CausalFairDataset + collate_fn |
| `src_script/eval/eval_causal_fairness.py` | `src/llm_clp/evaluation/metrics.py` | migrated | CFR, CTFG, FPED, FNED |
| `src_script/train/train_causal_fair.py` | `src/llm_clp/training/train_causal_fair.py` | migrated | Main training loop |
| `src_model/model_deberta_cf.py` | `src/llm_clp/models/classifier.py` | migrated | DebertaV3CausalFair |
| `src_script/train/train_baseline_davani.py` | `legacy/` | legacy | Davani LogitPairing baseline |
| `src_script/train/train_baseline_ear.py` | `legacy/` | legacy | EAR baseline |
| `src_script/train/train_baseline_getfair.py` | `legacy/` | legacy | GetFair baseline |
| `src_script/train/train_baseline_ccdf.py` | `legacy/` | legacy | CCDF baseline |
| `src_script/train/train_baseline_ramponi.py` | `legacy/` | legacy | AdvDebias baseline |
| `src_script/train/train_baseline_vanilla.py` | `legacy/` | legacy | Vanilla baseline |
| `src_script/eval/eval_backbone_baselines.py` | `legacy/` | legacy | 7-metric evaluator |
| `src_script/eval/eval_all_baselines.py` | `legacy/` | legacy | All-baseline evaluator |
| `src_script/viz/viz_performance_summary.py` | `legacy/` | legacy | Performance visualization |
| `src_script/viz/viz_feature_t_sne.py` | `legacy/` | legacy | t-SNE visualization |
| `src_model/model_deberta_v3_hcma.py` | `legacy/` | legacy | HCMA model (paper variant) |
| `src_model/model_deberta_v3_mtl.py` | `legacy/` | legacy | MTL model |
| `src_model/model_deberta_v3_adversarial.py` | `legacy/` | legacy | Adversarial debiasing model |
| `src_model/model_bert_cnn_bilstm.py` | `legacy/` | legacy | CNN-LSTM baseline |
| `src_model/model_backbone_baselines.py` | `legacy/` | legacy | All-in-one baseline models |

## How to Read This Map

- **migrated**: Code is available in `src/llm_clp/`. New experiments should use these.
- **pending**: Needs migration but not yet done. Use old path.
- **legacy**: Preserved for reproducibility. Contains the original experiments that produced paper results. Do not modify.

## Migration Notes

1. All CLI arguments are preserved across migrations
2. Output directory structure is unchanged (`src_result/eval/`, `src_result/models/`)
3. Data format (parquet) is unchanged
4. New package uses type hints throughout
5. Legacy scripts remain fully functional and produce identical results
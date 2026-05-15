"""
=============================================================================
统一评估脚本 (支持 BERT/RoBERTa/DeBERTa backbone 变体)
7 个指标: Macro-F1, AUC-ROC, CFR, CTFG, FPED, FNED, Per-group F1 Std

新增 method 前缀格式: {method}_{backbone}
  如: vanilla_bert, ours_roberta, ear_deberta, ccdf_bert, ...

骨干自动推断规则:
  - 若 --backbone 明确指定，使用该值
  - 否则从 --method 中解析后缀 (vanilla_bert → bert)
=============================================================================
"""
import os, sys, json, argparse, torch, torch.nn.functional as F
from transformers import AutoTokenizer
import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_clp.common.paths import eval_path

BACKBONE_PATHS = {
    'bert':    os.path.join(BASE_DIR, "models", "bert-base-uncased"),
    'roberta': os.path.join(BASE_DIR, "models", "roberta-base"),
    'deberta': os.path.join(BASE_DIR, "models", "deberta-v3-base"),
}


# =============================================================================
# 模型加载
# =============================================================================
def load_model(method, backbone, checkpoint, model_name, device):
    """
    method: vanilla / ear / getfair / ccdf / davani / ramponi / ours
    backbone: bert / roberta / deberta
    返回 (model, predict_fn)
    """
    from src.llm_clp.models.backbones.baseline_backbones import (
        BackboneVanilla, BackboneEAR, BackboneGetFair,
        BackboneCCDF, BiasOnlyModel, build_identity_mask,
        BackboneDavani, BackboneRamponi,
    )
    from src.llm_clp.models.backbones.causal_fair_backbone import BackboneCausalFair

    tokenizer_ref = AutoTokenizer.from_pretrained(model_name)

    if method == "vanilla":
        model = BackboneVanilla(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask)['logits']

    elif method == "ear":
        model = BackboneEAR(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask, return_attentions=False)['logits']

    elif method == "getfair":
        model = BackboneGetFair(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask, capture_embed_grad=False)['logits']

    elif method == "ccdf":
        ckpt = torch.load(checkpoint, map_location=device)
        model = BackboneCCDF(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(ckpt['main_model'])
        vocab_size = ckpt.get('vocab_size', tokenizer_ref.vocab_size)
        bias_model = BiasOnlyModel(vocab_size, embed_dim=128, num_classes=2).to(device)
        bias_model.load_state_dict(ckpt['bias_model'])
        bias_model.eval()
        tde_alpha  = ckpt.get('tde_alpha', 0.5)
        def predict_fn(m, ids, mask):
            main_logits = m(ids, mask)['logits']
            id_mask     = build_identity_mask(ids, tokenizer_ref).to(device)
            bias_logits = bias_model(ids, id_mask)
            return main_logits - tde_alpha * bias_logits

    elif method == "davani":
        model = BackboneDavani(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask)['logits']

    elif method == "ramponi":
        model = BackboneRamponi(model_name, backbone_type=backbone,
                                num_classes=2, num_identity_classes=2).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask, alpha=0.0)['logits']

    elif method == "ours":
        model = BackboneCausalFair(model_name, backbone_type=backbone).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        def predict_fn(m, ids, mask):
            return m(ids, mask, return_features=False)['logits']

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, predict_fn


# =============================================================================
# 批量预测
# =============================================================================
@torch.no_grad()
def predict_batch(model, predict_fn, tokenizer, texts, device, max_len=128, batch_size=32):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, max_length=max_len, padding='max_length',
                           truncation=True, return_tensors='pt')
        ids  = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            logits = predict_fn(model, ids, mask)
            probs  = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


# =============================================================================
# 公平性指标
# =============================================================================
def counterfactual_flip_rate(preds_orig, preds_cf):
    if len(preds_orig) == 0: return 0.0
    return float((preds_orig != preds_cf).sum()) / len(preds_orig)


def counterfactual_token_fairness_gap(probs_orig, probs_cf):
    if len(probs_orig) == 0: return 0.0
    return float(np.mean(np.abs(probs_orig - probs_cf)))


def compute_fped_fned(y_true, y_pred, groups):
    def fpr(yt, yp):
        denom = (yt == 0).sum()
        return float(((yp == 1) & (yt == 0)).sum()) / max(float(denom), 1)
    def fnr(yt, yp):
        denom = (yt == 1).sum()
        return float(((yp == 0) & (yt == 1)).sum()) / max(float(denom), 1)

    overall_fpr = fpr(y_true, y_pred)
    overall_fnr = fnr(y_true, y_pred)
    fped, fned  = 0.0, 0.0
    group_details = {}
    for g in sorted(set(groups)):
        mask = np.array(groups) == g
        if mask.sum() < 5: continue
        g_fpr = fpr(y_true[mask], y_pred[mask])
        g_fnr = fnr(y_true[mask], y_pred[mask])
        fped += abs(overall_fpr - g_fpr)
        fned += abs(overall_fnr - g_fnr)
        group_details[str(g)] = {
            'fpr': round(g_fpr, 4), 'fnr': round(g_fnr, 4),
            'fpr_gap': round(abs(overall_fpr - g_fpr), 4),
            'fnr_gap': round(abs(overall_fnr - g_fnr), 4),
            'count': int(mask.sum()),
        }
    return fped, fned, group_details


# =============================================================================
# 主评估
# =============================================================================
def evaluate_all(model, predict_fn, tokenizer, test_df, cf_test_df,
                 device, max_len=128, threshold=0.5):
    results = {}
    texts  = test_df['text'].tolist()
    labels = test_df['binary_label'].values

    # A. 任务性能
    print("\n[A] 任务性能...")
    probs = predict_batch(model, predict_fn, tokenizer, texts, device, max_len)
    preds = (probs >= threshold).astype(int)

    results['macro_f1']  = round(float(f1_score(labels, preds, average='macro')), 4)
    results['auc_roc']   = round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) > 1 else 0.5
    results['accuracy']  = round(float(accuracy_score(labels, preds)), 4)
    results['binary_f1'] = round(float(f1_score(labels, preds, average='binary')), 4)
    print(f"  Macro-F1={results['macro_f1']}  AUC={results['auc_roc']}")

    # B. 因果公平
    results['cfr']  = None
    results['ctfg'] = None
    if cf_test_df is not None and len(cf_test_df) > 0:
        print("[B] 因果公平...")
        probs_orig = predict_batch(model, predict_fn, tokenizer,
                                   cf_test_df['original_text'].tolist(), device, max_len)
        probs_cf   = predict_batch(model, predict_fn, tokenizer,
                                   cf_test_df['cf_text'].tolist(), device, max_len)
        preds_orig = (probs_orig >= threshold).astype(int)
        preds_cf   = (probs_cf   >= threshold).astype(int)
        results['cfr']  = round(counterfactual_flip_rate(preds_orig, preds_cf), 4)
        results['ctfg'] = round(counterfactual_token_fairness_gap(probs_orig, probs_cf), 4)
        print(f"  CFR={results['cfr']}  CTFG={results['ctfg']}")

    # C. 群体公平
    results['fped'] = None
    results['fned'] = None
    results['per_group_f1_std'] = None

    group_col = None
    if 'target_group' in test_df.columns:
        group_col = 'target_group'
    elif 'coarse_groups' in test_df.columns:
        test_df = test_df.copy()
        test_df['_primary_group'] = test_df['coarse_groups'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'none')
        group_col = '_primary_group'

    if group_col and test_df[group_col].nunique() > 1:
        print("[C] 群体公平...")
        groups = test_df[group_col].values
        fped, fned, group_details = compute_fped_fned(labels, preds, groups)
        results['fped'] = round(fped, 4)
        results['fned'] = round(fned, 4)
        results['group_details'] = group_details
        print(f"  FPED={results['fped']}  FNED={results['fned']}")

        per_group_f1 = {}
        for g in sorted(set(groups)):
            mask = np.array(groups) == g
            if mask.sum() >= 5 and len(set(labels[mask])) > 1:
                per_group_f1[str(g)] = float(f1_score(labels[mask], preds[mask], average='macro'))
        if per_group_f1:
            f1_vals = list(per_group_f1.values())
            results['per_group_f1_std'] = round(float(np.std(f1_vals)), 4)
            results['per_group_f1']     = {k: round(v, 4) for k, v in per_group_f1.items()}
            print(f"  Per-group F1 Std={results['per_group_f1_std']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backbone-aware 评估 (7指标)")
    parser.add_argument("--method",    type=str, required=True,
                        choices=["vanilla", "ear", "getfair", "ccdf", "davani", "ramponi", "ours"])
    parser.add_argument("--backbone",  type=str, required=True,
                        choices=["bert", "roberta", "deberta"])
    parser.add_argument("--checkpoint",type=str, required=True)
    parser.add_argument("--dataset",   type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--cf_method", type=str, default="llm",
                        choices=["swap", "llm"])
    parser.add_argument("--model_name",type=str, default=None,
                        help="覆盖自动backbone路径")
    parser.add_argument("--max_len",   type=int,   default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_dir",  type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--output",    type=str, default=None)
    args = parser.parse_args()

    model_name = args.model_name or BACKBONE_PATHS[args.backbone]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Config] method={args.method} backbone={args.backbone} dataset={args.dataset}")
    print(f"[Config] checkpoint={args.checkpoint}")

    model, predict_fn = load_model(args.method, args.backbone, args.checkpoint, model_name, device)
    tokenizer         = AutoTokenizer.from_pretrained(model_name)

    test_df   = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path   = os.path.join(args.data_dir, f"{args.dataset}_test_{cf_suffix}.parquet")
    cf_test_df = pd.read_parquet(cf_path) if os.path.exists(cf_path) else None
    if cf_test_df is not None:
        print(f"[CF] {len(cf_test_df)} 条反事实 ({args.cf_method})")

    results = evaluate_all(model, predict_fn, tokenizer, test_df, cf_test_df,
                           device, args.max_len, args.threshold)
    results['_meta'] = {
        'method': args.method, 'backbone': args.backbone,
        'dataset': args.dataset, 'cf_method': args.cf_method,
        'checkpoint': args.checkpoint,
    }

    if args.output:
        save_path = args.output
    else:
        ckpt_name  = os.path.basename(args.checkpoint).replace('.pth', '')
        save_path  = eval_path(f"{ckpt_name}_7metrics_{args.cf_method}.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  {args.method.upper()} ({args.backbone}) on {args.dataset} ({args.cf_method})")
    print(f"{'='*60}")
    print(f"  Macro-F1:  {results['macro_f1']}")
    print(f"  AUC-ROC:   {results['auc_roc']}")
    print(f"  CFR:       {results['cfr']}")
    print(f"  CTFG:      {results['ctfg']}")
    print(f"  FPED:      {results['fped']}")
    print(f"  FNED:      {results['fned']}")
    print(f"  F1-Std:    {results['per_group_f1_std']}")
    print(f"{'='*60}")
    print(f"保存至: {save_path}")


if __name__ == "__main__":
    main()

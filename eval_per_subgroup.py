"""
逐身份组评估: 打印 9 个身份组各自的 Subgroup/BPSN/BNSP AUC
找出最短的那块木板
"""
import os, sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path

IDENTITY_COLS = [
    'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
    'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
]

def power_mean(values, p=-5):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0: return np.nan
    arr = np.clip(arr, 1e-10, None)
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))

def evaluate(ckpt_path, label="Model"):
    device = torch.device("cuda:0")

    # 加载模型
    model = DebertaV3MTL().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    # 去掉 module. 前缀
    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace('module.', '')] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    # 加载数据
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=256)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    # 推理
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                out = model(ids, mask)
            probs = torch.sigmoid(out['logits_tox']).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch['y_tox'].numpy().flatten())

    probs_arr = np.array(all_probs)
    labels_binary = (np.array(all_labels) >= 0.5).astype(int)

    overall_auc = roc_auc_score(labels_binary, probs_arr)

    # 逐组计算
    eval_df = val_df.iloc[:len(probs_arr)].copy()
    eval_df['_prob'] = probs_arr
    target = (eval_df['target'] >= 0.5)

    results = []
    for col in IDENTITY_COLS:
        sub_mask = eval_df[col] >= 0.5
        n_samples = sub_mask.sum()
        n_toxic = (sub_mask & target).sum()
        n_nontoxic = (sub_mask & ~target).sum()

        row = {'identity': col, 'n_total': int(n_samples),
               'n_toxic': int(n_toxic), 'n_nontoxic': int(n_nontoxic)}

        # Subgroup AUC
        sub_df = eval_df[sub_mask]
        sub_target = target[sub_mask]
        if sub_target.nunique() >= 2:
            row['subgroup_auc'] = roc_auc_score(sub_target, sub_df['_prob'])
        else:
            row['subgroup_auc'] = np.nan

        # BPSN AUC
        bpsn_mask = (sub_mask & ~target) | (~sub_mask & target)
        bpsn_target = target[bpsn_mask]
        if bpsn_target.nunique() >= 2:
            row['bpsn_auc'] = roc_auc_score(bpsn_target, eval_df[bpsn_mask]['_prob'])
        else:
            row['bpsn_auc'] = np.nan

        # BNSP AUC
        bnsp_mask = (sub_mask & target) | (~sub_mask & ~target)
        bnsp_target = target[bnsp_mask]
        if bnsp_target.nunique() >= 2:
            row['bnsp_auc'] = roc_auc_score(bnsp_target, eval_df[bnsp_mask]['_prob'])
        else:
            row['bnsp_auc'] = np.nan

        results.append(row)

    df = pd.DataFrame(results)

    # 计算聚合指标
    pm_sub = power_mean(df['subgroup_auc'].values)
    pm_bpsn = power_mean(df['bpsn_auc'].values)
    pm_bnsp = power_mean(df['bnsp_auc'].values)
    bias_score = (pm_sub + pm_bpsn + pm_bnsp) / 3.0
    final_metric = 0.25 * overall_auc + 0.75 * bias_score

    # 打印结果
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Overall AUC: {overall_auc:.4f} | Final Metric: {final_metric:.4f}")
    print(f"  BiasScore: {bias_score:.4f} | PM(Sub): {pm_sub:.4f} | PM(BPSN): {pm_bpsn:.4f} | PM(BNSP): {pm_bnsp:.4f}")
    print(f"{'='*90}")

    print(f"\n{'Identity':<35s} {'N':>6s} {'Toxic':>6s} {'NonTox':>6s} | {'SubAUC':>8s} {'BPSN':>8s} {'BNSP':>8s}")
    print("-" * 90)

    for _, r in df.iterrows():
        sub = f"{r['subgroup_auc']:.4f}" if not np.isnan(r['subgroup_auc']) else "  N/A "
        bpsn = f"{r['bpsn_auc']:.4f}" if not np.isnan(r['bpsn_auc']) else "  N/A "
        bnsp = f"{r['bnsp_auc']:.4f}" if not np.isnan(r['bnsp_auc']) else "  N/A "
        print(f"{r['identity']:<35s} {r['n_total']:>6d} {r['n_toxic']:>6d} {r['n_nontoxic']:>6d} | {sub:>8s} {bpsn:>8s} {bnsp:>8s}")

    # 找出最短木板
    print(f"\n--- 最短木板 (Power Mean p=-5 最敏感) ---")
    worst_sub = df.loc[df['subgroup_auc'].idxmin()]
    worst_bpsn = df.loc[df['bpsn_auc'].idxmin()]
    worst_bnsp = df.loc[df['bnsp_auc'].idxmin()]
    print(f"  Subgroup AUC 最差: {worst_sub['identity']} = {worst_sub['subgroup_auc']:.4f}")
    print(f"  BPSN AUC 最差:    {worst_bpsn['identity']} = {worst_bpsn['bpsn_auc']:.4f}")
    print(f"  BNSP AUC 最差:    {worst_bnsp['identity']} = {worst_bnsp['bnsp_auc']:.4f}")

    return df

if __name__ == '__main__':
    # V3 best checkpoint
    v3_path = "src_result/models/DebertaV3Fair_S2_Fair_Seed42_Sample300000_0305_1905.pth"
    # V2 best checkpoint
    v2_path = "src_result/models/DebertaV3Fair_S2_Fair_Seed42_Sample300000_0305_1333.pth"

    if os.path.exists(v3_path):
        df_v3 = evaluate(v3_path, "Fair S2 V3 (Margin + Softmax Temperature)")
    if os.path.exists(v2_path):
        df_v2 = evaluate(v2_path, "Fair S2 V2 (Power Mean)")

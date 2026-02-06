"""
=============================================================================
### 评估脚本：eval_universal_runner.py ###
设计说明：
本脚本实现论文定义的完整三层评估指标体系：
A. 主任务分类指标：F1、Accuracy、PR-AUC (阈值通过 0.05-0.95 扫描获得最优 F1)
B. 偏见/群体鲁棒指标 (Nuanced Metrics by Borkan et al.):
   - Subgroup AUC
   - BPSN AUC (Background Positive, Subgroup Negative)
   - BNSP AUC (Background Negative, Subgroup Positive)
   - Mean Bias AUC (所有子群 x 3种AUC 的均值)
   - Worst-group Bias AUC (所有子群 x 3种AUC 的最小值)
=============================================================================
"""
import os
# [CRITICAL Fix] 必须在导入 torch 之前设置 OpenMP 线程数，否则与 sklearn 冲突必崩(SIGSEGV)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import pickle

# [Fix] 防止服务器无GUI环境下的 SegFault
import matplotlib
matplotlib.use('Agg')

# [Fix] 防止 Tokenizer 多线程与 DataLoader fork 冲突导致 SegFault
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================= 项目路径配置 =======================
# 关键修复：从深层目录退回到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

# 离线环境变量
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 导入模型与数据集定义
from model_deberta_v3_mtl import DebertaV3MTL
from model_bert_cnn_bilstm import BertCNNBiLSTM
# from model_text_cnn import TextCNN
# from model_bilstm import BiLSTM
from model_vanilla_bert import VanillaBERT
from model_vanilla_roberta import VanillaRoBERTa
from model_vanilla_deberta_v3 import VanillaDeBERTaV3
from exp_data_loader import ToxicityDataset
from path_config import get_eval_path

# ======================= 辅助 Dataset =======================
class SimpleTokenDataset(Dataset):
    """用于非 Transformer 模型的简易分词 Dataset"""
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return {'ids': torch.tensor(ids, dtype=torch.long), 'y_tox': torch.tensor(self.labels[idx], dtype=torch.float)}

# ======================= 指标计算函数 =======================
def calculate_roc_auc(y_true, y_prob):
    """计算 ROC-AUC，处理边界情况"""
    try: return metrics.roc_auc_score(y_true, y_prob)
    except: return np.nan

def calculate_pr_auc(y_true, y_prob):
    """计算 Precision-Recall AUC (PR-AUC)"""
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    return metrics.auc(recall, precision)

def scan_thresholds(y_true, y_prob):
    """
    阈值扫描策略：在 0.05 - 0.95 范围内搜索使 F1 最大的阈值。
    论文要求：避免固定 0.5 带来的不公平比较。
    返回：最优阈值、最优F1、完整扫描记录(用于可视化)
    """
    best_f1, best_thresh = 0, 0.5
    scan_history = []  # 记录完整扫描过程
    
    for thresh in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        
        scan_history.append({
            'threshold': float(thresh),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1, scan_history

def calculate_fairness_metrics(df, subgroups, model_col):
    """
    计算 Nuanced Metrics。
    对于每个身份子群 g，计算：
    1. Subgroup AUC: 仅在提到该身份的子集上计算 AUC。
    2. BPSN AUC: (Subgroup Negative, Background Positive) 混合集合上的 AUC。
    3. BNSP AUC: (Subgroup Positive, Background Negative) 混合集合上的 AUC。
    """
    records = []
    for subgroup in subgroups:
        sub_mask = df[subgroup] >= 0.5
        if sub_mask.sum() == 0: continue
        
        # Subgroup AUC
        sub_df = df[sub_mask]
        subgroup_auc = calculate_roc_auc(sub_df['target'] >= 0.5, sub_df[model_col])
        
        # BPSN AUC: (Subgroup 中无毒) U (Background 中有毒)
        bpsn_mask = ((df[subgroup] >= 0.5) & (df['target'] < 0.5)) | \
                    ((df[subgroup] < 0.5) & (df['target'] >= 0.5))
        bpsn_df = df[bpsn_mask]
        bpsn_auc = calculate_roc_auc(bpsn_df['target'] >= 0.5, bpsn_df[model_col])
        
        # BNSP AUC: (Subgroup 中有毒) U (Background 中无毒)
        bnsp_mask = ((df[subgroup] >= 0.5) & (df['target'] >= 0.5)) | \
                    ((df[subgroup] < 0.5) & (df['target'] < 0.5))
        bnsp_df = df[bnsp_mask]
        bnsp_auc = calculate_roc_auc(bnsp_df['target'] >= 0.5, bnsp_df[model_col])
        
        records.append({
            'subgroup': subgroup,
            'subgroup_auc': subgroup_auc,
            'bpsn_auc': bpsn_auc,
            'bnsp_auc': bnsp_auc
        })
    return pd.DataFrame(records)

# ======================= 主函数 =======================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nuanced Metrics Evaluation Runner")
    parser.add_argument("--checkpoint", type=str, required=True, help="待评估的权重文件路径")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["deberta_mtl", "bert_cnn", "text_cnn", "bilstm", 
                                 "vanilla_bert", "vanilla_roberta", "vanilla_deberta"],
                        help="模型类型标识")
    parser.add_argument("--output_prefix", type=str, default="eval_report", help="输出报告前缀")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [Fix] 显式清理显存碎片，防止 OOM 或 CUDA error
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # [1] 数据加载 -> 使用独立测试集 (学术严谨性)
    test_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "test_processed.parquet"))
    ckpt_name = os.path.basename(args.checkpoint)
    print(f"\n>>> 启动评估: {ckpt_name}")
    print(f">>> 模型类型: {args.model_type} | 测试集大小: {len(test_df)} | 设备: {device}")

    # [2] 模型实例化：根据类型和消融后缀自动适配
    if args.model_type == "deberta_mtl":
        use_pool = "NoPooling" not in ckpt_name
        model = DebertaV3MTL(use_attention_pooling=use_pool).to(device)
    elif args.model_type == "bert_cnn":
        model = BertCNNBiLSTM().to(device)
    elif args.model_type == "vanilla_bert":
        model = VanillaBERT().to(device)
    elif args.model_type == "vanilla_roberta":
        model = VanillaRoBERTa().to(device)
    elif args.model_type == "vanilla_deberta":
        model = VanillaDeBERTaV3().to(device)
    # elif args.model_type in ["text_cnn", "bilstm"]:
    #     vocab_path = args.checkpoint.replace(".pth", "_vocab.pkl")
    #     with open(vocab_path, 'rb') as f:
    #         vocab = pickle.load(f)
    #     if args.model_type == "text_cnn":
    #         model = TextCNN(vocab_size=len(vocab.stoi)).to(device)
    #     else:
    #         model = BiLSTM(vocab_size=len(vocab.stoi)).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # [DEBUG] 验证模型权重加载成功并进行快速预测测试
    if args.model_type == "vanilla_deberta":
        print(f">>> [DEBUG] 模型 classifier 权重 sum: {model.model.classifier.weight.sum().item():.4f}")
        # 快速预测测试
        test_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        test_texts = ["I love you!", "You are an idiot and should die!"]
        for txt in test_texts:
            inp = test_tok(txt, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            with torch.no_grad():
                out = model(inp['input_ids'].to(device), inp['attention_mask'].to(device))
                prob = torch.sigmoid(out['logits_tox']).item()
            print(f">>> [DEBUG] '{txt[:30]}...' -> {prob:.4f}")

    # [3] 推理
    probs, targets = [], []
    # if args.model_type in ["text_cnn", "bilstm"]:
    #     loader = DataLoader(SimpleTokenDataset(test_df['comment_text'].values, test_df['y_tox'].values, vocab), batch_size=64, shuffle=False)
    #     with torch.no_grad():
    #         for batch in tqdm(loader, desc="[Inference Classic]"):
    #             out = model(batch['ids'].to(device))
    #             probs.extend(torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy())
    #             targets.extend(batch['y_tox'].cpu().numpy())
    # else:
    # [Fix] 优先加载与权重文件匹配的本地 Tokenizer
    local_tokenizer_path = args.checkpoint.replace(".pth", "_tokenizer")
    if os.path.exists(local_tokenizer_path):
        print(f">>> [Load] 发现配套 Tokenizer: {local_tokenizer_path}")
        # [Fix] 强制使用 Slow Tokenizer (Python实现)，避免 C++/Rust 层面的 SIGSEGV
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_fast=False)
    else:
        base_model_name = "microsoft/deberta-v3-base" if "Deberta" in ckpt_name else \
                        "roberta-base" if "RoBERTa" in ckpt_name else "bert-base-uncased"
        # [Fix] 移除 local_files_only=True，它可能导致加载错误的 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    loader = DataLoader(ToxicityDataset(test_df, tokenizer), batch_size=16, shuffle=False)
    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Inference Transformer]"):
            out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            batch_probs = torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy()
            probs.extend(batch_probs)
            targets.extend(batch['y_tox'].cpu().numpy())
            
            # [DEBUG] 打印第一批的详细信息
            if batch_idx == 0 and args.model_type == "vanilla_deberta":
                print(f">>> [DEBUG BATCH0] input_ids shape: {batch['input_ids'].shape}")
                print(f">>> [DEBUG BATCH0] 第一个样本 input_ids[:20]: {batch['input_ids'][0][:20].tolist()}")
                print(f">>> [DEBUG BATCH0] logits_tox shape: {out['logits_tox'].shape}")
                print(f">>> [DEBUG BATCH0] batch_probs: {batch_probs}")
                print(f">>> [DEBUG BATCH0] y_tox: {batch['y_tox'].numpy()}")
                # 对比：用 DEBUG test 的方式重新推理第一个样本
                first_text = test_df['comment_text'].iloc[0]
                test_inp = tokenizer(first_text, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
                test_out = model(test_inp['input_ids'].to(device), test_inp['attention_mask'].to(device))
                test_prob = torch.sigmoid(test_out['logits_tox']).item()
                print(f">>> [DEBUG BATCH0] 重新推理第一个样本: '{first_text[:40]}...' -> {test_prob:.4f}")
            batch_idx += 1

    probs = np.array(probs)
    targets = np.array(targets)
    
    # [DEBUG] 打印预测分布和前几个样本
    if args.model_type == "vanilla_deberta":
        print(f">>> [DEBUG] probs 统计: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        print(f">>> [DEBUG] targets 统计: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
        print(f">>> [DEBUG] 前10个预测值: {probs[:10]}")
        print(f">>> [DEBUG] 前10个目标值: {targets[:10]}")
    
    test_df['model_probs'] = probs
    y_true_binary = (targets >= 0.5).astype(int)

    # ======================= [4] 指标计算 =======================
    # A. 主任务分类指标 (含完整阈值扫描记录)
    best_thresh, best_f1, scan_history = scan_thresholds(y_true_binary, probs)
    y_pred_binary = (probs >= best_thresh).astype(int)
    accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
    roc_auc = calculate_roc_auc(y_true_binary, probs)
    pr_auc = calculate_pr_auc(y_true_binary, probs)
    
    # A2. 固定阈值 0.5 的指标 (用于公平对比)
    y_pred_fixed = (probs >= 0.5).astype(int)
    fixed_f1 = metrics.f1_score(y_true_binary, y_pred_fixed)
    fixed_accuracy = metrics.accuracy_score(y_true_binary, y_pred_fixed)
    fixed_precision = metrics.precision_score(y_true_binary, y_pred_fixed, zero_division=0)
    fixed_recall = metrics.recall_score(y_true_binary, y_pred_fixed, zero_division=0)
    
    # B. 偏见/群体鲁棒指标
    identity_cols = ['male', 'female', 'black', 'white', 'muslim', 'jewish', 
                     'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness']
    bias_df = calculate_fairness_metrics(test_df, identity_cols, 'model_probs')
    
    # Mean Bias AUC: 对所有子群的 3 种 AUC 求均值
    all_bias_auc_values = bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.flatten()
    mean_bias_auc = float(np.nanmean(all_bias_auc_values))
    
    # Worst-group Bias AUC: 取所有子群 x 3种 AUC 中的最小值
    worst_bias_auc = float(np.nanmin(all_bias_auc_values))

    # ======================= [5] 生成阈值扫描曲线图 =======================
    import matplotlib.pyplot as plt
    
    thresholds = [s['threshold'] for s in scan_history]
    f1_scores = [s['f1'] for s in scan_history]
    precisions = [s['precision'] for s in scan_history]
    recalls = [s['recall'] for s in scan_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-o', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'g--s', label='Precision', linewidth=1.5, alpha=0.7)
    plt.plot(thresholds, recalls, 'r--^', label='Recall', linewidth=1.5, alpha=0.7)
    plt.axvline(x=best_thresh, color='orange', linestyle=':', linewidth=2, label=f'Best Threshold ({best_thresh:.2f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Threshold Scan Curve: {ckpt_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    thresh_plot_path = get_eval_path(f"{args.output_prefix}_threshold_scan.png")
    plt.savefig(thresh_plot_path, dpi=150)
    plt.close()

    # ======================= [6] 持久化结果报告 =======================
    report = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "optimal_threshold": float(best_thresh),
        # A. 主任务指标 (最优阈值)
        "primary_metrics_optimal": {
            "threshold": float(best_thresh),
            "f1": float(best_f1),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        },
        # A2. 主任务指标 (固定阈值 0.5，用于公平对比)
        "primary_metrics_fixed_0.5": {
            "threshold": 0.5,
            "f1": float(fixed_f1),
            "accuracy": float(fixed_accuracy),
            "precision": float(fixed_precision),
            "recall": float(fixed_recall)
        },
        # B. 偏见指标
        "bias_metrics": {
            "mean_bias_auc": mean_bias_auc,
            "worst_group_bias_auc": worst_bias_auc,
            "per_subgroup_details": bias_df.to_dict(orient='records')
        },
        # C. 完整阈值扫描记录
        "threshold_scan_history": scan_history
    }
    
    output_path = get_eval_path(f"{args.output_prefix}_metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f">>> 评估完成: {ckpt_name}")
    print(f">>> [最优阈值 {best_thresh:.2f}] F1: {best_f1:.4f} | Acc: {accuracy:.4f}")
    print(f">>> [固定阈值 0.50] F1: {fixed_f1:.4f} | Acc: {fixed_accuracy:.4f}")
    print(f">>> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f">>> Mean Bias AUC: {mean_bias_auc:.4f} | Worst-group: {worst_bias_auc:.4f}")
    print(f">>> 报告: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


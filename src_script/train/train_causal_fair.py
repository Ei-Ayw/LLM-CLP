"""
=============================================================================
主训练脚本: LLM 反事实增强 + 因果公平训练
L_total = L_CE + λ_clp × L_CLP + λ_con × L_SupCon
=============================================================================
"""
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_cf import DebertaV3CausalFair
from data_loader_cf import CausalFairDataset, get_causal_fair_loader
from loss_contrastive import CounterfactualSupConLoss, CounterfactualLogitPairing
from train_utils import EarlyStopping
from path_config import get_model_path, get_log_path


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_clp = CounterfactualLogitPairing()
    criterion_con = CounterfactualSupConLoss(temperature=args.temperature)

    total_loss = 0
    total_ce = 0
    total_clp = 0
    total_con = 0
    n_batches = 0

    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        # 原始样本
        orig_ids = batch['orig_input_ids'].to(device)
        orig_mask = batch['orig_attention_mask'].to(device)
        # 反事实样本
        cf_ids = batch['cf_input_ids'].to(device)
        cf_mask = batch['cf_attention_mask'].to(device)
        labels = batch['label'].to(device)
        has_cf = batch['has_cf'].to(device)

        with torch.cuda.amp.autocast():
            # Forward: 原始
            out_orig = model(orig_ids, orig_mask, return_features=True)
            # Forward: 反事实
            out_cf = model(cf_ids, cf_mask, return_features=True)

            # 1. 分类损失 (只在原始样本上)
            l_ce = criterion_ce(out_orig['logits'], labels)

            # 2. CLP: logit 配对损失 (只对有反事实的样本)
            if has_cf.sum() > 0:
                cf_mask_bool = has_cf.bool()
                l_clp = criterion_clp(
                    out_orig['logits'][cf_mask_bool],
                    out_cf['logits'][cf_mask_bool]
                )
            else:
                l_clp = torch.tensor(0.0, device=device)

            # 3. SupCon: 对比学习 (只对有反事实的样本)
            if has_cf.sum() > 1:
                cf_mask_bool = has_cf.bool()
                l_con = criterion_con(
                    out_orig['features'][cf_mask_bool],
                    out_cf['features'][cf_mask_bool],
                    labels[cf_mask_bool]
                )
            else:
                l_con = torch.tensor(0.0, device=device)

            # 总损失
            loss = l_ce + args.lambda_clp * l_clp + args.lambda_con * l_con

        loss_scaled = loss / args.grad_accum
        scaler.scale(loss_scaled).backward()

        if (i + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_ce += l_ce.item()
        total_clp += l_clp.item()
        total_con += l_con.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss/n_batches:.4f}",
            ce=f"{total_ce/n_batches:.4f}",
            clp=f"{total_clp/n_batches:.4f}",
            con=f"{total_con/n_batches:.4f}",
        )

    return {
        'loss': total_loss / n_batches,
        'ce': total_ce / n_batches,
        'clp': total_clp / n_batches,
        'con': total_con / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="[Eval]"):
        orig_ids = batch['orig_input_ids'].to(device)
        orig_mask = batch['orig_attention_mask'].to(device)
        labels = batch['label']

        with torch.cuda.amp.autocast():
            out = model(orig_ids, orig_mask, return_features=False)
            probs = F.softmax(out['logits'], dim=-1)[:, 1]  # P(toxic)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels_arr, preds_arr),
        'macro_f1': f1_score(labels_arr, preds_arr, average='macro'),
        'binary_f1': f1_score(labels_arr, preds_arr, average='binary'),
        'auc_roc': roc_auc_score(labels_arr, probs_arr) if len(set(labels_arr)) > 1 else 0.5,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="因果公平训练")
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen"])
    parser.add_argument("--cf_method", type=str, default="llm",
                        choices=["none", "swap", "llm"],
                        help="反事实方法: none=baseline, swap=传统换词, llm=LLM生成")
    parser.add_argument("--model_name", type=str,
                        default=os.path.join(BASE_DIR, "models", "deberta-v3-base"))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # 因果公平超参
    parser.add_argument("--lambda_clp", type=float, default=1.0,
                        help="CLP 损失权重")
    parser.add_argument("--lambda_con", type=float, default=0.5,
                        help="SupCon 损失权重")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="对比学习温度")

    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 实验名
    exp_name = (f"{args.dataset}_{args.cf_method}_"
                f"clp{args.lambda_clp}_con{args.lambda_con}_"
                f"seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}")
    print(f"[Experiment] {exp_name}")

    # ============= 数据 =============
    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 加载反事实
    cf_df = None
    if args.cf_method != "none":
        cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
        cf_path = os.path.join(args.data_dir, f"{args.dataset}_train_{cf_suffix}.parquet")
        if os.path.exists(cf_path):
            cf_df = pd.read_parquet(cf_path)
            print(f"[CF] 加载 {len(cf_df)} 条反事实 ({args.cf_method})")
        else:
            print(f"[Warning] 反事实文件不存在: {cf_path}")
            print(f"  请先运行反事实生成脚本")
            if args.cf_method == "llm":
                print(f"  python src_script/counterfactual/cf_generator_llm.py --api zhipu --api_key YOUR_KEY")
            return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader = get_causal_fair_loader(
        train_df, cf_df, tokenizer,
        batch_size=args.batch_size, max_len=args.max_len, shuffle=True,
    )
    val_loader = get_causal_fair_loader(
        val_df, None, tokenizer,
        batch_size=args.batch_size * 2, max_len=args.max_len, shuffle=False,
    )
    test_loader = get_causal_fair_loader(
        test_df, None, tokenizer,
        batch_size=args.batch_size * 2, max_len=args.max_len, shuffle=False,
    )

    # ============= 模型 =============
    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    print(f"[Model] {args.model_name} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # ============= 优化器 =============
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    early_stopping = EarlyStopping(patience=args.patience)

    # ============= 训练 =============
    best_f1 = 0
    history = {'train': [], 'val': []}
    save_path = get_model_path(f"{exp_name}.pth")

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args
        )
        val_metrics = evaluate(model, val_loader, device)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        print(f"  Train: loss={train_metrics['loss']:.4f} "
              f"CE={train_metrics['ce']:.4f} "
              f"CLP={train_metrics['clp']:.4f} "
              f"Con={train_metrics['con']:.4f}")
        print(f"  Val:   F1={val_metrics['macro_f1']:.4f} "
              f"AUC={val_metrics['auc_roc']:.4f} "
              f"Acc={val_metrics['accuracy']:.4f}")

        # 保存最优
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] Best F1={best_f1:.4f} → {save_path}")

        # 早停
        if early_stopping(-val_metrics['macro_f1']):
            print(f">>> [Early Stop] Best F1={best_f1:.4f}")
            break

    # ============= 测试 =============
    print(f"\n{'='*60}")
    print(f"加载最优模型，在测试集上评估...")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"  Test: F1={test_metrics['macro_f1']:.4f} "
          f"AUC={test_metrics['auc_roc']:.4f} "
          f"Acc={test_metrics['accuracy']:.4f}")

    # ============= 保存结果 =============
    results = {
        'experiment': exp_name,
        'args': vars(args),
        'best_val_f1': best_f1,
        'test_metrics': test_metrics,
        'history': history,
    }
    results_path = get_log_path(f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果保存至: {results_path}")

    # 提示下一步
    print(f"\n>>> 下一步: 运行因果公平评估")
    print(f"    python src_script/eval/eval_causal_fairness.py "
          f"--checkpoint {save_path} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

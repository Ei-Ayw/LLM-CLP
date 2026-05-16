"""
LLM-CLP（反事实 Logit Pairing）主训练脚本。

用法:
    python -m src.train.train_causal_fair \\
        --dataset hatexplain \\
        --cf_path data/hatexplain_train_cf_llm.parquet \\
        --model_name microsoft/deberta-v3-base \\
        --epochs 5 --seed 42
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

from src.data.dataset import get_causal_fair_loader
from src.models.losses import CounterfactualLogitPairing, CounterfactualSupConLoss
from src.models.classifier import DebertaV3CausalFair
from src.eval.metrics import evaluate_causal_fairness
from src.utils.seed import set_seed
from src.utils.io import ensure_dir


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = -float("inf")

    def __call__(self, value: float) -> bool:
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    lambda_clp: float,
    lambda_con: float,
    temperature: float,
    grad_accum: int,
) -> Dict[str, float]:
    """训练一个 epoch。

    损失函数：L_CE(x, y) + λ_clp * L_CLP(x, x_cf) + λ_con * L_SupCon(z, z_cf, y)
    """
    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_clp = CounterfactualLogitPairing()
    criterion_con = CounterfactualSupConLoss(temperature=temperature)

    total_loss, total_ce, total_clp, total_con = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        orig_ids = batch["orig_input_ids"].to(device)
        orig_mask = batch["orig_attention_mask"].to(device)
        cf_ids = batch.get("cf_input_ids")
        cf_mask = batch.get("cf_attention_mask")
        labels = batch["label"].to(device)
        has_cf = batch["has_cf"].to(device)

        with torch.cuda.amp.autocast():
            out_orig = model(orig_ids, orig_mask, return_features=True)
            l_ce = criterion_ce(out_orig["logits"], labels)

            l_clp = torch.tensor(0.0, device=device)
            if has_cf.sum() > 0 and cf_ids is not None:
                cf_mask_bool = has_cf.bool()
                out_cf = model(
                    cf_ids[cf_mask_bool.nonzero().squeeze()],
                    cf_mask[cf_mask_bool.nonzero().squeeze()],
                    return_features=True,
                )
                matched_orig = out_orig["logits"][cf_mask_bool]
                l_clp = criterion_clp(matched_orig, out_cf["logits"])

            l_con = torch.tensor(0.0, device=device)
            if has_cf.sum() > 1 and cf_ids is not None:
                cf_mask_bool = has_cf.bool()
                cf_indices = cf_mask_bool.nonzero().squeeze()
                out_cf = model(
                    cf_ids[cf_indices], cf_mask[cf_indices], return_features=True
                )
                l_con = criterion_con(
                    out_orig["features"][cf_mask_bool],
                    out_cf["features"],
                    labels[cf_mask_bool],
                )

            loss = l_ce + lambda_clp * l_clp + lambda_con * l_con

        scaler.scale(loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        )

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "clp": total_clp / n_batches,
        "con": total_con / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    """在数据集上评估模型性能。"""
    model.eval()
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="[Eval]"):
        orig_ids = batch["orig_input_ids"].to(device)
        orig_mask = batch["orig_attention_mask"].to(device)
        labels = batch["label"]

        with torch.cuda.amp.autocast():
            out = model(orig_ids, orig_mask, return_features=False)
            probs = F.softmax(out["logits"], dim=-1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "macro_f1": float(f1_score(labels_arr, preds_arr, average="macro")),
        "binary_f1": float(f1_score(labels_arr, preds_arr, average="binary")),
        "auc_roc": float(roc_auc_score(labels_arr, probs_arr))
        if len(set(labels_arr)) > 1
        else 0.5,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM-CLP 训练：反事实 Logit Pairing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据参数
    parser.add_argument("--dataset", required=True, choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--cf_path", type=str, default=None,
                        help="反事实 parquet 文件路径")
    parser.add_argument("--data_dir", type=str,
                        default=str(BASE_DIR / "data" / "causal_fair"))
    # 模型参数
    parser.add_argument("--model_name", type=str,
                        default=str(BASE_DIR / "models" / "deberta-v3-base"))
    # 训练超参数
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    # CLP 相关参数
    parser.add_argument("--lambda_clp", type=float, default=1.0,
                        help="CLP 损失权重")
    parser.add_argument("--lambda_con", type=float, default=0.5,
                        help="SupCon 损失权重")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="SupCon 温度参数")
    # 输出目录
    parser.add_argument("--output_dir", type=str,
                        default=str(BASE_DIR / "outputs"))

    args = parser.parse_args()

    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    print(f"[设备] {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    exp_name = f"{args.dataset}_llmclp_s{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[实验] {exp_name}")

    # 加载数据
    train_df = pd.read_parquet(Path(args.data_dir) / f"{args.dataset}_train.parquet")
    val_df = pd.read_parquet(Path(args.data_dir) / f"{args.dataset}_val.parquet")
    test_df = pd.read_parquet(Path(args.data_dir) / f"{args.dataset}_test.parquet")

    cf_df = None
    if args.cf_path and os.path.exists(args.cf_path):
        cf_df = pd.read_parquet(args.cf_path)
        print(f"[CF] 加载 {len(cf_df)} 条反事实配对")
    else:
        print("[CF] 未提供反事实数据，跳过 CLP 损失")

    print(f"[数据] 训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

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

    # 模型初始化
    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    print(f"[模型] {args.model_name} | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器配置
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    total_steps = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    early_stopping = EarlyStopping(patience=args.patience)

    # 训练循环
    best_f1 = 0.0
    save_path = Path(args.output_dir) / f"{exp_name}.pth"
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            lambda_clp=args.lambda_clp,
            lambda_con=args.lambda_con,
            temperature=args.temperature,
            grad_accum=args.grad_accum,
        )
        val_metrics = evaluate(model, val_loader, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"  训练: loss={train_metrics['loss']:.4f} "
              f"CE={train_metrics['ce']:.4f} CLP={train_metrics['clp']:.4f}")
        print(f"  验证: F1={val_metrics['macro_f1']:.4f} "
              f"AUC={val_metrics['auc_roc']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), save_path)
            print(f"  [保存] 最佳 F1={best_f1:.4f}")

        if early_stopping(val_metrics["macro_f1"]):
            print(f">>> [早停] 最佳 F1={best_f1:.4f}")
            break

    # 测试阶段
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n>>> [测试] F1={test_metrics['macro_f1']:.4f} "
          f"AUC={test_metrics['auc_roc']:.4f}")

    # 保存结果
    results = {
        "experiment": exp_name,
        "args": vars(args),
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    results_path = Path(args.output_dir) / f"{exp_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n结果: {results_path}")
    print(f"模型: {save_path}")


if __name__ == "__main__":
    main()
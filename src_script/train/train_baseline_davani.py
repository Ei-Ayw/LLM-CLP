"""
=============================================================================
Baseline: Davani et al., 2021 - Counterfactual Logit Pairing
论文: "Dealing with Disagreements: Looking Beyond the Majority Vote
       in Subjective Annotations"
链接: https://aclanthology.org/2021.woah-1.10.pdf

核心思想:
1. 使用反事实数据对 (原始文本, 反事实文本)
2. Logit Pairing: 强制模型对原始和反事实产生相似的 logits
3. L_total = L_CE(orig) + L_CE(cf) + λ_lp * MSE(logits_orig, logits_cf)

与我们方法的区别: Davani 只用 logit pairing (MSE)，不用对比学习
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Config, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))
from train_utils import EarlyStopping
from path_config import get_model_path, get_log_path


class DebertaV3Davani(nn.Module):
    """DeBERTa-V3 with Logit Pairing"""
    def __init__(self, model_path, num_classes=2):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_hidden))
        return {"logits": logits}


class CounterfactualPairDataset(Dataset):
    """训练集: 原始文本 + 反事实文本配对"""
    def __init__(self, train_df, cf_df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 原始训练数据
        self.orig_texts = train_df['text'].values
        self.orig_labels = train_df['binary_label'].values

        # 构建 post_id -> cf 映射 (每个原始样本可能有多个反事实)
        self.cf_map = {}
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                orig = row['original_text']
                if orig not in self.cf_map:
                    self.cf_map[orig] = []
                self.cf_map[orig].append(row['cf_text'])

    def __len__(self):
        return len(self.orig_texts)

    def __getitem__(self, idx):
        text = str(self.orig_texts[idx])
        label = self.orig_labels[idx]

        enc = self.tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True,
            return_attention_mask=True,
        )

        item = {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'has_cf': torch.tensor(0, dtype=torch.long),
        }

        # 随机选一个反事实
        if text in self.cf_map and len(self.cf_map[text]) > 0:
            cf_text = np.random.choice(self.cf_map[text])
            cf_enc = self.tokenizer.encode_plus(
                str(cf_text), add_special_tokens=True,
                max_length=self.max_len, padding='max_length', truncation=True,
                return_attention_mask=True,
            )
            item['cf_input_ids'] = torch.tensor(cf_enc['input_ids'], dtype=torch.long)
            item['cf_attention_mask'] = torch.tensor(cf_enc['attention_mask'], dtype=torch.long)
            item['has_cf'] = torch.tensor(1, dtype=torch.long)

        return item


def cf_collate_fn(batch):
    """自定义 collate: 处理有/无反事实的混合 batch"""
    keys = ['input_ids', 'attention_mask', 'label', 'has_cf']
    result = {k: torch.stack([b[k] for b in batch]) for k in keys}

    # 只对有反事实的样本堆叠 cf 字段
    cf_items = [b for b in batch if b['has_cf'].item() == 1]
    if cf_items:
        result['cf_input_ids'] = torch.stack([b['cf_input_ids'] for b in cf_items])
        result['cf_attention_mask'] = torch.stack([b['cf_attention_mask'] for b in cf_items])
        result['cf_indices'] = torch.tensor(
            [i for i, b in enumerate(batch) if b['has_cf'].item() == 1], dtype=torch.long
        )
    return result


class SimpleTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text'].values
        self.labels = df['binary_label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            str(self.texts[idx]), add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True,
            return_attention_mask=True,
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    total_loss, total_ce, total_lp, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            l_ce = args._criterion_ce(out['logits'], labels)

            # Logit Pairing
            l_lp = torch.tensor(0.0, device=device)
            if 'cf_input_ids' in batch and args.lambda_lp > 0:
                cf_ids = batch['cf_input_ids'].to(device)
                cf_mask = batch['cf_attention_mask'].to(device)
                cf_indices = batch['cf_indices'].to(device)

                cf_out = model(cf_ids, cf_mask)
                # MSE between original logits (at cf positions) and cf logits
                orig_logits_matched = out['logits'][cf_indices]
                l_lp = F.mse_loss(orig_logits_matched, cf_out['logits'])

                # 反事实样本也参与 CE (标签与原始相同，因为只换了身份词)
                cf_labels = labels[cf_indices]
                l_ce_cf = args._criterion_ce(cf_out['logits'], cf_labels)
                l_ce = (l_ce + l_ce_cf) / 2

            loss = l_ce + args.lambda_lp * l_lp

        loss_scaled = loss / args.grad_accum
        scaler.scale(loss_scaled).backward()

        if (i + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_ce += l_ce.item()
        total_lp += l_lp.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", ce=f"{total_ce/n:.4f}", lp=f"{total_lp/n:.4f}")

    return {'loss': total_loss/n, 'ce': total_ce/n, 'lp': total_lp/n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval]"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            probs = F.softmax(out['logits'], dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch['label'].numpy())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(labels_arr, preds_arr),
        'macro_f1': f1_score(labels_arr, preds_arr, average='macro'),
        'binary_f1': f1_score(labels_arr, preds_arr, average='binary'),
        'auc_roc': roc_auc_score(labels_arr, probs_arr) if len(set(labels_arr)) > 1 else 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Davani et al., 2021 Baseline")
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--model_name", type=str,
                        default=os.path.join(BASE_DIR, "models", "deberta-v3-base"))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_lp", type=float, default=1.0,
                        help="Logit Pairing 权重")
    parser.add_argument("--cf_method", type=str, default="swap",
                        choices=["swap", "llm"],
                        help="反事实生成方法")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"Davani_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

    # 加载数据 (parquet)
    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))

    # 加载反事实数据
    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path = os.path.join(args.data_dir, f"{args.dataset}_train_{cf_suffix}.parquet")
    cf_df = None
    if os.path.exists(cf_path):
        cf_df = pd.read_parquet(cf_path)
        print(f"[CF] 加载反事实: {len(cf_df)} 条 ({args.cf_method})")
    else:
        print(f"[CF] 未找到反事实文件: {cf_path}，退化为普通 CE 训练")

    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = CounterfactualPairDataset(train_df, cf_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=cf_collate_fn)
    val_loader = DataLoader(SimpleTextDataset(val_df, tokenizer, args.max_len),
                            batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(SimpleTextDataset(test_df, tokenizer, args.max_len),
                             batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    label_counts = train_df['binary_label'].value_counts().sort_index()
    n_samples = len(train_df)
    class_weight = torch.tensor(
        [n_samples / (2 * label_counts[c]) for c in range(2)], dtype=torch.float32
    ).to(device)
    args._criterion_ce = nn.CrossEntropyLoss(weight=class_weight)

    model = DebertaV3Davani(args.model_name, num_classes=2).to(device)
    print(f"[Model] Davani LogitPairing | lambda_lp={args.lambda_lp} cf={args.cf_method}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    early_stopping = EarlyStopping(patience=args.patience)

    best_f1 = 0
    save_path = get_model_path(f"{exp_name}.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Train: loss={train_metrics['loss']:.4f} CE={train_metrics['ce']:.4f} LP={train_metrics['lp']:.4f}")
        print(f"  Val:   F1={val_metrics['macro_f1']:.4f} AUC={val_metrics['auc_roc']:.4f}")

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] Best F1={best_f1:.4f}")

        if early_stopping(-val_metrics['macro_f1']):
            print(f">>> [Early Stop] Best F1={best_f1:.4f}")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: F1={test_metrics['macro_f1']:.4f} AUC={test_metrics['auc_roc']:.4f}")

    results = {
        'experiment': exp_name, 'args': vars(args),
        'best_val_f1': best_f1, 'test_metrics': test_metrics,
    }
    results_path = get_log_path(f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"结果保存至: {results_path}")


if __name__ == "__main__":
    main()

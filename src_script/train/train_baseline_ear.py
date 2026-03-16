"""
=============================================================================
Baseline: EAR (Entropy-based Attention Regularization)
Paper: Kennedy et al., ACL Findings 2022
核心思想: 对 attention weights 施加熵正则化，迫使模型不过度关注身份词
L_total = L_CE + λ_ear * L_EAR
L_EAR = -H(attention) = Σ a_i * log(a_i)  (负熵，最小化使注意力更均匀)
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


class DebertaV3EAR(nn.Module):
    """DeBERTa-V3 + Entropy-based Attention Regularization"""

    def __init__(self, model_path, num_classes=2):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_attentions=False):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_hidden))
        result = {"logits": logits}
        if return_attentions and outputs.attentions is not None:
            result["attentions"] = outputs.attentions  # tuple of (B, H, L, L)
        return result


def attention_entropy_loss(attentions, attention_mask):
    """计算注意力熵正则化损失 (负熵 → 最小化使注意力更均匀)
    对最后一层的所有 head 计算平均熵"""
    last_attn = attentions[-1]  # (B, H, L, L)
    # mask padding positions
    mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, L)
    attn_masked = last_attn * mask
    # renormalize
    attn_masked = attn_masked / (attn_masked.sum(dim=-1, keepdim=True) + 1e-9)
    # entropy: -Σ p*log(p)
    entropy = -(attn_masked * (attn_masked + 1e-9).log()).sum(dim=-1)  # (B, H, L)
    # average over valid positions
    valid_mask = attention_mask.unsqueeze(1).float()  # (B, 1, L)
    entropy = (entropy * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-9)  # (B, H)
    mean_entropy = entropy.mean()
    # 我们要最大化熵 → 最小化负熵
    return -mean_entropy


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
    total_loss, total_ce, total_ear, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask, return_attentions=True)
            l_ce = args._criterion_ce(out['logits'], labels)
            l_ear = attention_entropy_loss(out['attentions'], mask)
            loss = l_ce + args.lambda_ear * l_ear

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
        total_ear += l_ear.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", ce=f"{total_ce/n:.4f}", ear=f"{total_ear/n:.4f}")

    return {'loss': total_loss/n, 'ce': total_ce/n, 'ear': total_ear/n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval]"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out = model(ids, mask, return_attentions=False)
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
    parser = argparse.ArgumentParser(description="EAR Baseline Training")
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
    parser.add_argument("--lambda_ear", type=float, default=0.1,
                        help="EAR 正则化权重")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"EAR_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

    # Data
    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(SimpleTextDataset(train_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(SimpleTextDataset(val_df, tokenizer, args.max_len),
                            batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(SimpleTextDataset(test_df, tokenizer, args.max_len),
                             batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    # Class weight
    label_counts = train_df['binary_label'].value_counts().sort_index()
    n_samples = len(train_df)
    class_weight = torch.tensor(
        [n_samples / (2 * label_counts[c]) for c in range(2)], dtype=torch.float32
    ).to(device)
    args._criterion_ce = nn.CrossEntropyLoss(weight=class_weight)

    # Model
    model = DebertaV3EAR(args.model_name, num_classes=2).to(device)
    print(f"[Model] EAR | lambda_ear={args.lambda_ear}")

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
        print(f"  Train: loss={train_metrics['loss']:.4f} CE={train_metrics['ce']:.4f} EAR={train_metrics['ear']:.4f}")
        print(f"  Val:   F1={val_metrics['macro_f1']:.4f} AUC={val_metrics['auc_roc']:.4f}")

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] Best F1={best_f1:.4f}")

        if early_stopping(-val_metrics['macro_f1']):
            print(f">>> [Early Stop] Best F1={best_f1:.4f}")
            break

    # Test
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
    print(f"\n>>> 下一步: python src_script/eval/eval_causal_fairness.py --checkpoint {save_path} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

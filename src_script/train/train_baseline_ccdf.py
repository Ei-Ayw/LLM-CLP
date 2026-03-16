"""
=============================================================================
Baseline: CCDF (Causal Counterfactual Debiasing Framework, arXiv 2024)
核心思想: 因果推断去偏 — 通过 TDE (Total Direct Effect) 去除身份词的虚假因果效应
1. 训练一个 bias-only 模型 (只看身份词特征)
2. 主模型预测时减去 bias-only 模型的贡献 (TDE)
L_total = L_CE + λ_kl * KL(P_main || P_debiased)
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

IDENTITY_TOKENS = {
    'black', 'white', 'asian', 'hispanic', 'african', 'european',
    'muslim', 'christian', 'jewish', 'islam', 'islamic', 'mosque', 'church',
    'quran', 'bible', 'hijab',
    'women', 'men', 'woman', 'man', 'she', 'he', 'her', 'his',
    'gay', 'lesbian', 'homosexual', 'lgbtq', 'queer', 'trans',
    'disabled', 'immigrant', 'refugee',
    'arab', 'chinese', 'indian', 'mexican', 'hindu', 'buddhist',
}


class BiasOnlyModel(nn.Module):
    """Bias-only 模型: 只基于身份词特征做预测 (捕获虚假相关)"""
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, identity_mask):
        """只对身份词 token 做平均池化"""
        embeds = self.embedding(input_ids)  # (B, L, D)
        mask = identity_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # (B, D)
        return self.classifier(pooled)


class DebertaV3CCDF(nn.Module):
    """DeBERTa-V3 主模型"""
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


def build_identity_mask(input_ids, tokenizer):
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for i in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        for j, token in enumerate(tokens):
            clean = token.replace('▁', '').replace('Ġ', '').lower()
            if clean in IDENTITY_TOKENS:
                mask[i, j] = 1.0
    return mask


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


def train_bias_model(bias_model, loader, device, tokenizer, epochs=5, lr=1e-3):
    """第一阶段: 训练 bias-only 模型"""
    print("\n[Phase 1] Training bias-only model...")
    optimizer = AdamW(bias_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    bias_model.train()

    for epoch in range(epochs):
        total_loss, n = 0, 0
        for batch in tqdm(loader, desc=f"[Bias Epoch {epoch+1}]"):
            ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            id_mask = build_identity_mask(ids, tokenizer).to(device)

            logits = bias_model(ids, id_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  Bias Epoch {epoch+1}: loss={total_loss/n:.4f}")

    bias_model.eval()
    print("[Phase 1] Bias model trained.")


def train_one_epoch(model, bias_model, loader, optimizer, scheduler, scaler, device, args, tokenizer):
    model.train()
    bias_model.eval()
    total_loss, total_ce, total_kl, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            l_ce = args._criterion_ce(out['logits'], labels)

            # TDE: 主模型 logits - bias 模型 logits
            l_kl = torch.tensor(0.0, device=device)
            if args.lambda_kl > 0:
                with torch.no_grad():
                    id_mask = build_identity_mask(ids, tokenizer).to(device)
                    bias_logits = bias_model(ids, id_mask)

                # debiased logits = main - bias
                debiased_logits = out['logits'] - args.tde_alpha * bias_logits
                # KL divergence: 让主模型的输出接近 debiased 输出
                p_main = F.log_softmax(out['logits'], dim=-1)
                p_debiased = F.softmax(debiased_logits.detach(), dim=-1)
                l_kl = F.kl_div(p_main, p_debiased, reduction='batchmean')

            loss = l_ce + args.lambda_kl * l_kl

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
        total_kl += l_kl.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", ce=f"{total_ce/n:.4f}", kl=f"{total_kl/n:.4f}")

    return {'loss': total_loss/n, 'ce': total_ce/n, 'kl': total_kl/n}


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
    parser = argparse.ArgumentParser(description="CCDF Baseline Training")
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
    parser.add_argument("--lambda_kl", type=float, default=1.0,
                        help="KL 散度损失权重")
    parser.add_argument("--tde_alpha", type=float, default=0.5,
                        help="TDE 减去 bias 的系数")
    parser.add_argument("--bias_epochs", type=int, default=5,
                        help="Bias-only 模型训练轮数")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"CCDF_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

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

    label_counts = train_df['binary_label'].value_counts().sort_index()
    n_samples = len(train_df)
    class_weight = torch.tensor(
        [n_samples / (2 * label_counts[c]) for c in range(2)], dtype=torch.float32
    ).to(device)
    args._criterion_ce = nn.CrossEntropyLoss(weight=class_weight)

    # Phase 1: Train bias-only model
    vocab_size = tokenizer.vocab_size
    bias_model = BiasOnlyModel(vocab_size, embed_dim=128, num_classes=2).to(device)
    train_bias_model(bias_model, train_loader, device, tokenizer, epochs=args.bias_epochs)

    # Phase 2: Train main model with TDE debiasing
    model = DebertaV3CCDF(args.model_name, num_classes=2).to(device)
    print(f"\n[Phase 2] Main model training | lambda_kl={args.lambda_kl} tde_alpha={args.tde_alpha}")

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
        train_metrics = train_one_epoch(model, bias_model, train_loader, optimizer, scheduler, scaler, device, args, tokenizer)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Train: loss={train_metrics['loss']:.4f} CE={train_metrics['ce']:.4f} KL={train_metrics['kl']:.4f}")
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
    print(f"\n>>> 下一步: python src_script/eval/eval_causal_fairness.py --checkpoint {save_path} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

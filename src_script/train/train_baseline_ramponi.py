"""
=============================================================================
Baseline: Ramponi & Tonelli, 2022 - Adversarial Debiasing
论文: "Features or Spurious Artifacts? Data-centric Baselines for Fair
       and Robust Hate Speech Detection"
链接: https://aclanthology.org/2022.naacl-main.221.pdf
代码: https://github.com/dhfbk/hate-speech-artifacts

核心思想:
1. 使用对抗训练移除虚假相关特征（如身份词）
2. 主分类器学习预测毒性
3. 对抗分类器尝试从主分类器的表示中预测"是否包含身份词"
4. 通过 Gradient Reversal Layer，主分类器学习让表示对身份不变
5. L_total = L_CE + λ_adv * L_adversarial (GRL 自动处理符号)
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

IDENTITY_KEYWORDS = {
    'black', 'white', 'asian', 'hispanic', 'latino', 'african', 'european',
    'muslim', 'christian', 'jewish', 'islam', 'islamic', 'mosque', 'church',
    'quran', 'bible', 'hijab',
    'women', 'men', 'woman', 'man', 'female', 'male', 'she', 'he', 'her', 'his',
    'gay', 'lesbian', 'homosexual', 'lgbtq', 'queer', 'transgender', 'trans',
    'disabled', 'immigrant', 'refugee',
    'arab', 'chinese', 'indian', 'mexican', 'hindu', 'buddhist',
}


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer: forward 不变，backward 反转梯度"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DebertaV3Ramponi(nn.Module):
    """DeBERTa-V3 with Adversarial Debiasing (GRL)"""
    def __init__(self, model_path, num_classes=2, num_identity_classes=2):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path)
        hidden_size = self.config.hidden_size

        # 主分类器 (毒性)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 对抗分类器 (身份属性)
        self.adv_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_identity_classes),
        )

    def forward(self, input_ids, attention_mask, alpha=1.0):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        # 主任务
        main_logits = self.classifier(self.dropout(cls_hidden))

        # 对抗任务 (通过 GRL)
        reversed_hidden = GradientReversalFunction.apply(cls_hidden, alpha)
        adv_logits = self.adv_classifier(reversed_hidden)

        return {"logits": main_logits, "adv_logits": adv_logits}


class IdentityTextDataset(Dataset):
    """数据集: 自动从文本中提取身份标签"""
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text'].values
        self.labels = df['binary_label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 提取身份标签: 文本中是否包含身份词
        if 'has_identity' in df.columns:
            self.identity_labels = df['has_identity'].astype(int).values
        else:
            self.identity_labels = np.array([
                int(any(kw in str(t).lower() for kw in IDENTITY_KEYWORDS))
                for t in self.texts
            ])

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
            'identity_label': torch.tensor(self.identity_labels[idx], dtype=torch.long),
        }


class SimpleTextDataset(Dataset):
    """评估用简单数据集"""
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


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args, epoch, total_epochs):
    model.train()
    total_loss, total_ce, total_adv, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    # 渐进式增加对抗强度 (DANN schedule)
    p = float(epoch) / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        id_labels = batch['identity_label'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask, alpha=alpha)
            l_ce = args._criterion_ce(out['logits'], labels)
            l_adv = F.cross_entropy(out['adv_logits'], id_labels)
            # GRL 已经处理了符号，所以这里直接加
            loss = l_ce + args.lambda_adv * l_adv

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
        total_adv += l_adv.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", ce=f"{total_ce/n:.4f}",
                         adv=f"{total_adv/n:.4f}", alpha=f"{alpha:.3f}")

    return {'loss': total_loss/n, 'ce': total_ce/n, 'adv': total_adv/n, 'alpha': alpha}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval]"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out = model(ids, mask, alpha=0.0)  # 评估时不需要对抗
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
    parser = argparse.ArgumentParser(description="Ramponi & Tonelli, 2022 Baseline")
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
    parser.add_argument("--lambda_adv", type=float, default=0.1,
                        help="对抗损失权重")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"Ramponi_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

    # 加载数据 (parquet)
    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(IdentityTextDataset(train_df, tokenizer, args.max_len),
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

    model = DebertaV3Ramponi(args.model_name, num_classes=2, num_identity_classes=2).to(device)
    print(f"[Model] Ramponi Adversarial | lambda_adv={args.lambda_adv}")

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
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                         device, args, epoch, args.epochs)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Train: loss={train_metrics['loss']:.4f} CE={train_metrics['ce']:.4f} "
              f"ADV={train_metrics['adv']:.4f} α={train_metrics['alpha']:.3f}")
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

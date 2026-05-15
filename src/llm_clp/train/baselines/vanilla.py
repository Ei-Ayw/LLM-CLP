"""
=============================================================================
Baseline: Vanilla DeBERTa-v3-base (无任何去偏方法)
标准微调，作为所有去偏方法的对照基线
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
from src.llm_clp.common.training import EarlyStopping
from src.llm_clp.common.paths import model_path, log_path


class DebertaV3Vanilla(nn.Module):
    """Vanilla DeBERTa-v3 基线模型（无任何去偏方法）"""

    def __init__(self, model_path, num_classes=2):
        """
        初始化 Vanilla 模型
        Args:
            model_path: 预训练模型路径
            num_classes: 分类类别数（默认为2：有毒/无毒）
        """
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        前向传播
        Args:
            input_ids: 输入的 token IDs
            attention_mask: 注意力掩码
        Returns:
            包含 logits 的字典
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 位置的隐藏状态
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_hidden))
        return {"logits": logits}


class SimpleTextDataset(Dataset):
    """简单文本数据集类，用于加载毒性分类数据"""

    def __init__(self, df, tokenizer, max_len=128):
        """
        初始化数据集
        Args:
            df: 包含 'text' 和 'binary_label' 列的 DataFrame
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        self.texts = df['text'].values
        self.labels = df['binary_label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """获取单条数据样本"""
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


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, criterion):
    """
    训练一个 epoch
    Args:
        model: 模型
        loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: 梯度缩放器（用于混合精度训练）
        device: 设备（cuda/cpu）
        criterion: 损失函数
    Returns:
        平均训练损失
    """
    model.train()
    total_loss, n = 0, 0
    pbar = tqdm(loader, desc="[训练]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 混合精度训练
        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            loss = criterion(out['logits'], labels)

        # 梯度累积（每2步更新一次）
        loss_scaled = loss / 2  # grad_accum=2
        scaler.scale(loss_scaled).backward()

        # 每2步执行一次优化器更新
        if (i + 1) % 2 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    """
    评估模型性能
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备（cuda/cpu）
    Returns:
        包含 accuracy, macro_f1, binary_f1, auc_roc 的字典
    """
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[评估]"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        # 混合精度推理
        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            # 获取正类（有毒）的概率
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
    """主函数：解析参数、加载数据、训练并评估模型"""
    parser = argparse.ArgumentParser(description="Vanilla DeBERTa-v3 基线模型（无去偏方法）")
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"],
                        help="数据集名称")
    parser.add_argument("--model_name", type=str,
                        default=os.path.join(BASE_DIR, "models", "deberta-v3-base"),
                        help="预训练模型路径")
    parser.add_argument("--max_len", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=6,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热步数比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减系数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"),
                        help="数据目录")
    parser.add_argument("--patience", type=int, default=3,
                        help="早停耐心值（多少轮验证集性能无提升后停止）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = f"Vanilla_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[实验] {exp_name}")

    # 加载训练、验证、测试数据集
    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[数据] 训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

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
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    model = DebertaV3Vanilla(args.model_name, num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = len(train_loader) // 2 * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    early_stopping = EarlyStopping(patience=args.patience)

    best_f1 = 0
    save_path = model_path(f"{exp_name}.pth")

    for epoch in range(args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, criterion)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  训练: 损失={train_loss:.4f}")
        print(f"  验证: F1={val_metrics['macro_f1']:.4f} AUC={val_metrics['auc_roc']:.4f}")

        # 保存最佳模型
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), save_path)
            print(f"  [保存] 最佳 F1={best_f1:.4f}")

        # 检查早停
        if early_stopping(-val_metrics['macro_f1']):
            print(f">>> [早停] 最佳 F1={best_f1:.4f}")
            break

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n测试: F1={test_metrics['macro_f1']:.4f} AUC={test_metrics['auc_roc']:.4f}")

    # 保存实验结果
    results = {
        'experiment': exp_name, 'args': vars(args),
        'best_val_f1': best_f1, 'test_metrics': test_metrics,
    }
    results_path = log_path(f"{exp_name}_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"结果保存至: {results_path}")


if __name__ == "__main__":
    main()

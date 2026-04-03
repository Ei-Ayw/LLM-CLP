"""
=============================================================================
Backbone Davani — 训练 Davani 基线 (BERT / RoBERTa / DeBERTa-v3)
Davani: Counterfactual Logit Pairing
L_total = L_CE(orig) + L_CE(cf) + λ_lp * MSE(logits_orig, logits_cf)
=============================================================================
"""
import os, sys, argparse, json, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_backbone_baselines import BackboneDavani
from train_utils import EarlyStopping
from path_config import get_model_path, get_log_path

BACKBONE_PATHS = {
    'bert':    os.path.join(BASE_DIR, "models", "bert-base-uncased"),
    'roberta': os.path.join(BASE_DIR, "models", "roberta-base"),
    'deberta': os.path.join(BASE_DIR, "models", "deberta-v3-base"),
}


class CounterfactualPairDataset(Dataset):
    def __init__(self, train_df, cf_df, tokenizer, max_len=128):
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.orig_texts  = train_df['text'].values
        self.orig_labels = train_df['binary_label'].values
        self.cf_map = {}
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                orig = row['original_text']
                if orig not in self.cf_map:
                    self.cf_map[orig] = []
                self.cf_map[orig].append(row['cf_text'])

    def __len__(self): return len(self.orig_texts)

    def __getitem__(self, idx):
        text  = str(self.orig_texts[idx])
        label = self.orig_labels[idx]
        enc   = self.tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_attention_mask=True,
        )
        item = {
            'input_ids':      torch.tensor(enc['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'],  dtype=torch.long),
            'label':          torch.tensor(label,                  dtype=torch.long),
            'has_cf':         torch.tensor(0,                      dtype=torch.long),
        }
        if text in self.cf_map and self.cf_map[text]:
            cf_text = np.random.choice(self.cf_map[text])
            cf_enc  = self.tokenizer.encode_plus(
                str(cf_text), add_special_tokens=True,
                max_length=self.max_len, padding='max_length',
                truncation=True, return_attention_mask=True,
            )
            item['cf_input_ids']      = torch.tensor(cf_enc['input_ids'],      dtype=torch.long)
            item['cf_attention_mask'] = torch.tensor(cf_enc['attention_mask'],  dtype=torch.long)
            item['has_cf']            = torch.tensor(1,                          dtype=torch.long)
        return item


def cf_collate_fn(batch):
    keys   = ['input_ids', 'attention_mask', 'label', 'has_cf']
    result = {k: torch.stack([b[k] for b in batch]) for k in keys}
    cf_items = [b for b in batch if b['has_cf'].item() == 1]
    if cf_items:
        result['cf_input_ids']      = torch.stack([b['cf_input_ids']      for b in cf_items])
        result['cf_attention_mask'] = torch.stack([b['cf_attention_mask'] for b in cf_items])
        result['cf_indices']        = torch.tensor(
            [i for i, b in enumerate(batch) if b['has_cf'].item() == 1], dtype=torch.long)
    return result


class SimpleTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts  = df['text'].values
        self.labels = df['binary_label'].values
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.texts[idx]), add_special_tokens=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_attention_mask=True,
        )
        return {
            'input_ids':      torch.tensor(enc['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'],  dtype=torch.long),
            'label':          torch.tensor(self.labels[idx],       dtype=torch.long),
        }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    criterion, lambda_lp, grad_accum=2):
    model.train()
    total_loss, total_ce, total_lp, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl  = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            out  = model(ids, mask)
            l_ce = criterion(out['logits'], lbl)

            l_lp = torch.tensor(0.0, device=device)
            if 'cf_input_ids' in batch and lambda_lp > 0:
                cf_ids    = batch['cf_input_ids'].to(device)
                cf_mask   = batch['cf_attention_mask'].to(device)
                cf_indices = batch['cf_indices'].to(device)
                cf_out    = model(cf_ids, cf_mask)
                orig_logits_matched = out['logits'][cf_indices]
                l_lp = F.mse_loss(orig_logits_matched, cf_out['logits'])
                cf_labels = lbl[cf_indices]
                l_ce_cf   = criterion(cf_out['logits'], cf_labels)
                l_ce      = (l_ce + l_ce_cf) / 2

            loss = l_ce + lambda_lp * l_lp

        scaler.scale(loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            if scheduler: scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item(); total_ce += l_ce.item()
        total_lp   += l_lp.item() if isinstance(l_lp, torch.Tensor) else l_lp
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", lp=f"{total_lp/n:.4f}")
    return {'loss': total_loss/n, 'ce': total_ce/n, 'lp': total_lp/n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval]"):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out   = model(ids, mask)
            probs = F.softmax(out['logits'], dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch['label'].numpy())
    pa, la = np.array(all_probs), np.array(all_labels)
    pd_ = (pa >= 0.5).astype(int)
    return {
        'accuracy':  float(accuracy_score(la, pd_)),
        'macro_f1':  float(f1_score(la, pd_, average='macro')),
        'binary_f1': float(f1_score(la, pd_, average='binary')),
        'auc_roc':   float(roc_auc_score(la, pa)) if len(set(la)) > 1 else 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    type=str, default="bert",
                        choices=["bert", "roberta", "deberta"])
    parser.add_argument("--dataset",     type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--model_name",  type=str, default=None)
    parser.add_argument("--max_len",     type=int,   default=128)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--grad_accum",  type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=6)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--lambda_lp",   type=float, default=1.0)
    parser.add_argument("--cf_method",   type=str,   default="swap",
                        choices=["swap", "llm"])
    parser.add_argument("--data_dir",    type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    args = parser.parse_args()

    model_path = args.model_name or BACKBONE_PATHS[args.backbone]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if n_gpu > 1:
        print(f"[GPU] Using {n_gpu} GPUs (DataParallel)")

    exp_name = f"Davani_{args.backbone}_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))

    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path   = os.path.join(args.data_dir, f"{args.dataset}_train_{cf_suffix}.parquet")
    cf_df     = None
    if os.path.exists(cf_path):
        cf_df = pd.read_parquet(cf_path)
        print(f"[CF] {len(cf_df)} pairs ({args.cf_method})")
    else:
        print(f"[CF] Not found: {cf_path}, fallback to CE-only")
    print(f"[Data] Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

    tokenizer    = AutoTokenizer.from_pretrained(model_path)
    train_loader = DataLoader(CounterfactualPairDataset(train_df, cf_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=cf_collate_fn)
    val_loader   = DataLoader(SimpleTextDataset(val_df,  tokenizer, args.max_len),
                              batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(SimpleTextDataset(test_df, tokenizer, args.max_len),
                              batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    lc = train_df['binary_label'].value_counts().sort_index()
    cw = torch.tensor([len(train_df)/(2*lc[c]) for c in range(2)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    model     = BackboneDavani(model_path, backbone_type=args.backbone, num_classes=2).to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.cuda.amp.GradScaler()
    total_steps  = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    early_stopping = EarlyStopping(patience=args.patience)

    best_f1   = 0
    save_path = get_model_path(f"{exp_name}.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tm = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             device, criterion, args.lambda_lp, args.grad_accum)
        vm = evaluate(model, val_loader, device)
        print(f"  Train loss={tm['loss']:.4f} CE={tm['ce']:.4f} LP={tm['lp']:.4f}")
        print(f"  Val   F1={vm['macro_f1']:.4f} AUC={vm['auc_roc']:.4f}")
        if vm['macro_f1'] > best_f1:
            best_f1 = vm['macro_f1']
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)
            print(f"  [Save] Best F1={best_f1:.4f}")
        if early_stopping(-vm['macro_f1']):
            print(f">>> [Early Stop] Best F1={best_f1:.4f}"); break

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.load_state_dict(torch.load(save_path, map_location=device))
    test_met = evaluate(model, test_loader, device)
    print(f"\nTest F1={test_met['macro_f1']:.4f} AUC={test_met['auc_roc']:.4f}")

    results = {'experiment': exp_name, 'args': vars(args),
               'model_path': model_path, 'best_val_f1': best_f1, 'test_metrics': test_met}
    with open(get_log_path(f"{exp_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"结果保存至: {get_log_path(exp_name+'_results.json')}")


if __name__ == "__main__":
    main()

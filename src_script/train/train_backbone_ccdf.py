"""
=============================================================================
Backbone CCDF — 训练 CCDF 基线 (BERT / RoBERTa / DeBERTa-v3)
CCDF: Causal Counterfactual Debiasing Framework (TDE)
L_total = L_CE + λ_kl * KL(P_main || P_debiased)
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

from model_backbone_baselines import BackboneCCDF, BiasOnlyModel, build_identity_mask
from train_utils import EarlyStopping
from path_config import get_model_path, get_log_path

BACKBONE_PATHS = {
    'bert':    os.path.join(BASE_DIR, "models", "bert-base-uncased"),
    'roberta': os.path.join(BASE_DIR, "models", "roberta-base"),
    'deberta': os.path.join(BASE_DIR, "models", "deberta-v3-base"),
}


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


def train_bias_model(bias_model, loader, device, tokenizer, epochs=5, lr=1e-3):
    """Phase 1: 训练 bias-only 模型"""
    print("\n[Phase 1] Training bias-only model...")
    optimizer = AdamW(bias_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    bias_model.train()
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for batch in tqdm(loader, desc=f"[Bias Epoch {epoch+1}]"):
            ids    = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            id_mask = build_identity_mask(ids, tokenizer).to(device)
            logits = bias_model(ids, id_mask)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1
        print(f"  Bias Epoch {epoch+1}: loss={total_loss/n:.4f}")
    bias_model.eval()
    print("[Phase 1] Bias model trained.")


def train_one_epoch(model, bias_model, loader, optimizer, scheduler, scaler,
                    device, criterion, tokenizer, lambda_kl, tde_alpha, grad_accum=2):
    model.train()
    bias_model.eval()
    total_loss, total_ce, total_kl, n = 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl  = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            out   = model(ids, mask)
            l_ce  = criterion(out['logits'], lbl)

            l_kl = torch.tensor(0.0, device=device)
            if lambda_kl > 0:
                with torch.no_grad():
                    id_mask    = build_identity_mask(ids, tokenizer).to(device)
                    bias_logits = bias_model(ids, id_mask)
                debiased_logits = out['logits'] - tde_alpha * bias_logits
                p_main    = F.log_softmax(out['logits'], dim=-1)
                p_debias  = F.softmax(debiased_logits.detach(), dim=-1)
                l_kl = F.kl_div(p_main, p_debias, reduction='batchmean')

            loss = l_ce + lambda_kl * l_kl

        scaler.scale(loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            if scheduler: scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item(); total_ce += l_ce.item()
        total_kl   += l_kl.item() if isinstance(l_kl, torch.Tensor) else l_kl
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", kl=f"{total_kl/n:.4f}")
    return {'loss': total_loss/n, 'ce': total_ce/n, 'kl': total_kl/n}


@torch.no_grad()
def evaluate_tde(model, bias_model, loader, device, tokenizer, tde_alpha):
    model.eval(); bias_model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval-TDE]"):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            id_mask    = build_identity_mask(ids, tokenizer).to(device)
            bias_logits = bias_model(ids, id_mask)
            debiased    = out['logits'] - tde_alpha * bias_logits
            probs = F.softmax(debiased, dim=-1)[:, 1]
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
    parser.add_argument("--patience",    type=int,   default=2)
    parser.add_argument("--lambda_kl",   type=float, default=1.0)
    parser.add_argument("--tde_alpha",   type=float, default=0.5)
    parser.add_argument("--bias_epochs", type=int,   default=5)
    parser.add_argument("--data_dir",    type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    args = parser.parse_args()

    model_path = args.model_name or BACKBONE_PATHS[args.backbone]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if n_gpu > 1:
        print(f"[GPU] Using {n_gpu} GPUs (DataParallel)")

    exp_name = f"CCDF_{args.backbone}_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"[Experiment] {exp_name}")

    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[Data] Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

    tokenizer    = AutoTokenizer.from_pretrained(model_path)
    train_loader = DataLoader(SimpleTextDataset(train_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(SimpleTextDataset(val_df,   tokenizer, args.max_len),
                              batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(SimpleTextDataset(test_df,  tokenizer, args.max_len),
                              batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    lc = train_df['binary_label'].value_counts().sort_index()
    cw = torch.tensor([len(train_df)/(2*lc[c]) for c in range(2)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Phase 1: bias-only model
    vocab_size = tokenizer.vocab_size
    bias_model = BiasOnlyModel(vocab_size, embed_dim=128, num_classes=2).to(device)
    train_bias_model(bias_model, train_loader, device, tokenizer, epochs=args.bias_epochs)

    # Phase 2: main model
    model     = BackboneCCDF(model_path, backbone_type=args.backbone, num_classes=2).to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    print(f"\n[Phase 2] Main model | lambda_kl={args.lambda_kl} tde_alpha={args.tde_alpha}")
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
        tm = train_one_epoch(model, bias_model, train_loader, optimizer, scheduler, scaler,
                             device, criterion, tokenizer, args.lambda_kl, args.tde_alpha,
                             args.grad_accum)
        vm = evaluate_tde(model, bias_model, val_loader, device, tokenizer, args.tde_alpha)
        print(f"  Train loss={tm['loss']:.4f} CE={tm['ce']:.4f} KL={tm['kl']:.4f}")
        print(f"  Val   F1={vm['macro_f1']:.4f} AUC={vm['auc_roc']:.4f}")
        if vm['macro_f1'] > best_f1:
            best_f1 = vm['macro_f1']
            torch.save({
                'main_model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'bias_model': bias_model.state_dict(),
                'tde_alpha':  args.tde_alpha,
                'vocab_size': vocab_size,
            }, save_path)
            print(f"  [Save] Best F1={best_f1:.4f}")
        if early_stopping(-vm['macro_f1']):
            print(f">>> [Early Stop] Best F1={best_f1:.4f}"); break

    ckpt = torch.load(save_path, map_location=device)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.load_state_dict(ckpt['main_model'])
    bias_model.load_state_dict(ckpt['bias_model'])

    test_met = evaluate_tde(model, bias_model, test_loader, device, tokenizer, args.tde_alpha)
    print(f"\nTest (TDE) F1={test_met['macro_f1']:.4f} AUC={test_met['auc_roc']:.4f}")

    results = {'experiment': exp_name, 'args': vars(args),
               'model_path': model_path, 'best_val_f1': best_f1, 'test_metrics': test_met}
    with open(get_log_path(f"{exp_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"结果保存至: {get_log_path(exp_name+'_results.json')}")


if __name__ == "__main__":
    main()

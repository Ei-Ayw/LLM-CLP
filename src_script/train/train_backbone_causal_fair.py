"""
=============================================================================
Backbone CausalFair — 训练 CausalFair 方法 (BERT / RoBERTa / DeBERTa-v3)
L_total = L_CE + λ_clp × L_CLP + λ_con × L_SupCon
=============================================================================
"""
import os, sys, argparse, json, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_backbone_cf import BackboneCausalFair
from data_loader_cf import get_causal_fair_loader
from loss_contrastive import CounterfactualSupConLoss, CounterfactualLogitPairing
from train_utils import EarlyStopping
from path_config import get_model_path, get_log_path

BACKBONE_PATHS = {
    'bert':    os.path.join(BASE_DIR, "models", "bert-base-uncased"),
    'roberta': os.path.join(BASE_DIR, "models", "roberta-base"),
    'deberta': os.path.join(BASE_DIR, "models", "deberta-v3-base"),
}


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    criterion_ce  = nn.CrossEntropyLoss()
    criterion_clp = CounterfactualLogitPairing()
    criterion_con = CounterfactualSupConLoss(temperature=args.temperature)

    total_loss, total_ce, total_clp, total_con, n = 0, 0, 0, 0, 0
    pbar = tqdm(loader, desc="[Train]")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        orig_ids  = batch['orig_input_ids'].to(device)
        orig_mask = batch['orig_attention_mask'].to(device)
        cf_ids    = batch['cf_input_ids'].to(device)
        cf_mask   = batch['cf_attention_mask'].to(device)
        labels    = batch['label'].to(device)
        has_cf    = batch['has_cf'].to(device)

        with torch.cuda.amp.autocast():
            out_orig = model(orig_ids, orig_mask, return_features=True)
            out_cf   = model(cf_ids,   cf_mask,   return_features=True)

            l_ce = criterion_ce(out_orig['logits'], labels)

            l_clp = torch.tensor(0.0, device=device)
            if has_cf.sum() > 0:
                cf_bool = has_cf.bool()
                l_clp = criterion_clp(
                    out_orig['logits'][cf_bool],
                    out_cf['logits'][cf_bool]
                )

            l_con = torch.tensor(0.0, device=device)
            if has_cf.sum() > 1:
                cf_bool = has_cf.bool()
                l_con = criterion_con(
                    out_orig['features'][cf_bool],
                    out_cf['features'][cf_bool],
                    labels[cf_bool]
                )

            loss = l_ce + args.lambda_clp * l_clp + args.lambda_con * l_con

        scaler.scale(loss / args.grad_accum).backward()
        if (i + 1) % args.grad_accum == 0:
            scaler.step(optimizer); scaler.update()
            if scheduler: scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item(); total_ce  += l_ce.item()
        total_clp  += l_clp.item(); total_con += l_con.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", ce=f"{total_ce/n:.4f}",
                         clp=f"{total_clp/n:.4f}", con=f"{total_con/n:.4f}")

    return {'loss': total_loss/n, 'ce': total_ce/n,
            'clp': total_clp/n,  'con': total_con/n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[Eval]"):
        orig_ids  = batch['orig_input_ids'].to(device)
        orig_mask = batch['orig_attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            out   = model(orig_ids, orig_mask, return_features=False)
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
    parser.add_argument("--backbone",     type=str, default="bert",
                        choices=["bert", "roberta", "deberta"])
    parser.add_argument("--dataset",      type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--cf_method",    type=str, default="llm",
                        choices=["none", "swap", "llm"])
    parser.add_argument("--model_name",   type=str, default=None)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--grad_accum",   type=int,   default=2)
    parser.add_argument("--epochs",       type=int,   default=6)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--patience",     type=int,   default=3)
    parser.add_argument("--lambda_clp",   type=float, default=1.0)
    parser.add_argument("--lambda_con",   type=float, default=0.5)
    parser.add_argument("--temperature",  type=float, default=0.07)
    parser.add_argument("--data_dir",     type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    args = parser.parse_args()

    model_path = args.model_name or BACKBONE_PATHS[args.backbone]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if n_gpu > 1:
        print(f"[GPU] Using {n_gpu} GPUs (DataParallel)")

    exp_name = (f"Ours_{args.backbone}_{args.dataset}_{args.cf_method}_"
                f"clp{args.lambda_clp}_con{args.lambda_con}_"
                f"seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}")
    print(f"[Experiment] {exp_name}")

    train_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    print(f"[Data] Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

    cf_df = None
    if args.cf_method != "none":
        cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
        cf_path   = os.path.join(args.data_dir, f"{args.dataset}_train_{cf_suffix}.parquet")
        if os.path.exists(cf_path):
            cf_df = pd.read_parquet(cf_path)
            print(f"[CF] {len(cf_df)} pairs ({args.cf_method})")
        else:
            print(f"[CF] Not found: {cf_path}, fallback to CE-only")

    tokenizer    = AutoTokenizer.from_pretrained(model_path)
    train_loader = get_causal_fair_loader(
        train_df, cf_df, tokenizer,
        batch_size=args.batch_size, max_len=args.max_len, shuffle=True)
    val_loader   = get_causal_fair_loader(
        val_df, None, tokenizer,
        batch_size=args.batch_size*2, max_len=args.max_len, shuffle=False)
    test_loader  = get_causal_fair_loader(
        test_df, None, tokenizer,
        batch_size=args.batch_size*2, max_len=args.max_len, shuffle=False)

    model     = BackboneCausalFair(model_path, backbone_type=args.backbone, num_classes=2).to(device)
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters()):,}")
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
        tm = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args)
        vm = evaluate(model, val_loader, device)
        print(f"  Train loss={tm['loss']:.4f} CE={tm['ce']:.4f} CLP={tm['clp']:.4f} Con={tm['con']:.4f}")
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

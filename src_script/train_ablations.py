import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse

# Set project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# Set Hugging Face cache directory and mirror endpoint
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset

def train_fn(model, loader, optimizer, scheduler, device, ablation_type, accumulation_steps=2):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Ablation: {ablation_type}")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        
        outputs = model(input_ids, attention_mask)
        loss_tox = criterion(outputs['logits_tox'], y_tox)
        
        if ablation_type == "no_mtl":
            loss = loss_tox
        else:
            loss_sub = criterion(outputs['logits_sub'], batch['y_sub'].to(device))
            loss_id = criterion(outputs['logits_id'], batch['y_id'].to(device))
            loss = loss_tox + 0.5 * loss_sub + 0.2 * loss_id
            
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(loss=total_loss / (i + 1))
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True, choices=["no_pooling", "no_mtl", "no_reweight"])
    args = parser.parse_args()
    
    MODEL_PATH = "microsoft/deberta-v3-base"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.parquet")
    OUTPUT_DIR = os.path.join(BASE_DIR, "src_result")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    use_pooling = False if args.ablation == "no_pooling" else True
    model = DebertaToxicityMTL(MODEL_PATH, use_attention_pooling=use_pooling).to(device)
    
    # Using a medium-sized subset for ablations to save time
    train_df = pd.read_parquet(TRAIN_FILE).sample(min(300000, 1700000))
    train_ds = ToxicityDataset(train_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Run 2 epochs for ablation comparison
    for epoch in range(2):
        print(f"Ablation: {args.ablation}, Epoch {epoch+1}")
        train_fn(model, train_loader, optimizer, None, device, args.ablation)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"res_ablation_{args.ablation}.pth"))

if __name__ == "__main__":
    main()

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# Set project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# Set Hugging Face cache directory and mirror endpoint
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset
import os
import numpy as np

def weighted_tox_loss(logits, targets, has_id):
    """
    Identity-Aware Reweighting for Toxicity loss.
    targets: (B, 1), has_id: (B)
    """
    weights = torch.ones_like(targets)
    has_id = has_id.unsqueeze(-1)
    
    # Case: Non-toxic but contains identity (y=0, has_id=1) -> w=2.5
    weights[(targets < 0.5) & (has_id == 1)] = 2.5
    # Case: Toxic and contains identity (y=1, has_id=1) -> w=1.5
    weights[(targets >= 0.5) & (has_id == 1)] = 1.5
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, targets)
    return (loss * weights).mean()

def train_fn(model, loader, optimizer, scheduler, device, accumulation_steps=2):
    model.train()
    criterion_sub_id = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training (Stage 2)")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        has_id = batch['has_id'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        # Reweighted Loss
        loss_tox = weighted_tox_loss(outputs['logits_tox'], y_tox, has_id)
        loss_sub = criterion_sub_id(outputs['logits_sub'], y_sub)
        loss_id = criterion_sub_id(outputs['logits_id'], y_id)
        
        # MTL weighting: L = L_tox + 0.5*L_sub + 0.2*L_id
        loss = loss_tox + 0.5 * loss_sub + 0.2 * loss_id
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(loss=total_loss / (i + 1))
        
    return total_loss / len(loader)

def eval_fn(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    preds_tox = []
    targets_tox = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            
            outputs = model(input_ids, attention_mask)
            loss_tox = criterion(outputs['logits_tox'], y_tox)
            
            total_loss += loss_tox.item()
            preds_tox.extend(torch.sigmoid(outputs['logits_tox']).cpu().detach().numpy())
            targets_tox.extend(y_tox.cpu().detach().numpy())
            
    return total_loss / len(loader), np.array(preds_tox), np.array(targets_tox)

def main():
    # Config
    MODEL_PATH = "microsoft/deberta-v3-base"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_STAGE1 = os.path.join(BASE_DIR, "src_result", "res_stage1_best.pth")
    TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.parquet")
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    OUTPUT_DIR = os.path.join(BASE_DIR, "src_result")
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 2
    MAX_LEN = 256
    EPOCHS = 2
    LR = 1e-5 # Lower LR for Stage 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = DebertaToxicityMTL(MODEL_PATH).to(device)
    
    if os.path.exists(CHECKPOINT_STAGE1):
        print(f"Loading Stage 1 checkpoint from {CHECKPOINT_STAGE1}")
        model.load_state_dict(torch.load(CHECKPOINT_STAGE1, map_location=device))
    else:
        print("Warning: Stage 1 checkpoint not found. Starting from scratch.")
    
    print("Loading datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    val_df = pd.read_parquet(VAL_FILE)
    
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=MAX_LEN)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_train_steps = int(len(train_ds) / BATCH_SIZE / ACCUMULATION_STEPS * EPOCHS)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, ACCUMULATION_STEPS)
        val_loss, _, _ = eval_fn(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "res_stage2_final.pth"))
            print("Model saved.")

if __name__ == "__main__":
    main()

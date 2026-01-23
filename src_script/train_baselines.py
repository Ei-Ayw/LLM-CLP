import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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

from model_baselines import BertCNNBiLSTM
from data_loader import ToxicityDataset

def train_fn(model, loader, optimizer, scheduler, device, accumulation_steps=2):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Baseline Training")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs['logits_tox'], y_tox)
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(loss=total_loss / (i + 1))
    return total_loss / len(loader)

def main():
    MODEL_NAME = "bert-base-uncased" # Baseline usually uses base BERT
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.parquet")
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    OUTPUT_DIR = os.path.join(BASE_DIR, "src_result")
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = BertCNNBiLSTM(MODEL_NAME).to(device)
    
    train_df = pd.read_parquet(TRAIN_FILE).sample(200000) # Baseline training on subset for speed
    val_df = pd.read_parquet(VAL_FILE)
    
    train_ds = ToxicityDataset(train_df, tokenizer)
    val_ds = ToxicityDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_train_steps = int(len(train_ds) / BATCH_SIZE * EPOCHS)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"res_baseline_bert_cnn_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    main()

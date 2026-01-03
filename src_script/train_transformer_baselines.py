import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from data_loader import ToxicityDataset

def train_fn(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['y_tox'].to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = criterion(outputs.logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["bert-base-uncased", "roberta-base", "microsoft/deberta-base"])
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.parquet")
    OUTPUT_DIR = os.path.join(BASE_DIR, "src_result")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
    
    df = pd.read_parquet(TRAIN_FILE).sample(200000)
    dataset = ToxicityDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader)*2)
    
    for epoch in range(2):
        print(f"Model: {args.model_name}, Epoch {epoch+1}")
        train_fn(model, loader, optimizer, scheduler, device)
        save_path = os.path.join(OUTPUT_DIR, f"res_baseline_{args.model_name.replace('-','_')}.pth")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.texts = df['comment_text'].values
        self.y_tox = df['y_tox'].values
        
        self.subtype_cols = [
            'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
        ]
        self.identity_cols = [
            'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 
            'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
        ]
        
        self.y_sub = df[self.subtype_cols].values
        self.y_id = df[self.identity_cols].values
        self.has_id = df['has_identity'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'y_tox': torch.tensor(self.y_tox[idx], dtype=torch.float),
            'y_sub': torch.tensor(self.y_sub[idx], dtype=torch.float),
            'y_id': torch.tensor(self.y_id[idx], dtype=torch.float),
            'has_id': torch.tensor(self.has_id[idx], dtype=torch.long)
        }

def get_dataloader(file_path, tokenizer, batch_size=16, max_len=256, shuffle=True):
    df = pd.read_parquet(file_path)
    dataset = ToxicityDataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader

if __name__ == "__main__":
    # Test
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    # Just a small sample for testing
    dummy_df = pd.DataFrame({
        'comment_text': ["Hello world", "Bad word"],
        'y_tox': [0, 1],
        'severe_toxicity': [0.0, 0.1],
        'obscene': [0.0, 0.1],
        'threat': [0.0, 0.1],
        'insult': [0.0, 0.1],
        'identity_attack': [0.0, 0.1],
        'sexual_explicit': [0.0, 0.1],
        'male': [0.0, 0.0],
        'female': [0.0, 0.0],
        'black': [0.0, 0.0],
        'white': [0.0, 0.0],
        'muslim': [0.0, 0.0],
        'jewish': [0.0, 0.0],
        'christian': [0.0, 0.0],
        'homosexual_gay_or_lesbian': [0.0, 0.0],
        'psychiatric_or_mental_illness': [0.0, 0.0],
        'has_identity': [0, 0]
    })
    ds = ToxicityDataset(dummy_df, tokenizer)
    item = ds[0]
    print("Dataset item keys:", item.keys())

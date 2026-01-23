import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertCNNBiLSTM(nn.Module):
    """
    Baseline model: BERT + CNN + BiLSTM
    """
    def __init__(self, model_name="bert-base-uncased", num_classes=1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        embedding_dim = self.bert.config.hidden_size
        
        # CNN layer: extracting local features
        self.conv = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        
        # BiLSTM layer: extracting global sequential features
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Linear(128 * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, L, D)
        
        # CNN (requires (B, D, L))
        x = last_hidden_state.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)  # (B, L, 256)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, L, 256)
        
        # Global Max Pooling (common for this architecture)
        # pooled_out, _ = torch.max(lstm_out, dim=1)
        # Or just use the last hidden state of LSTM (usually the Bi-part)
        # but max pooling is more robust for toxicity
        x, _ = torch.max(lstm_out, dim=1)
        
        logits = self.classifier(self.dropout(x))
        return {"logits_tox": logits}

if __name__ == "__main__":
    model = BertCNNBiLSTM()
    print("Baseline model BERT+CNN+BiLSTM initialized.")
    dummy_ids = torch.randint(0, 100, (2, 32))
    dummy_mask = torch.ones((2, 32))
    out = model(dummy_ids, dummy_mask)
    print(f"Output shape: {out['logits_tox'].shape}")

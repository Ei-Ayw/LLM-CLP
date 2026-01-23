import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, last_hidden_state, attention_mask):
        # score shape: (batch_size, seq_len, 1)
        score = self.score_layer(last_hidden_state)
        
        # Masking: attention_mask is (batch_size, seq_len)
        # score should be masked before softmax
        if attention_mask is not None:
            # fill score with -1e9 where attention_mask is 0
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            
        weights = torch.softmax(score, dim=1)
        # weighted_sum shape: (batch_size, hidden_size)
        weighted_sum = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_sum

class DebertaToxicityMTL(nn.Module):
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path_or_name, local_files_only=True)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, local_files_only=True)
        self.use_attention_pooling = use_attention_pooling
        
        hidden_size = self.config.hidden_size
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(hidden_size)
            self.projection = nn.Linear(hidden_size * 2, 512)
        else:
            self.projection = nn.Linear(hidden_size, 512)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # Multi-task heads
        self.tox_head = nn.Linear(512, 1)
        self.sub_head = nn.Linear(512, num_subtypes)
        self.id_head = nn.Linear(512, num_identities)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # h_cls: [CLS] token (usually index 0 in DeBERTa)
        h_cls = last_hidden_state[:, 0, :]
        
        if self.use_attention_pooling:
            # h_att: Attention Pooling over all tokens
            h_att = self.pooling(last_hidden_state, attention_mask)
            # Concatenation
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls
            
        # Shared projection
        z = self.dropout(self.gelu(self.projection(h)))
        
        # Predict logits
        logits_tox = self.tox_head(z)
        logits_sub = self.sub_head(z)
        logits_id = self.id_head(z)
        
        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id": logits_id
        }

if __name__ == "__main__":
    # Test block
    model = DebertaToxicityMTL("microsoft/deberta-v3-base")
    print("Model initialized successfully.")
    
    # Dummy input
    dummy_input_ids = torch.randint(0, 100, (2, 32))
    dummy_mask = torch.ones((2, 32))
    
    outputs = model(dummy_input_ids, dummy_mask)
    print("Output shapes:")
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")

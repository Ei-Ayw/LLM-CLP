import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1)
            
    def forward(self, last_hidden_state, attention_mask):
        score = self.score_layer(last_hidden_state)
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        weights = torch.softmax(score, dim=1)
        weighted_sum = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_sum

class DebertaV3MTL(nn.Module):
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        try:
            self.config = DebertaV2Config.from_pretrained(model_path_or_name)
            # 移除 low_cpu_mem_usage=True 以解决 "Cannot copy out of meta tensor" 报错
            self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, use_safetensors=False)
        except Exception as e:
            print(f"\n❌ [CRITICAL] 无法加载 DeBERTa-MTL 权重: {e}")
            print(f"👉 提示: 请确保已运行 'python download_models.py' 且下载完整。")
            raise e

        # [Fix] 禁用 gradient checkpointing - 与 DDP 冲突
        # 如需单卡节省显存，可手动启用: self.deberta.gradient_checkpointing_enable()
        # self.deberta.gradient_checkpointing_enable()
        self.use_attention_pooling = use_attention_pooling
        hidden_size = self.config.hidden_size
        
        if self.use_attention_pooling:
            self.att_pooling = AttentionPooling(hidden_size)
            self.projection = nn.Linear(hidden_size * 2, 512)
        else:
            self.projection = nn.Linear(hidden_size, 512)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.tox_head = nn.Linear(512, 1)
        self.subtype_head = nn.Linear(512, num_subtypes)
        self.identity_head = nn.Linear(512, num_identities)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        h_cls = last_hidden_state[:, 0, :]
        if self.use_attention_pooling:
            h_att = self.att_pooling(last_hidden_state, attention_mask)
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls
        z = self.dropout(self.activation(self.projection(h)))
        logits_tox = self.tox_head(z)
        logits_sub = self.subtype_head(z)
        logits_id = self.identity_head(z)
        return {"logits_tox": logits_tox, "logits_sub": logits_sub, "logits_id": logits_id, "features": z}

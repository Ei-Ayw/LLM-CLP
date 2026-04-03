import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class AttentionPooling(nn.Module):
    """
    简单线性注意力池化: Linear(hidden, 1) → softmax → weighted_sum
    """
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
    """
    DeBERTa-v3 多任务学习模型 (简化架构)
    - 简单线性 AttentionPooling
    - 共享 projection layer (所有任务共用)
    - 单一 dropout
    """
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        try:
            self.config = DebertaV2Config.from_pretrained(model_path_or_name)
            self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, use_safetensors=False)
        except Exception as e:
            print(f"\n[CRITICAL] Failed to load DeBERTa weights: {e}")
            raise e

        self.use_attention_pooling = use_attention_pooling
        hidden_size = self.config.hidden_size

        if self.use_attention_pooling:
            self.att_pooling = AttentionPooling(hidden_size)
            pool_dim = hidden_size * 2
        else:
            pool_dim = hidden_size

        # 共享 projection: 所有任务共用同一个投影层
        self.projection = nn.Linear(pool_dim, 512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        # 任务头
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

        # 共享特征
        z = self.dropout(self.activation(self.projection(h)))

        logits_tox = self.tox_head(z)
        logits_sub = self.subtype_head(z)
        logits_id = self.identity_head(z)
        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id": logits_id,
            "features": z,
        }

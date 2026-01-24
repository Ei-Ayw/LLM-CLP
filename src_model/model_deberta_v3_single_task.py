import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

# =============================================================================
# ### 消融模型-2：DebertaV3_SingleTask ###
# 设计说明：
# 该模型移除了多任务（Subtype, Identity）的监督，仅训练 Toxicity 单任务。
# 保留本文提出的 Attention Pooling，旨在证明多任务联合学习对缓解偏见及提升性能的作用。
# =============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1)
    def forward(self, last_hidden_state, attention_mask):
        score = self.score_layer(last_hidden_state)
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        weights = torch.softmax(score, dim=1)
        return torch.sum(weights * last_hidden_state, dim=1)

class DebertaV3SingleTask(nn.Module):
    """
    消融实验模型：保留 Attention Pooling，但移除多任务模块。
    """
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path_or_name, local_files_only=True)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, local_files_only=True)
        
        hidden_size = self.config.hidden_size
        self.att_pooling = AttentionPooling(hidden_size)
        self.projection = nn.Linear(hidden_size * 2, 512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # 仅保留毒性分类头
        self.tox_head = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        h_cls = last_hidden_state[:, 0, :]
        h_att = self.att_pooling(last_hidden_state, attention_mask)
        h = torch.cat([h_cls, h_att], dim=-1)
        
        z = self.dropout(self.activation(self.projection(h)))
        return {"logits_tox": self.tox_head(z)}

if __name__ == "__main__":
    pass

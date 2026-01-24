import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

# =============================================================================
# ### 消融模型-1：DebertaV3_CLS_Only ###
# 设计说明：
# 该模型移除了本文提出的 Attention Pooling 层，回退到经典的 [CLS] 向量作为表示。
# 用于证明 Attention Pooling 在捕捉局部毒性特征上的优越性。
# =============================================================================
class DebertaV3CLSOnly(nn.Module):
    """
    消融实验模型：仅使用 CLS 池化，保留 MTL 头。
    """
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path_or_name, local_files_only=True)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, local_files_only=True)
        
        hidden_size = self.config.hidden_size
        
        # 仅使用 CLS 的线性映射
        self.projection = nn.Linear(hidden_size, 512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # 多任务头
        self.tox_head = nn.Linear(512, 1)
        self.subtype_head = nn.Linear(512, num_subtypes)
        self.identity_head = nn.Linear(512, num_identities)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # 仅提取 [CLS] 向量
        h = outputs.last_hidden_state[:, 0, :]
        
        z = self.dropout(self.activation(self.projection(h)))
        
        return {
            "logits_tox": self.tox_head(z),
            "logits_sub": self.subtype_head(z),
            "logits_id": self.identity_head(z)
        }

if __name__ == "__main__":
    pass

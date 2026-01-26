import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

# =============================================================================
# ### 原生 RoBERTa 包装器 ###
# 设计说明：
# 提供对原生 roberta-base 的标准化封装。
# =============================================================================
class VanillaRoBERTa(nn.Module):
    """
    原生 RoBERTa 序列分类模型包装类。
    """
    def __init__(self, model_path="roberta-base", num_labels=1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # RoBERTa 不使用 token_type_ids
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return {"logits_tox": out.logits}

if __name__ == "__main__":
    pass

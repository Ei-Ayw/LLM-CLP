import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

# =============================================================================
# ### 原生 BERT 包装器 ###
# 设计说明：
# 提供对原生 bert-base-uncased 的标准化封装，作为论文的基础对比组。
# =============================================================================
class VanillaBERT(nn.Module):
    """
    原生 BERT 序列分类模型包装类。
    """
    def __init__(self, model_path="bert-base-uncased", num_labels=1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return {"logits_tox": out.logits}

if __name__ == "__main__":
    # print("VanillaBERT initialized.")
    pass

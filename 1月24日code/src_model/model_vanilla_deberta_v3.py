import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

# =============================================================================
# ### 基准模型：VanillaDeBERTaV3 ###
# 设计说明：
# 提供对原生 microsoft/deberta-v3-base 的标准化封装。
# 不包含任何额外的池化层、多任务头或重加权逻辑，作为论文的最基础对比。
# =============================================================================
class VanillaDeBERTaV3(nn.Module):
    """
    原生 DeBERTa V3 序列分类模型包装类。
    """
    def __init__(self, model_path="microsoft/deberta-v3-base", num_labels=1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config, local_files_only=True)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return {"logits_tox": out.logits}

if __name__ == "__main__":
    pass

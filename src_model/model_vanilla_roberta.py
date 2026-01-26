import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
import os

class VanillaRoBERTa(nn.Module):
    def __init__(self, model_path="roberta-base", num_labels=1):
        super().__init__()
        try:
            self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                config=self.config,
                low_cpu_mem_usage=True # 优化大模型加载
            )
        except Exception as e:
            print(f"\n❌ [CRITICAL] 无法加载 RoBERTa 权重: {e}")
            print(f"👉 提示: 请确保已运行 'python download_models.py' 且下载完整。")
            raise e
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return {"logits_tox": out.logits}

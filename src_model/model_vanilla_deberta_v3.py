import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class VanillaDeBERTaV3(nn.Module):
    def __init__(self, model_path="microsoft/deberta-v3-base", num_labels=1):
        super().__init__()
        try:
            self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
            # 移除 low_cpu_mem_usage=True 以解决 "Cannot copy out of meta tensor" 报错
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                config=self.config,
                use_safetensors=True
            )
        except Exception as e:
            print(f"\n❌ [CRITICAL] 无法加载 DeBERTa-v3 权重: {e}")
            print(f"👉 提示: 请确保已运行 'python download_models.py' 且下载完整。")
            raise e
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return {"logits_tox": out.logits}

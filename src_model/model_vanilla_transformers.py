import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

# =============================================================================
# ### 原生 Transformer 包装器 ###
# 设计说明：
# 本模块提供了对常见预训练模型（BERT, RoBERTa, DeBERTa）的标准化包装。
# 所有的模型均不包含额外的池化层或多任务头，仅作为基准（Baseline）进行对比，
# 以验证本文提出的 Attention Pooling 和 MTL 架构的有效性。
# =============================================================================

class VanillaTransformer(nn.Module):
    """
    通用原生 Transformer 封装类。
    """
    def __init__(self, model_name_or_path, num_labels=1):
        super().__init__()
        # 使用 Transformers 官方的序列分类结构模型
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config, local_files_only=True)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 保持输出接口统一
        return {"logits_tox": out.logits}

# =============================================================================
# ### 具体子类定义 ###
# 为了在代码中提供更好的辨识度。
# =============================================================================

class VanillaBERT(VanillaTransformer):
    def __init__(self, model_path="bert-base-uncased"):
        super().__init__(model_path)

class VanillaRoBERTa(VanillaTransformer):
    def __init__(self, model_path="roberta-base"):
        super().__init__(model_path)

class VanillaDeBERTa(VanillaTransformer):
    def __init__(self, model_path="microsoft/deberta-v3-base"):
        super().__init__(model_path)

if __name__ == "__main__":
    # 测试代码
    print(">>> 正在初始化原生 Transformer 基准模型...")
    # 注意：运行此测试需要本地 pretrained_models 中存在对应权重
    try:
        model = VanillaBERT("bert-base-uncased")
        print("VanillaBERT 初始化成功。")
    except Exception as e:
        print(f"初始化跳过: {e}")

"""
=============================================================================
模型定义: DeBERTa-V3 + 特征输出 (用于对比学习)
在 VanillaDeBERTa 基础上增加 features 输出
=============================================================================
"""
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config


class DebertaV3CausalFair(nn.Module):
    """
    DeBERTa-V3 用于因果公平训练

    输出:
      - logits: (B, num_classes) 分类 logits
      - features: (B, hidden_size) CLS 特征向量 (用于对比学习)
    """

    def __init__(self, model_path="microsoft/deberta-v3-base", num_classes=2):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(model_path)
        hidden_size = self.config.hidden_size  # 768

        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 特征投影头 (对比学习用)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                return_features=True):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # 分类
        logits = self.classifier(self.dropout(cls_hidden))  # (B, num_classes)

        result = {"logits": logits}

        if return_features:
            # 投影到对比学习空间
            features = self.projector(cls_hidden)  # (B, 128)
            result["features"] = features
            result["cls_hidden"] = cls_hidden  # 原始 CLS (用于 CLP)

        return result

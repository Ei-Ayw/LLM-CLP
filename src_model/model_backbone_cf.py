"""
=============================================================================
Backbone-agnostic CausalFair 模型
支持 BERT / RoBERTa / DeBERTa-v3 三种骨干网络

L_total = L_CE + λ_clp × L_CLP + λ_con × L_SupCon

用法:
  model = BackboneCausalFair(model_path, backbone_type='bert', num_classes=2)
=============================================================================
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BackboneCausalFair(nn.Module):
    """
    通用骨干 CausalFair 模型

    输出:
      - logits:    (B, num_classes)  分类 logits
      - features:  (B, 128)          投影到对比学习空间的特征 (可选)
      - cls_hidden:(B, hidden_size)  原始 CLS 向量 (用于 CLP)
    """

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size

        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 投影头 (对比学习)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, input_ids, attention_mask,
                token_type_ids=None, return_features=True):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # (B, H)

        logits = self.classifier(self.dropout(cls_hidden))
        result = {"logits": logits}

        if return_features:
            result["features"] = self.projector(cls_hidden)  # (B, 128)
            result["cls_hidden"] = cls_hidden

        return result

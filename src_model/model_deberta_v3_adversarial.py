import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import DebertaV2Model, DebertaV2Config

from model_deberta_v3_mtl import AttentionPooling


class GradientReversalFunction(Function):
    """Ganin et al. 2016: 前向不变，反向梯度 × (-lambda_)"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


class DebertaV3Adversarial(nn.Module):
    """
    DeBERTa-v3 对抗去偏模型

    架构:
      DeBERTa Backbone → CLS + AttentionPooling → Projection(1536→512) → z
                                                                          |
                                                               ┌──────────┼──────────┐
                                                               │          │          │
                                                           tox_head    GRL(λ)   subtype_head
                                                           512→1     (梯度反转)   512→6
                                                                        │
                                                                    adv_discriminator
                                                                    512→512→256→9

    GRL: 前向不变，反向 × (-λ)。backbone 被迫学到无法预测身份的表示。
    条件 GRL: 只对非毒样本反转梯度 (训练脚本控制)。
    """

    def __init__(self, model_path_or_name="microsoft/deberta-v3-base",
                 num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        try:
            self.config = DebertaV2Config.from_pretrained(model_path_or_name)
            self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, use_safetensors=False)
        except Exception as e:
            print(f"\n[CRITICAL] Failed to load DeBERTa weights: {e}")
            raise e

        self.use_attention_pooling = use_attention_pooling
        hidden_size = self.config.hidden_size

        if self.use_attention_pooling:
            self.att_pooling = AttentionPooling(hidden_size)
            pool_dim = hidden_size * 2
        else:
            pool_dim = hidden_size

        # 共享 projection
        self.projection = nn.Linear(pool_dim, 512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        # 任务头: toxicity + subtype (保留辅助正则化)
        self.tox_head = nn.Linear(512, 1)
        self.subtype_head = nn.Linear(512, num_subtypes)

        # 对抗头: GRL + 强化 discriminator (3层)
        self.grl = GradientReversalLayer()
        self.adv_discriminator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_identities),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, lambda_adv=0.0):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state
        h_cls = last_hidden_state[:, 0, :]

        if self.use_attention_pooling:
            h_att = self.att_pooling(last_hidden_state, attention_mask)
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls

        # 共享特征
        z = self.dropout(self.activation(self.projection(h)))

        logits_tox = self.tox_head(z)
        logits_sub = self.subtype_head(z)

        # 对抗分支: GRL 反转梯度
        z_rev = self.grl(z, lambda_adv)
        logits_id_adv = self.adv_discriminator(z_rev)

        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id_adv": logits_id_adv,
            "features": z,
        }

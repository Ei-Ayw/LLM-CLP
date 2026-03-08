import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import DebertaV2Model, DebertaV2Config

from model_deberta_v3_mtl import AttentionPooling


# =============================================================================
# Gradient Reversal Layer (Ganin et al. 2016)
# =============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# =============================================================================
# 身份属性分层定义
# =============================================================================
# 9 个 specific identities → 5 个 coarse groups
IDENTITY_COLS = [
    'male', 'female', 'homosexual_gay_or_lesbian',
    'christian', 'jewish', 'muslim',
    'black', 'white',
    'psychiatric_or_mental_illness'
]

COARSE_GROUPS = {
    'gender': [0, 1],               # male, female
    'sexual_orientation': [2],       # homosexual_gay_or_lesbian
    'religion': [3, 4, 5],          # christian, jewish, muslim
    'race': [6, 7],                 # black, white
    'disability': [8],              # psychiatric_or_mental_illness
}
NUM_COARSE = len(COARSE_GROUPS)  # 5


class DebertaV3HCMA(nn.Module):
    """
    Hierarchical Conditional Metric-Aligned Debiasing Model

    架构:
      DeBERTa Backbone → CLS + AttentionPooling → Projection(1536→512) → z
                                                                          │
                                              ┌───────────┬───────────┬───┴───────────┐
                                              │           │           │               │
                                          tox_head   sub_head   id_exist_head    GRL(λ)
                                          512→1      512→6      512→1           (梯度反转)
                                                                                    │
                                                                ┌───────────────────┤
                                                                │                   │
                                                          id_coarse_head     adv_specific
                                                            512→5            512→512→256→9
                                                         (no GRL,保留)     (GRL,对抗压制)

    分层条件对抗:
      - Level 1 (existence): 是否提及任何身份 → 保留
      - Level 2 (coarse): 粗粒度类别(性别/宗教/种族/性取向/残障) → 保留
      - Level 3 (specific): 具体身份(muslim/christian/black/white等) → GRL 对抗压制
    """

    def __init__(self, model_path_or_name="microsoft/deberta-v3-base",
                 num_subtypes=6, num_identities=9):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path_or_name)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, use_safetensors=False)

        hidden_size = self.config.hidden_size
        self.att_pooling = AttentionPooling(hidden_size)
        pool_dim = hidden_size * 2  # CLS + AttentionPooling

        # 共享 projection
        self.projection = nn.Linear(pool_dim, 512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        # === 任务头 ===
        # 1. 主任务: 毒性预测
        self.tox_head = nn.Linear(512, 1)
        # 2. 辅助任务: 6 个子类型
        self.subtype_head = nn.Linear(512, num_subtypes)

        # === 分层身份头 ===
        # Level 1: 身份存在性 (是否提及任何身份) → 正常训练，保留
        self.id_exist_head = nn.Linear(512, 1)
        # Level 2: 粗粒度类别 (5组) → 正常训练，保留
        self.id_coarse_head = nn.Linear(512, NUM_COARSE)

        # Level 3: 具体身份 (9个) → 通过 GRL 对抗压制
        self.grl = GradientReversalLayer()
        self.adv_specific = nn.Sequential(
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
        h_att = self.att_pooling(last_hidden_state, attention_mask)
        h = torch.cat([h_cls, h_att], dim=-1)

        # 共享特征
        z = self.dropout(self.activation(self.projection(h)))

        # 主任务 + 辅助
        logits_tox = self.tox_head(z)
        logits_sub = self.subtype_head(z)

        # 分层身份头
        logits_id_exist = self.id_exist_head(z)           # 保留
        logits_id_coarse = self.id_coarse_head(z)         # 保留
        z_rev = self.grl(z, lambda_adv)
        logits_id_specific = self.adv_specific(z_rev)     # 对抗压制

        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id_exist": logits_id_exist,
            "logits_id_coarse": logits_id_coarse,
            "logits_id_specific": logits_id_specific,
            "features": z,
        }

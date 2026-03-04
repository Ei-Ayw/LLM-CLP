"""
=============================================================================
IACD: Identity-Agnostic Contrastive Debiasing
File 1/3: Model + Loss
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2Config


# =============================================================================
# Model: DebertaV3IACD
# =============================================================================
class DebertaV3IACD(nn.Module):
    """
    DeBERTa-v3 + Identity-Agnostic Contrastive Debiasing

    架构极简化:
      - [CLS] token (768-dim) 直出，无 Attention Pooling，无信息瓶颈
      - 分类头: Dropout → Linear(768, 1) → toxicity logit
      - 对比投影头: MLP(768→256→128) → L2Norm → z (仅训练时使用)
    """
    def __init__(self, model_path="microsoft/deberta-v3-base"):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_path)
        self.deberta = DebertaV2Model.from_pretrained(
            model_path, use_safetensors=False
        )
        hidden = self.config.hidden_size  # 768

        # 分类头: 极简设计，不压缩维度
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )

        # 对比投影头: MLP → L2Norm (训练时使用，推理时丢弃)
        self.projector = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                return_proj=False):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        h = outputs.last_hidden_state[:, 0]  # [CLS] token, (B, 768)

        logits = self.classifier(h)  # (B, 1)

        out = {"logits_tox": logits, "features": h}

        if return_proj:
            z = F.normalize(self.projector(h), dim=-1)  # (B, 128), L2-normed
            out["z"] = z

        return out


# =============================================================================
# Loss: FairSupConLoss
# =============================================================================
class FairSupConLoss(nn.Module):
    """
    Identity-Agnostic Supervised Contrastive Loss (核心创新)

    在标准 SupCon 基础上, 对跨身份正样本对施加额外权重 (1 + lambda),
    迫使模型在表示空间中对齐 "同毒性-不同身份" 的样本.

    数学公式:
      L_i = -1/|P(i)| * sum_{j in P(i)} w_ij * log[ exp(z_i·z_j/τ) / sum_{k≠i} exp(z_i·z_k/τ) ]

      其中:
        P(i) = {j : y_j == y_i, j ≠ i}  (同毒性类别的正样本)
        w_ij = 1 + λ * 1[identity(i) ≠ identity(j)]  (跨身份加权)

    Args:
        temperature: 对比学习温度参数 τ (default: 0.07)
        cross_weight: 跨身份正样本对的额外权重 λ (default: 0.5)
    """
    def __init__(self, temperature=0.07, cross_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_weight = cross_weight

    def forward(self, z, y_bin, identity_group):
        """
        Args:
            z: L2-normalized embeddings [B, D]
            y_bin: binary toxicity labels [B] (0 or 1)
            identity_group: identity membership [B, 9] (binary, >=0.5 means present)
        Returns:
            loss: scalar
        """
        device = z.device
        B = z.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 相似度矩阵 [B, B], 除以温度
        sim = torch.mm(z, z.t()) / self.temperature

        # 排除自身的 mask
        self_mask = 1.0 - torch.eye(B, device=device)

        # 正样本 mask: 同一毒性类别 (排除自身)
        pos_mask = (y_bin.unsqueeze(0) == y_bin.unsqueeze(1)).float() * self_mask

        # 跨身份 mask: 两个样本的身份组构成不同
        has_id = (identity_group.sum(dim=1) > 0)  # [B]
        # 至少一方有身份标签
        either_has_id = (has_id.unsqueeze(0) | has_id.unsqueeze(1)).float()
        # 身份向量余弦相似度
        id_float = identity_group.float()
        id_norm = id_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
        id_cos = torch.mm(id_float, id_float.t()) / (id_norm * id_norm.t())
        # 身份构成不同 (余弦 < 0.99) 且至少一方有身份标签
        cross_id_mask = ((id_cos < 0.99) * either_has_id).float()

        # 加权矩阵: 跨身份正样本获得额外权重
        weights = torch.ones(B, B, device=device)
        weights = weights + self.cross_weight * cross_id_mask

        # Numerical stability: 减去每行最大值
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        sim = sim - logits_max

        # log_prob = sim_ij - log(sum_{k≠i} exp(sim_ik))
        exp_sim = torch.exp(sim) * self_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 对每个 anchor, 在正样本上求加权平均
        weighted_pos = weights * pos_mask * log_prob
        n_pos = pos_mask.sum(dim=1).clamp(min=1)

        loss = -(weighted_pos.sum(dim=1) / n_pos).mean()
        return loss

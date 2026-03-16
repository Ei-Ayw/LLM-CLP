"""
=============================================================================
对比学习损失函数
- CounterfactualSupConLoss: 反事实配对的对比学习
- CLP: Counterfactual Logit Pairing
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualSupConLoss(nn.Module):
    """
    反事实监督对比学习损失

    核心思想:
      - 原始样本和它的反事实是正对 (拉近) → 实现身份不变性
      - 同标签的样本也是正对 (拉近) → 保持判别力
      - 不同标签的样本是负对 (推远)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_orig, z_cf, labels):
        """
        Args:
            z_orig:  (B, D) 原始样本的特征
            z_cf:    (B, D) 反事实样本的特征
            labels:  (B,)   类别标签

        Returns:
            scalar loss
        """
        B = z_orig.size(0)
        device = z_orig.device

        # 拼接: [orig_0, ..., orig_B, cf_0, ..., cf_B] → (2B, D)
        z_all = torch.cat([z_orig, z_cf], dim=0)
        z_all = F.normalize(z_all, dim=1)

        # 相似度矩阵: (2B, 2B)
        sim = torch.mm(z_all, z_all.t()) / self.temperature

        # 正对 mask: (2B, 2B)
        labels_all = labels.repeat(2)  # (2B,)
        label_match = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1))  # 同标签

        # 原始-反事实配对 (始终是正对)
        cf_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=device)
        for i in range(B):
            cf_mask[i, B + i] = True    # orig_i ↔ cf_i
            cf_mask[B + i, i] = True    # cf_i ↔ orig_i

        positive_mask = (label_match | cf_mask).float()

        # 去掉自身
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        positive_mask = positive_mask * (~self_mask).float()

        # 分母: 所有非自身的 exp(sim)
        exp_sim = torch.exp(sim) * (~self_mask).float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 分子: 正对的 log_prob 均值
        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(positive_mask * log_prob).sum(dim=1) / pos_count

        return loss.mean()


class CounterfactualLogitPairing(nn.Module):
    """
    Counterfactual Logit Pairing (CLP)
    强制模型对原始样本和反事实样本输出相同的 logits

    L_CLP = ||logits(x) - logits(x_cf)||²
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits_orig, logits_cf):
        """
        Args:
            logits_orig: (B, C) 原始样本 logits
            logits_cf:   (B, C) 反事实样本 logits

        Returns:
            scalar loss
        """
        return F.mse_loss(logits_orig, logits_cf)

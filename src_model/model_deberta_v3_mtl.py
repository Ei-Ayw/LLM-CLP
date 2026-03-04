import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class AttentionPooling(nn.Module):
    """
    改进的注意力池化：加入非线性变换提升表达能力
    旧版: Linear(hidden, 1) → softmax → weighted_sum
    新版: Linear(hidden, hidden//4) → Tanh → Linear(hidden//4, 1) → softmax → weighted_sum
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        score = self.score_layer(last_hidden_state)
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        weights = torch.softmax(score, dim=1)
        weighted_sum = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_sum

class DebertaV3MTL(nn.Module):
    """
    DeBERTa-v3 多任务学习模型
    改进点:
    1. AttentionPooling 加入非线性 (Tanh bottleneck)
    2. 主任务 (toxicity) 和辅助任务 (subtype/identity) 使用独立的 projection head
       → 避免梯度冲突 (身份识别 vs 毒性判断 目标矛盾)
    3. Uncertainty Weighting: 可学习的多任务权重参数
    """
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
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

        # 三任务完全隔离: 各自独立的 projection + dropout + head
        # 避免任何梯度交叉污染
        self.proj_tox = nn.Linear(pool_dim, 512)
        self.proj_sub = nn.Linear(pool_dim, 512)
        self.proj_id  = nn.Linear(pool_dim, 512)

        self.activation = nn.GELU()
        self.drop_tox = nn.Dropout(0.2)
        self.drop_sub = nn.Dropout(0.1)   # 辅助任务用较低dropout，防止欠拟合
        self.drop_id  = nn.Dropout(0.1)

        # 任务头
        self.tox_head = nn.Linear(512, 1)
        self.subtype_head = nn.Linear(512, num_subtypes)
        self.identity_head = nn.Linear(512, num_identities)

        # Uncertainty Weighting (Kendall et al., 2018)
        # log_var 越大 → 该任务权重越小 (不确定性高的任务降权)
        self.log_var_tox = nn.Parameter(torch.zeros(1))
        self.log_var_sub = nn.Parameter(torch.zeros(1))
        self.log_var_id = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        h_cls = last_hidden_state[:, 0, :]
        if self.use_attention_pooling:
            h_att = self.att_pooling(last_hidden_state, attention_mask)
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls

        # 主任务特征
        z_tox = self.drop_tox(self.activation(self.proj_tox(h)))
        # 辅助任务特征 — 完全物理隔离，各走各的
        z_sub = self.drop_sub(self.activation(self.proj_sub(h)))
        z_id  = self.drop_id(self.activation(self.proj_id(h)))

        logits_tox = self.tox_head(z_tox)
        logits_sub = self.subtype_head(z_sub)
        logits_id = self.identity_head(z_id)
        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id": logits_id,
            "features": z_tox,
            # +0.0 创建计算节点，避免 DDP "marked ready twice" 错误
            "log_vars": (self.log_var_tox + 0.0, self.log_var_sub + 0.0, self.log_var_id + 0.0)
        }

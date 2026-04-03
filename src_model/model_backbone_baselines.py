"""
=============================================================================
Backbone-agnostic baseline model classes
支持 BERT / RoBERTa / DeBERTa-v3 三种骨干网络

包含:
  - BackboneVanilla        (Vanilla 基线)
  - BackboneEAR            (Entropy-based Attention Regularisation)
  - BackboneGetFair        (Gradient Fairness)
  - BackboneCCDF           (Counterfactual Causal Debiasing Framework)
  - BackboneDavani         (Logit Pairing)
  - BackboneRamponi        (Adversarial Debiasing via GRL)
  - BiasOnlyModel          (CCDF 偏置模型, backbone无关)

用法:
  model = BackboneVanilla(model_path, backbone_type='bert', num_classes=2)
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# ---------------------------------------------------------------------------
# 工具: 从 backbone 输出中提取 CLS 向量
# ---------------------------------------------------------------------------
def get_cls(outputs):
    """从 transformers AutoModel 输出取 CLS (last_hidden_state[:, 0, :])"""
    return outputs.last_hidden_state[:, 0, :]


# ---------------------------------------------------------------------------
# 1. Vanilla
# ---------------------------------------------------------------------------
class BackboneVanilla(nn.Module):
    """标准微调, 无去偏"""

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = get_cls(outputs)
        logits = self.classifier(self.dropout(cls))
        return {"logits": logits}


# ---------------------------------------------------------------------------
# 2. EAR (Entropy-based Attention Regularisation)
# ---------------------------------------------------------------------------
class BackboneEAR(nn.Module):
    """EAR: 在最后一层注意力熵上施加正则化"""

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        # 需要输出注意力
        self.backbone = AutoModel.from_pretrained(
            model_path, output_attentions=True
        )
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask,
                token_type_ids=None, return_attentions=False):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = get_cls(outputs)
        logits = self.classifier(self.dropout(cls))
        result = {"logits": logits}
        if return_attentions and outputs.attentions:
            result["attentions"] = outputs.attentions
        return result


def attention_entropy_loss(attentions, attention_mask):
    """计算最后一层注意力的负熵 (最小化 = 最大化熵 = 更均匀分布)"""
    last_attn = attentions[-1]  # (B, H, L, L)
    # 只看 [CLS] 行 (第 0 行)
    cls_attn = last_attn[:, :, 0, :]  # (B, H, L)
    # 对 padding 位置置 0 (保证 softmax 只在真实 token 上)
    mask_f = attention_mask.unsqueeze(1).float()  # (B, 1, L)
    cls_attn = cls_attn * mask_f
    # 归一化
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)
    # 熵: -sum(p log p)
    entropy = -(cls_attn * (cls_attn + 1e-9).log()).sum(dim=-1)  # (B, H)
    return -entropy.mean()  # 最小化负熵 = 最大化熵


# ---------------------------------------------------------------------------
# 3. GetFair (Gradient Fairness)
# ---------------------------------------------------------------------------
IDENTITY_TOKENS = {
    'black', 'white', 'asian', 'hispanic', 'african', 'european',
    'muslim', 'christian', 'jewish', 'islam', 'islamic', 'mosque', 'church',
    'quran', 'bible', 'hijab',
    'women', 'men', 'woman', 'man', 'she', 'he', 'her', 'his',
    'gay', 'lesbian', 'homosexual', 'lgbtq', 'queer', 'trans',
    'disabled', 'immigrant', 'refugee',
    'arab', 'chinese', 'indian', 'mexican', 'hindu', 'buddhist',
    'dalits', 'roma',
}


def build_identity_mask(input_ids, tokenizer):
    """构建身份词 token mask: 1=身份词, 0=其他"""
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    for i in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        for j, token in enumerate(tokens):
            clean = token.replace('▁', '').replace('Ġ', '').replace('##', '').lower()
            if clean in IDENTITY_TOKENS:
                mask[i, j] = 1.0
    return mask


class BackboneGetFair(nn.Module):
    """GetFair: 梯度公平约束"""

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._embed_output = None
        self._embed_grad = None

    def _get_word_embeddings(self):
        """获取 word_embeddings 层 (BERT/RoBERTa/DeBERTa 命名一致)"""
        return self.backbone.embeddings.word_embeddings

    def _save_embed(self, module, input, output):
        self._embed_output = output

    def forward(self, input_ids, attention_mask,
                token_type_ids=None, capture_embed_grad=False):
        if capture_embed_grad:
            hook = self._get_word_embeddings().register_forward_hook(self._save_embed)

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)

        if capture_embed_grad:
            hook.remove()
            if self._embed_output is not None and self._embed_output.requires_grad:
                self._embed_output.register_hook(lambda g: setattr(self, '_embed_grad', g))

        cls = get_cls(outputs)
        logits = self.classifier(self.dropout(cls))
        return {"logits": logits}


# ---------------------------------------------------------------------------
# 4. CCDF (Counterfactual Causal Debiasing Framework)
# ---------------------------------------------------------------------------
class BiasOnlyModel(nn.Module):
    """CCDF 的偏置模型 (词袋级，与 backbone 无关)"""

    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pool = lambda x: x.mean(dim=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, identity_mask):
        """
        identity_mask: (B, L) float, 1=身份词, 0=other
        """
        emb = self.embed(input_ids)                 # (B, L, D)
        # 只保留身份词 embedding
        masked = emb * identity_mask.unsqueeze(-1)  # (B, L, D)
        denom = identity_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        pooled = masked.sum(dim=1) / denom.squeeze(-1)  # (B, D)
        return self.fc(pooled)


class BackboneCCDF(nn.Module):
    """CCDF 主模型 (TDE 两阶段, backbone 可换)"""

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = get_cls(outputs)
        logits = self.classifier(self.dropout(cls))
        return {"logits": logits}


# ---------------------------------------------------------------------------
# 5. Davani (Logit Pairing)
# ---------------------------------------------------------------------------
class BackboneDavani(nn.Module):
    """Davani: Logit Pairing — 原始和反事实样本输出相同 logits"""

    def __init__(self, model_path, backbone_type='bert', num_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = get_cls(outputs)
        logits = self.classifier(self.dropout(cls))
        return {"logits": logits}


# ---------------------------------------------------------------------------
# 6. Ramponi (Adversarial Debiasing via GRL)
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class BackboneRamponi(nn.Module):
    """Ramponi: 对抗去偏 (GRL)"""

    def __init__(self, model_path, backbone_type='bert',
                 num_classes=2, num_identity_classes=2):
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.adv_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_identity_classes),
        )

    def forward(self, input_ids, attention_mask,
                token_type_ids=None, alpha=1.0):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.backbone_type == 'bert':
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls = get_cls(outputs)

        main_logits = self.classifier(self.dropout(cls))
        rev_cls = GradientReversalFunction.apply(cls, alpha)
        adv_logits = self.adv_classifier(rev_cls)

        return {"logits": main_logits, "adv_logits": adv_logits}

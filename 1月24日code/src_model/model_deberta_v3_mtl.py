import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

# =============================================================================
# ### 核心模块 1：AttentionPooling (注意力池化层) ###
# 设计意图：
# 传统的 Transformer 模型通常只使用 [CLS] 向量作为全局句表示。
# 注意力池化允许模型根据上下文自动预测每个 Token 的重要性权重，
# 从而更敏锐地捕捉到那些分布不均的“毒性关键词”（如侮辱性词汇）。
# =============================================================================
class AttentionPooling(nn.Module):
    """
    通过学习一个权重层来对隐藏层状态进行加权平均，替代简单的 Mean 或 Max Pooling。
    """
    def __init__(self, hidden_size):
        super().__init__()
        # 权重映射层，将每个 token 压缩到一个标量得分
        self.score_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, last_hidden_state, attention_mask):
        # 1. 计算 token 原始得分，形状: (batch_size, seq_len, 1)
        score = self.score_layer(last_hidden_state)
        
        # 2. Mask 处理：屏蔽填充（Padding）符号，防止其干扰权重分布
        if attention_mask is not None:
            score = score.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            
        # 3. Softmax 归一化，得到每个 token 的注意力权重 (总和为1)
        weights = torch.softmax(score, dim=1)
        
        # 4. 加权求和得到句子表示，形状: (batch_size, hidden_size)
        weighted_sum = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_sum

# =============================================================================
# ### 主模型：DebertaV3MTL (多任务学习增强版) ###
# 设计说明：
# 本模型基于 DeBERTa-v3-base，结合了注意力池化机制和多任务监督头。
# 多任务设计（毒性、子类、身份）是本论文缓解模型偏见的核心。
# =============================================================================
class DebertaV3MTL(nn.Module):
    """
    DeBERTa V3 Multi-Task Learning 模型，用于毒性分类及偏见缓解。
    """
    def __init__(self, model_path_or_name="microsoft/deberta-v3-base", num_subtypes=6, num_identities=9, use_attention_pooling=True):
        super().__init__()
        # 加载预训练配置与权重
        self.config = DebertaV2Config.from_pretrained(model_path_or_name, local_files_only=True)
        self.deberta = DebertaV2Model.from_pretrained(model_path_or_name, local_files_only=True)
        
        # =====================================================================
        # 梯度检查点 (Gradient Checkpointing) - 针对 3090 24G 优化
        # 原理：用计算换显存，不保存中间激活值而是重新计算
        # 效果：显存降低约 50%，允许 batch_size 从 16 提升到 32
        # =====================================================================
        self.deberta.gradient_checkpointing_enable()
        
        self.use_attention_pooling = use_attention_pooling
        
        hidden_size = self.config.hidden_size
        
        # ### 特征提取策略 ###
        # 将 [CLS] 向量（全局语义）与 Attention Pooling 向量（关键局部特征）拼接。
        if self.use_attention_pooling:
            self.att_pooling = AttentionPooling(hidden_size)
            self.projection = nn.Linear(hidden_size * 2, 512) # 拼接后维度：768 * 2 = 1536
        else:
            self.projection = nn.Linear(hidden_size, 512)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # ### 多任务解耦头 (Decoupled Task Heads) ###
        # 1. 核心任务：毒性二分类 (Binary Toxicity)
        self.tox_head = nn.Linear(512, 1)
        # 2. 辅助任务：细分类别预测 (Severe Toxicity, Obscene, Threat, etc.)
        self.subtype_head = nn.Linear(512, num_subtypes)
        # 3. 辅助任务：身份词辨识 (Identity Group Prediction)
        self.identity_head = nn.Linear(512, num_identities)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. Backbone 前向传播
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 2. 提取特征
        h_cls = last_hidden_state[:, 0, :] # 经典 [CLS] 向量
        
        if self.use_attention_pooling:
            h_att = self.att_pooling(last_hidden_state, attention_mask)
            # 全局 + 局部特征融合
            h = torch.cat([h_cls, h_att], dim=-1)
        else:
            h = h_cls
            
        # 3. 共享特征映射与正则化
        z = self.dropout(self.activation(self.projection(h)))
        
        # 4. 多任务预测输出
        logits_tox = self.tox_head(z)
        logits_sub = self.subtype_head(z)
        logits_id = self.identity_head(z)
        
        return {
            "logits_tox": logits_tox,
            "logits_sub": logits_sub,
            "logits_id": logits_id,
            "features": z  # 用于可视化分析的中间特征
        }

if __name__ == "__main__":
    # 简单的结构测试
    test_model = DebertaV3MTL("microsoft/deberta-v3-base")
    print(">>> DebertaV3MTL 模型初始化完成。")
    
    mock_ids = torch.randint(0, 1000, (2, 64))
    mock_mask = torch.ones((2, 64))
    
    out = test_model(mock_ids, mock_mask)
    print(">>> 各任务输出张量形状：")
    for key, val in out.items():
        print(f"  - {key}: {list(val.shape)}")
